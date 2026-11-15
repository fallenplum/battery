"""Read released MATR HDF5 records with explicit author-generated coordinates.

The author regenerates summary.cycle as1:k. Instrument Cycle_Index is absent
from these archives. Summary and waveform positions remain separate and the
reader never certifies the original instrument cycle coordinate.
"""

from __future__ import annotations

import hashlib
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np


CONTINUATIONS = {"b2c7": "b1c0", "b2c8": "b1c1", "b2c9": "b1c2",
                 "b2c15": "b1c3", "b2c16": "b1c4"}
PREDECESSORS = {value: key for key, value in CONTINUATIONS.items()}
BATCH_DATES = {"2017-05-12": 1, "2017-06-30": 2,
               "2018-04-12": 3, "2019-01-24": 4}
RAW_FIELDS = ("V", "I", "Qd", "Qc", "t", "T")


def source_batch_number(path):
    name = Path(path).name
    for date, number in BATCH_DATES.items():
        if date in name or date.replace("-", "") in name:
            return number
    raise ValueError("Supply batch_number for a source filename without its date")


def reference_at(dataset, index):
    return dataset[()].reshape(-1)[index]


def source_groups(handle, index):
    batch = handle["batch"]
    count = int(batch["cycles"].size)
    if index < 0 or index >= count:
        raise IndexError("Source cell index lies outside the original batch")
    return batch, handle[reference_at(batch["summary"], index)], handle[reference_at(batch["cycles"], index)]


def summary_arrays(summary):
    return {name: np.asarray(dataset[()]).reshape(-1) for name, dataset in summary.items()}


def integer_cycle(value):
    number = float(value)
    return int(number) if np.isfinite(number) and number == round(number) else None


def readable_policy(handle, batch, index):
    codes = handle[reference_at(batch["policy_readable"], index)][()].reshape(-1)
    return "".join(chr(int(value)) for value in codes)


def static_source_field(handle, batch, field, index):
    if field not in batch:
        return None
    dataset = handle[reference_at(batch[field], index)]
    values = np.asarray(dataset[()]).reshape(-1)
    if dataset.attrs.get("MATLAB_class") == b"char":
        return "".join(chr(int(value)) for value in values)
    if dataset.attrs.get("MATLAB_class") == b"string":
        return mcos_scalar_string(handle, values)
    return values.item() if len(values) == 1 else values.tolist()


def mcos_scalar_string(handle, object_header):
    """Resolve the named save-object heap property, then decode scalar UTF16."""
    header = np.asarray(object_header, dtype=np.uint32).reshape(-1)
    if header.tolist()[:4] != [3707764736, 2, 1, 1] or len(header) != 6:
        raise ValueError("Source identity is not a supported scalar MCOS string")
    references = handle["#subsystem#/MCOS"][()].reshape(-1)
    blob = handle[references[0]][()].reshape(-1).tobytes()
    words = np.frombuffer(blob, dtype="<u4")
    if len(words) < 10 or words[0] != 2:
        raise ValueError("Unsupported MCOS metadata version")
    names = [""] + blob[40:int(words[2])].decode("utf-8").split("\x00")
    classes = np.frombuffer(blob[int(words[2]):int(words[3])], dtype="<u4").reshape(-1, 4)
    objects = np.frombuffer(blob[int(words[4]):int(words[5])], dtype="<u4").reshape(-1, 6)
    object_id, class_id = int(header[4]), int(header[5])
    if int(objects[object_id, 0]) != class_id or names[int(classes[class_id, 1])] != "string":
        raise ValueError("MCOS identity class binding differs from its object header")
    properties = []
    cursor, end = int(words[3]), int(words[4])
    while cursor < end:
        count = int(words[cursor // 4])
        cursor += 4
        stop = cursor + count * 12
        if stop > end:
            raise ValueError("MCOS saved property block exceeds its metadata interval")
        properties.append(np.frombuffer(blob[cursor:stop], dtype="<u4").reshape(-1, 3))
        cursor = (stop + 7) // 8 * 8
    saved = properties[int(objects[object_id, 3])]
    if saved.shape != (1, 3) or names[int(saved[0, 0])] != "any" or int(saved[0, 1]) != 1:
        raise ValueError("Unsupported MCOS scalar-string saved-property representation")
    payload = np.asarray(handle[references[2 + int(saved[0, 2])]][()]).reshape(-1)
    if payload.dtype != np.dtype("uint64") or payload.tolist()[:4] != [1, 2, 1, 1]:
        raise ValueError("Unsupported scalar-string payload header")
    length = int(payload[4])
    text = payload[5:].astype("<u8").tobytes().decode("utf-16-le")
    if length > len(text):
        raise ValueError("MCOS source identity string is truncated")
    return text[:length]


def cell_metadata(path, index, batch_number=None, source_handle=None):
    path = Path(path)
    number = source_batch_number(path) if batch_number is None else int(batch_number)
    identifier = f"b{number}c{index}"
    context = nullcontext(source_handle) if source_handle is not None else h5py.File(path, "r")
    with context as handle:
        batch, summary, waveforms = source_groups(handle, index)
        arrays = summary_arrays(summary)
        cycles = arrays["cycle"]
        field_counts = {name: int(dataset.size) for name, dataset in waveforms.items()}
        raw_count = field_counts["I"]
        required_counts = [field_counts.get(name) for name in ("V", "I", "Qd", "Qc", "t")]
        count_aligned = all(count == raw_count for count in required_counts) and len(cycles) == len(arrays["QDischarge"]) == raw_count
        original_integer = all(integer_cycle(value) is not None for value in cycles)
        increasing = bool(np.isfinite(cycles).all() and (np.diff(cycles) > 0).all())
        barcode = static_source_field(handle, batch, "barcode", index)
        channel = static_source_field(handle, batch, "channel_id", index)
        failures = []
        if not count_aligned:
            failures.append("SUMMARY_WAVEFORM_RECORD_COUNTS_DIFFER")
        if not original_integer:
            failures.append("SUMMARY_CYCLE_NOT_FINITE_INTEGER")
        if not increasing:
            failures.append("SUMMARY_CYCLE_NOT_STRICTLY_INCREASING")
        meta = {"source_file_name": path.name, "source_batch": number,
                "source_cell_index": index, "source_cell_id": identifier,
                "source_segment_id": identifier,
                "cell_uid": "MATR_" + CONTINUATIONS.get(identifier, identifier),
                "continuation_of": CONTINUATIONS.get(identifier),
                "continuation_segment": PREDECESSORS.get(identifier),
                "source_summary_group": summary.name, "source_waveform_group": waveforms.name,
                "source_barcode": barcode,
                "source_barcode_sha256": hashlib.sha256(str(barcode).encode()).hexdigest() if barcode is not None else None,
                "source_barcode_normalized_sha256": hashlib.sha256(str(barcode).strip().upper().encode()).hexdigest() if barcode is not None else None,
                "source_barcode_normalization": "strip surrounding whitespace and uppercase ASCII barcode",
                "source_channel_id": channel,
                "policy_original": readable_policy(handle, batch, index),
                "summary_fields": list(arrays),
                "summary_field_lengths": {name: len(values) for name, values in arrays.items()},
                "waveform_field_record_counts": field_counts, "raw_records": raw_count,
                "summary_sequence_first": integer_cycle(cycles[0]) if len(cycles) else None,
                "summary_sequence_last": integer_cycle(cycles[-1]) if len(cycles) else None,
                "summary_sequence_max": integer_cycle(np.max(cycles)) if len(cycles) else None,
                "summary_waveform_record_count_aligned": count_aligned,
                "summary_axis_finite_integer": original_integer,
                "summary_axis_positive_finite_integer": bool(original_integer and (cycles >= 1).all()),
                "summary_axis_strictly_increasing": increasing,
                "axis_binding_failures": failures,
                "axis_binding_status": "COUNT_ALIGNED_RELEASE_SUMMARY_COORDINATE" if not failures else "FAILED_SOURCE_AXIS_BINDING",
                "summary_cycle_provenance": "Author cell_analysis.m assigns summary.cycle=(1:k), discarding original instrument Cycle_Index values",
                "instrument_cycle_axis_authenticated": False,
                "raw_cycle_certified": False,
                "certification_pending": "Original instrument Cycle_Index or independent no-gap source proof, plus per-record phase capacity and summary agreement",
                "time_axis": "Original t values retained without unit conversion or continuity repair"}
    return meta, arrays


def iter_raw_cycles(path, index, metadata=None, summaries=None, batch_number=None):
    if metadata is None or summaries is None:
        metadata, summaries = cell_metadata(path, index, batch_number)
    with h5py.File(path, "r") as handle:
        _, _, waveforms = source_groups(handle, index)
        references = {name: waveforms[name][()].reshape(-1) for name in RAW_FIELDS if name in waveforms}
        for record in range(metadata["raw_records"]):
            summary_row = {name: float(values[record]) if record < len(values) else None
                           for name, values in summaries.items()}
            summary_cycle = summary_row.get("cycle")
            local_cycle = integer_cycle(summary_cycle) if summary_cycle is not None else None
            arrays = {name: np.asarray(handle[refs[record]][()]).reshape(-1)
                      if record < len(refs) else np.asarray([], dtype=float)
                      for name, refs in references.items()}
            for name in RAW_FIELDS:
                arrays.setdefault(name, np.asarray([], dtype=float))
            yield {**arrays, "cell_uid": metadata["cell_uid"],
                   "source_batch": metadata["source_batch"], "source_cell_index": index,
                   "source_segment_id": metadata["source_segment_id"],
                   "source_record_index": record, "source_record_cycle": record + 1,
                   "summary_cycle_sequence": local_cycle,
                   "instrument_original_cycle_id": None,
                   "cycle_axis_scope": "author_regenerated_release_summary_cycle",
                   "summary_cycle_value_raw": summary_cycle,
                   "summary_QD_Ah": summary_row.get("QDischarge"),
                   "summary_QC_Ah": summary_row.get("QCharge"), "summary": summary_row,
                   "source_axis_count_aligned": metadata["summary_waveform_record_count_aligned"],
                   "raw_cycle_certified": False,
                   "waveform_point_counts": {name: len(array) for name, array in arrays.items()},
                   "t_unit": "source_native_pending_integral_check"}


def read_source_cell(path, index, batch_number=None):
    """Return metadata and a lazy raw-cycle iterator, each bound to one source cell."""
    metadata, summaries = cell_metadata(path, index, batch_number)
    return metadata, iter_raw_cycles(path, index, metadata, summaries)


def iter_source_cells(path, batch_number=None):
    with h5py.File(path, "r") as handle:
        count = int(handle["batch"]["cycles"].size)
    for index in range(count):
        yield read_source_cell(path, index, batch_number)


def continuation_global_offset(predecessor_metadata):
    """Return the preceding released-summary offset, with no instrument claim."""
    if not predecessor_metadata["summary_axis_finite_integer"] or not predecessor_metadata["summary_axis_strictly_increasing"]:
        raise ValueError("Cannot attach continuation to a failed original summary axis")
    last = predecessor_metadata["summary_sequence_last"]
    maximum = predecessor_metadata["summary_sequence_max"]
    if last is None or last != maximum:
        raise ValueError("Predecessor last original cycle differs from its maximum")
    return last
