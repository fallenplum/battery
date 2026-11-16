"""Replay serialized P0 artifacts on an independent local compute device."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .data import ROOT, file_hash, write_json
from .train import configure, load_selected, load_source, make_model, predict, read_json, tensor_fingerprint


def replay_unit(folder, data, inputs, backbone_dir, device):
    folder = Path(folder)
    receipt = read_json(folder / 'SOURCE_DEVELOPMENT_COMPLETE.json')
    for record in receipt['files']:
        if file_hash(folder / record['path']) != record['sha256']:
            raise RuntimeError('Local artifact checksum differs at ' + record['path'])
    identity = read_json(folder / 'RUN_CONFIG.json')
    baseline_scaler = None
    if identity['method'] == 'B3_battery_direct':
        baseline_scaler = read_json(folder / 'BATTERY_BASELINE_SCALER.json')
    model = make_model(
        backbone_dir, inputs['policy'].shape[-1], identity['bridge'],
        identity['method'], identity['seed'], identity['rank'],
        baseline_scaler=baseline_scaler,
    ).to(device)
    checkpoint = torch.load(folder / 'CHECKPOINT_SELECTED.pt', map_location='cpu', weights_only=False)
    if checkpoint['identity'] != identity:
        raise RuntimeError('Serialized checkpoint identity differs from the run configuration')
    authoritative_random_state = False
    if identity['method'] == 'B1_random_frozen':
        random_folder = ROOT / 'results/drs_peft/P0/random_backbones'
        manifest = read_json(random_folder / 'RANDOM_BACKBONE_MANIFEST.json')
        record = next(item for item in manifest if item['seed'] == identity['seed'])
        state_path = random_folder / record['path']
        if file_hash(state_path) != record['sha256']:
            raise RuntimeError('Authoritative random backbone file changed')
        state = torch.load(state_path, map_location='cpu', weights_only=True)
        model.backbone.load_state_dict(state, strict=True)
        authoritative_random_state = True
    if identity['method'] != 'B3_battery_direct':
        actual_hash = tensor_fingerprint(model.backbone)
        if identity['method'] == 'B2_standard_lora':
            actual_hash = model.official_backbone_identity['actual_initial_tensor_sha256']
        expected_hash = checkpoint['backbone_id']['actual_initial_tensor_sha256']
        if actual_hash != expected_hash:
            raise RuntimeError('Local frozen backbone tensor identity differs from the trained model')
    load_selected(model, checkpoint['model'])
    with np.load(folder / 'SOURCE_VALIDATION_PREDICTIONS.npz', allow_pickle=False) as archive:
        ids = archive['ids'].tolist()
        reference = archive['prediction']
        query = archive['query']
    replay = predict(model, inputs, data.rows(ids), query, device)
    maximum_error = float(np.abs(replay - reference).max())
    equal = bool(np.array_equal(replay, reference))
    passed = bool(np.allclose(replay, reference, atol=1e-4, rtol=0))
    if not passed:
        raise RuntimeError(f'Independent device replay exceeds the registered 1e-4 tolerance, error={maximum_error}')
    record = dict(
        status='PASSED', scope='independent local device serialized P0 replay',
        method=identity['method'], bridge=identity['bridge'], seed=identity['seed'],
        cells=len(ids), maximum_absolute_error=maximum_error,
        bitwise_equal=equal, absolute_tolerance=1e-4, relative_tolerance=0,
        device=device, gpu=torch.cuda.get_device_name() if device.startswith('cuda') else None,
        torch=torch.__version__, checkpoint_sha256=file_hash(folder / 'CHECKPOINT_SELECTED.pt'),
        target_query_labels_used=0,
        authoritative_random_backbone_loaded=authoritative_random_state,
    )
    write_json(folder / 'INDEPENDENT_LOCAL_REPLAY.json', record)
    print(record, flush=True)
    del model
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results/drs_peft/P0')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'runtime/drs_peft/data_v1')
    parser.add_argument('--backbone-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    configure()
    data, inputs, _, _ = load_source(args.data_root)
    markers = sorted(args.output_root.rglob('SOURCE_DEVELOPMENT_COMPLETE.json'))
    if not markers:
        raise RuntimeError('No completed P0 source units found')
    for marker in markers:
        replay_unit(marker.parent, data, inputs, args.backbone_dir, args.device)


if __name__ == '__main__':
    main()
