"""Parse pickle as a data graph without importing or invoking pickle globals.

REDUCE/NEWOBJ/BUILD are represented symbolically. Numeric NumPy buffers are
decoded explicitly after a non-object dtype and byte-length check.
"""

from __future__ import annotations

import pickletools
from dataclasses import dataclass

import numpy as np


@dataclass(eq=False)
class Symbol:
    kind: str
    target: object
    args: object = None
    state: object = None


def parse_pickle_data(payload, capture_cycle_prefix=None):
    stack, memo, globals_seen = [], {}, set()
    captured_cycles = []
    marker = object()

    def take_mark():
        index = len(stack) - 1
        while index >= 0 and stack[index] is not marker:
            index -= 1
        if index < 0:
            raise ValueError("Missing pickle MARK")
        items = stack[index + 1:]
        del stack[index:]
        return items

    for opcode, arg, position in pickletools.genops(payload):
        name = opcode.name
        if name in ("PROTO", "FRAME"):
            continue
        if name == "STOP":
            if len(stack) != 1:
                raise ValueError("Unexpected pickle stack at STOP")
            return stack[0], sorted(globals_seen)
        if name == "MARK":
            stack.append(marker)
        elif name in ("SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE", "SHORT_BINBYTES", "BINBYTES", "BINBYTES8", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4", "INT", "FLOAT", "BINFLOAT", "STRING", "BINSTRING", "SHORT_BINSTRING"):
            stack.append(arg)
        elif name == "NONE":
            stack.append(None)
        elif name == "NEWTRUE":
            stack.append(True)
        elif name == "NEWFALSE":
            stack.append(False)
        elif name == "EMPTY_DICT":
            stack.append({})
        elif name == "EMPTY_LIST":
            stack.append([])
        elif name == "EMPTY_TUPLE":
            stack.append(())
        elif name == "TUPLE":
            stack.append(tuple(take_mark()))
        elif name in ("TUPLE1", "TUPLE2", "TUPLE3"):
            count = int(name[-1])
            args = tuple(stack[-count:])
            del stack[-count:]
            stack.append(args)
        elif name in ("BINPUT", "LONG_BINPUT", "PUT"):
            memo[int(arg)] = stack[-1]
        elif name == "MEMOIZE":
            memo[len(memo)] = stack[-1]
        elif name in ("BINGET", "LONG_BINGET", "GET"):
            stack.append(memo[int(arg)])
        elif name == "APPEND":
            value = stack.pop()
            stack[-1].append(value)
        elif name == "APPENDS":
            items = take_mark()
            stack[-1].extend(items)
        elif name == "SETITEM":
            value, key = stack.pop(), stack.pop()
            stack[-1][key] = value
        elif name == "SETITEMS":
            items = take_mark()
            if len(items) % 2:
                raise ValueError("Odd SETITEMS length")
            for key, value in zip(items[::2], items[1::2]):
                stack[-1][key] = value
            if capture_cycle_prefix and isinstance(stack[-1], dict) and {"cycle_number", "voltage_in_V", "current_in_A"}.issubset(stack[-1]):
                captured_cycles.append(stack[-1])
                if len(captured_cycles) >= capture_cycle_prefix:
                    return {"cycle_data": captured_cycles}, sorted(globals_seen)
        elif name == "GLOBAL":
            module, target = arg.split(" ", 1)
            full_name = module + "." + target
            globals_seen.add(full_name)
            stack.append(Symbol("global", full_name))
        elif name == "STACK_GLOBAL":
            target, module = stack.pop(), stack.pop()
            if not isinstance(module, str) or not isinstance(target, str):
                raise ValueError("GLOBAL operands must be strings")
            full_name = module + "." + target
            globals_seen.add(full_name)
            stack.append(Symbol("global", full_name))
        elif name in ("REDUCE", "NEWOBJ"):
            args, target = stack.pop(), stack.pop()
            stack.append(Symbol(name.lower(), target, args))
        elif name == "BUILD":
            state = stack.pop()
            if not isinstance(stack[-1], Symbol):
                raise ValueError("BUILD target must remain symbolic")
            stack[-1].state = state
        elif name == "POP":
            stack.pop()
        elif name == "DUP":
            stack.append(stack[-1])
        else:
            raise ValueError(f"Unsupported static pickle opcode {name} at {position}")
    raise ValueError("Pickle STOP missing")


def global_name(value):
    if isinstance(value, Symbol) and value.kind == "global":
        return value.target
    return None


def symbol_name(value):
    return global_name(value.target) if isinstance(value, Symbol) else None


def numpy_array(value):
    if symbol_name(value) not in ("numpy.core.multiarray._reconstruct", "numpy._core.multiarray._reconstruct"):
        raise ValueError("Expected symbolic NumPy ndarray")
    version, shape, dtype_symbol, fortran, payload = value.state
    if version != 1 or not isinstance(shape, tuple) or len(shape) > 3:
        raise ValueError("Unsupported NumPy array state")
    if symbol_name(dtype_symbol) != "numpy.dtype":
        raise ValueError("Expected explicit NumPy dtype")
    code = dtype_symbol.args[0]
    if code in ("O8", "O4", "O"):
        if not isinstance(payload, list) or any(not isinstance(v, (str, int, float, bool, type(None))) for v in payload):
            raise ValueError("Object array supports primitive data literals only")
        return np.asarray(payload, dtype=object).reshape(shape, order="F" if fortran else "C")
    dtype = np.dtype(code)
    if dtype.kind not in "biufc" or dtype.itemsize > 16 or dtype.hasobject:
        raise ValueError("Unsafe or unsupported numeric dtype")
    endian = dtype_symbol.state[1] if dtype_symbol.state else "="
    dtype = dtype.newbyteorder(endian)
    count = int(np.prod(shape))
    if not isinstance(payload, bytes) or len(payload) != count * dtype.itemsize:
        raise ValueError("Numeric buffer length mismatch")
    return np.frombuffer(payload, dtype=dtype).reshape(shape, order="F" if fortran else "C")


def index_values(value):
    if symbol_name(value) != "pandas.core.indexes.base._new_Index":
        raise ValueError("Expected symbolic pandas Index")
    target, state = value.args
    if global_name(target) == "pandas.core.indexes.range.RangeIndex":
        return range(state["start"], state["stop"], state["step"])
    if global_name(target) == "pandas.core.indexes.base.Index":
        return numpy_array(state["data"]).tolist()
    raise ValueError("Unsupported symbolic Index class")


def dataframe_columns(value):
    """Return column arrays by reading pandas' compatibility state as data."""
    if symbol_name(value) != "pandas.core.frame.DataFrame":
        raise ValueError("Expected symbolic pandas DataFrame")
    manager = value.state["_mgr"]
    if symbol_name(manager) != "pandas.core.internals.managers.BlockManager":
        raise ValueError("Expected symbolic BlockManager")
    state = manager.state[3]["0.14.1"]
    names = list(index_values(state["axes"][0]))
    expected_rows = len(index_values(state["axes"][1]))
    columns = {}
    for block in state["blocks"]:
        values = numpy_array(block["values"])
        locations = block["mgr_locs"]
        if symbol_name(locations) == "builtins.slice":
            if any(v is not None and not isinstance(v, int) for v in locations.args):
                raise ValueError("Slice arguments must be integer literals")
            positions = list(range(len(names)))[slice(*locations.args)]
        else:
            positions = numpy_array(locations).tolist()
        if values.shape != (len(positions), expected_rows):
            raise ValueError("DataFrame block dimensions disagree")
        for row, position in enumerate(positions):
            columns[names[position]] = values[row]
    if set(columns) != set(names) or len(columns) != len(names):
        raise ValueError("DataFrame columns incomplete or duplicated")
    return {name: columns[name] for name in names}


def brief(value, depth=0):
    if depth > 4:
        return type(value).__name__
    if isinstance(value, Symbol):
        return {"symbol": value.kind, "target": symbol_name(value) or global_name(value),
                "args": brief(value.args, depth + 1), "state": brief(value.state, depth + 1)}
    if isinstance(value, bytes):
        return {"bytes": len(value)}
    if isinstance(value, dict):
        return {str(k): brief(v, depth + 1) for k, v in list(value.items())[:8]}
    if isinstance(value, (tuple, list)):
        return [brief(v, depth + 1) for v in value[:6]]
    return value
