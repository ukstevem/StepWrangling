#!/usr/bin/env python3
"""
Two-phase STEP assembly names extraction:
1. Lightweight text scan to extract PRODUCT names in file order (using stricter regex).
2. Use pythonOCC STEPControl_Reader to traverse shape hierarchy and map shapes to names by index.
"""
import re
import sys
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopoDS import TopoDS_Iterator


def parse_product_names(step_path):
    """
    Parse PRODUCT lines from the STEP file to collect part names in declaration order.
    Uses a stricter regex to capture the second parameter (name).
    """
    # Match lines like: #12 = PRODUCT('id','Name',
    pattern = re.compile(r"^\s*#\d+\s*=\s*PRODUCT\s*\(\s*'[^']*'\s*,\s*'([^']*)'", re.IGNORECASE)
    names = []
    with open(step_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = pattern.match(line)
            if m:
                names.append(m.group(1))
    return names


def print_step_tree(step_path):
    # Phase 1: text scan
    names = parse_product_names(step_path)
    print(f"Parsed {len(names)} product names from STEP.")

    # Phase 2: shape traversal
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != IFSelect_RetDone:
        print(f"Error: cannot read STEP file '{step_path}'")
        sys.exit(1)
    reader.TransferRoots()

    compound = reader.OneShape()

    # Recursive traverse with global index
    idx = 0
    def recurse(shape, indent=0):
        nonlocal idx
        if idx < len(names):
            name = names[idx]
        else:
            name = '<unnamed>'
        print('  ' * indent + name)
        idx += 1
        it = TopoDS_Iterator(shape)
        while it.More():
            recurse(it.Value(), indent + 1)
            it.Next()

    print(f"Assembly tree for '{step_path}':")
    recurse(compound)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <file.step>")
        sys.exit(1)
    print_step_tree(sys.argv[1])
