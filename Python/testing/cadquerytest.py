#!/usr/bin/env python3
"""
STEP assembly extraction using CadQuery’s high-level API.
1. Install: pip install cadquery
2. Use Assembly.importStep() to capture part names and hierarchy.
3. Recursively print the tree.
"""
import sys
from cadquery import Assembly


def print_tree(node, indent=0):
    """
    Recursively prints the assembly hierarchy.
    """
    # Assembly name or default
    name = getattr(node, 'name', None) or '<unnamed>'
    print('  ' * indent + name)
    # Iterate over child assemblies
    for child_name, child_obj in node.children.items():
        print_tree(child_obj, indent + 1)


def main(step_path):
    try:
        # import STEP as an Assembly
        asm = Assembly.importStep(step_path)
    except Exception as e:
        print(f"Error loading STEP with CadQuery Assembly.importStep: {e}")
        sys.exit(1)

    print(f"Assembly tree for '{step_path}':")
    print_tree(asm)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <file.step>")
        sys.exit(1)
    main(sys.argv[1])
