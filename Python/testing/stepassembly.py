# IFC-based assembly tree printer (with debug)
# Usage: python ifc_tree.py <file.ifc>
import sys
import os

# Debug: show arguments
print(f"[DEBUG] Args: {sys.argv}")

# Check IFC file path
if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <file.ifc>")
    sys.exit(1)
ifc_path = sys.argv[1]
print(f"[DEBUG] IFC path: {ifc_path}")
if not os.path.isfile(ifc_path):
    print(f"Error: IFC file '{{ifc_path}}' not found.")
    sys.exit(1)

# Import IfcOpenShell
try:
    import ifcopenshell
except ImportError:
    sys.stderr.write("Error: ifcopenshell not found. Install via 'pip install ifcopenshell'.\n")
    sys.exit(1)

# Function: print tree

def print_ifc_tree(entity, visited=None, indent=0):
    if visited is None:
        visited = set()
    if entity.id() in visited:
        return
    visited.add(entity.id())

    name = getattr(entity, 'Name', None)
    label = name if name else entity.is_a()
    print('  ' * indent + str(label))

    # Traverse decomposition
    rels = getattr(entity, 'IsDecomposedBy', [])
    print(f"[DEBUG] Entity {entity.id()} has {len(rels)} decompositions")
    for rel in rels:
        children = getattr(rel, 'RelatedObjects', [])
        print(f"[DEBUG] Rel has {len(children)} children")
        for child in children:
            print_ifc_tree(child, visited, indent + 1)

# Main processing

print(f"[DEBUG] Loading IFC model...")
try:
    model = ifcopenshell.open(ifc_path)
    print("[DEBUG] Loaded with standard schema")
except Exception:
    print("[DEBUG] Standard load failed, trying dynamic schema...")
    model = ifcopenshell.open(ifc_path, dynamic_schema=True)
    print("[DEBUG] Loaded with dynamic schema")

# Gather products
products = model.by_type('IfcProduct')
print(f"[DEBUG] Total IfcProduct entities: {len(products)}")

# Gather decomposed ids
decomposed_ids = set()
for rel in model.by_type('IfcRelAggregates'):
    objs = getattr(rel, 'RelatedObjects', [])
    print(f"[DEBUG] Found IfcRelAggregates with {len(objs)} related objects")
    for child in objs:
        decomposed_ids.add(child.id())
print(f"[DEBUG] Total decomposed ids: {len(decomposed_ids)}")

# Identify top-level
top_level = [p for p in products if p.id() not in decomposed_ids]
print(f"[DEBUG] Top-level IfcProduct count: {len(top_level)}")
if not top_level:
    print(f"No top-level IfcProduct entities found in '{ifc_path}'.")
    sys.exit(0)

print(f"IFC assembly tree for '{ifc_path}':")
for root in top_level:
    print_ifc_tree(root)
