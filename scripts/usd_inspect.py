#!/usr/bin/env python3
"""
Inspect USD file structure: hierarchy, prim types, attributes, and relationships.

Usage:
    python scripts/usd_inspect.py <file.usd[a|z]> [options]

Options:
    --depth N       Max traversal depth (default: unlimited)
    --attrs         Print prim attributes and their values
    --meta          Print prim metadata
    --variants      Print variant sets
    --refs          Print references / payloads
    --filter TYPE   Only show prims of this type (e.g. Mesh, Xform, Material)
    --no-tree       Skip hierarchy tree, just summary
"""

import argparse
import sys

try:
    from pxr import Usd, UsdGeom, Sdf, Kind
except ImportError:
    print("ERROR: pxr (OpenUSD) not found.")
    print("Install: pip install usd-core   OR use Isaac Sim python environment.")
    sys.exit(1)


# ── helpers ──────────────────────────────────────────────────────────────────

def indent(depth):
    return "│   " * depth

def prim_label(prim):
    type_name = prim.GetTypeName() or "<untyped>"
    name      = prim.GetName()
    path      = prim.GetPath()
    active    = "" if prim.IsActive() else " [INACTIVE]"
    abstract  = " [ABSTRACT]" if prim.IsAbstract() else ""
    return f"{name}  [{type_name}]  {path}{active}{abstract}"

def print_attrs(prim, depth):
    for attr in prim.GetAttributes():
        val = attr.Get()
        print(f"{indent(depth+1)}@ {attr.GetName()} ({attr.GetTypeName()}) = {val}")

def print_rels(prim, depth):
    for rel in prim.GetRelationships():
        targets = rel.GetTargets()
        print(f"{indent(depth+1)}-> {rel.GetName()} => {targets}")

def print_meta(prim, depth):
    meta = prim.GetAllMetadata()
    for k, v in meta.items():
        print(f"{indent(depth+1)}# {k}: {v}")

def print_variants(prim, depth):
    vsets = prim.GetVariantSets()
    for name in vsets.GetNames():
        vset = vsets.GetVariantSet(name)
        choices = vset.GetVariantNames()
        current = vset.GetVariantSelection()
        print(f"{indent(depth+1)}[variant] {name}: {choices}  (selected: {current!r})")

def print_refs(prim, depth):
    stack = prim.GetPrimStack()
    for spec in stack:
        refs = spec.referenceList.prependedItems
        plds = spec.payloadList.prependedItems
        for r in refs:
            print(f"{indent(depth+1)}[ref]     {r.assetPath}  layer: {spec.layer.identifier}")
        for p in plds:
            print(f"{indent(depth+1)}[payload] {p.assetPath}  layer: {spec.layer.identifier}")


# ── tree traversal ────────────────────────────────────────────────────────────

def traverse(prim, depth, args, counters):
    type_name = prim.GetTypeName()

    # filter
    if args.filter and type_name != args.filter:
        for child in prim.GetChildren():
            traverse(child, depth, args, counters)
        return

    # depth limit
    if args.depth is not None and depth > args.depth:
        return

    pad = indent(depth)
    connector = "├── " if depth > 0 else ""
    print(f"{pad}{connector}{prim_label(prim)}")

    counters[type_name] = counters.get(type_name, 0) + 1

    if args.meta:
        print_meta(prim, depth)
    if args.variants:
        print_variants(prim, depth)
    if args.refs:
        print_refs(prim, depth)
    if args.attrs:
        print_attrs(prim, depth)
        print_rels(prim, depth)

    for child in prim.GetChildren():
        traverse(child, depth + 1, args, counters)


# ── summary ───────────────────────────────────────────────────────────────────

def print_summary(stage, counters):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # layer stack
    print(f"\nRoot layer : {stage.GetRootLayer().identifier}")
    layers = stage.GetLayerStack()
    if len(layers) > 1:
        print("Layer stack:")
        for lyr in layers:
            print(f"  {lyr.identifier}")

    # default prim
    dp = stage.GetDefaultPrim()
    if dp:
        print(f"\nDefault prim: {dp.GetPath()}")

    # up axis / meters per unit
    up  = UsdGeom.GetStageUpAxis(stage)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    print(f"Up axis     : {up}")
    print(f"Meters/unit : {mpu}")

    # prim type counts
    print(f"\nPrim type counts:")
    for t, c in sorted(counters.items(), key=lambda x: -x[1]):
        print(f"  {t or '<untyped>':30s} {c}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inspect USD file structure.")
    parser.add_argument("usd_file",              help="Path to .usd/.usda/.usdc/.usdz file")
    parser.add_argument("--depth",   type=int,   default=None,  help="Max hierarchy depth")
    parser.add_argument("--attrs",   action="store_true",       help="Show prim attributes")
    parser.add_argument("--meta",    action="store_true",       help="Show prim metadata")
    parser.add_argument("--variants",action="store_true",       help="Show variant sets")
    parser.add_argument("--refs",    action="store_true",       help="Show references/payloads")
    parser.add_argument("--filter",  default=None,              help="Filter by prim type")
    parser.add_argument("--no-tree", action="store_true",       help="Skip hierarchy, show summary only")
    args = parser.parse_args()

    print(f"Opening: {args.usd_file}")
    stage = Usd.Stage.Open(args.usd_file)
    if not stage:
        print(f"ERROR: failed to open {args.usd_file}")
        sys.exit(1)

    counters = {}

    if not args.no_tree:
        print("\n" + "=" * 60)
        print("HIERARCHY")
        print("=" * 60)
        root = stage.GetPseudoRoot()
        for child in root.GetChildren():
            traverse(child, 0, args, counters)
    else:
        # still count prims for summary
        for prim in stage.Traverse():
            t = prim.GetTypeName()
            counters[t] = counters.get(t, 0) + 1

    print_summary(stage, counters)


if __name__ == "__main__":
    main()
