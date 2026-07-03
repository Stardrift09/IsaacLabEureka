"""Dump collision-mesh info for a USD file. Run with the eureka conda python.

Usage:
    ~/miniconda3/envs/eureka/bin/python scripts/inspect_collision.py <path_to.usd>
"""
import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

usd_path = sys.argv[1] if len(sys.argv) > 1 else (
    "libero/COMMON/stable_hope_objects/alphabet_soup/usd/alphabet_soup.usd"
)

stage = Usd.Stage.Open(usd_path)
print(f"Opened: {usd_path}\n")

found = 0
for prim in stage.Traverse():
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    found += 1
    approx = "(none / triangleMesh)"
    if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        attr = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr()
        if attr and attr.Get():
            approx = attr.Get()
    enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()

    # bounds of the prim's geometry
    bbox = ""
    if prim.IsA(UsdGeom.Boundable):
        rng = UsdGeom.Boundable(prim).ComputeExtent(Usd.TimeCode.Default())
        if rng:
            bbox = f"  extent={list(rng)}"

    print(f"[{prim.GetTypeName()}] {prim.GetPath()}")
    print(f"    collisionEnabled={enabled}  approximation={approx}{bbox}")

if found == 0:
    print("No prim has CollisionAPI -> object has NO collision mesh.")
else:
    print(f"\n{found} collider prim(s) found.")

app.close()
