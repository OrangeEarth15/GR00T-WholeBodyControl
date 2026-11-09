"""Make the 2.3-converted G1 USD readable by Newton's importer.

The Isaac Sim 5.1 URDF importer puts PhysicsCollisionAPI / PhysicsMeshCollisionAPI on an Xform
(`.../collisions/<link>/node_STL_BINARY_`) whose child `mesh` holds the geometry. PhysX accepts
that; Newton only reads collision APIs on gprims and skips the 18 mesh colliders (hands, wrists).
This flattens the stage and moves the collision APIs and their physics attributes onto the child
Mesh. Needs only pxr (usd-core).

    python gear_sonic/scripts/isaaclab3/newtonize_usd.py <dir>/g1_sonic.usd <dir>/g1_sonic_newton.usd
"""

import sys

from pxr import Sdf, Usd

src, dst = sys.argv[1], sys.argv[2]
layer = Usd.Stage.Open(src).Flatten()
COLLISION_APIS = (
    "PhysicsCollisionAPI",
    "PhysicsMeshCollisionAPI",
    "PhysxCollisionAPI",
    "PhysxConvexHullCollisionAPI",
)
moved = []


def api_items(spec):
    op = spec.GetInfo("apiSchemas")
    return list(op.prependedItems) + list(op.explicitItems) + list(op.appendedItems)


def visit(path):
    spec = layer.GetPrimAtPath(path)
    if not path.IsPrimPath() or spec is None or spec.typeName != "Xform":
        return
    apis = api_items(spec)
    collision = [a for a in apis if a in COLLISION_APIS]
    if not collision or "mesh" not in spec.nameChildren:
        return
    mesh = spec.nameChildren["mesh"]
    spec.SetInfo(
        "apiSchemas",
        Sdf.TokenListOp.Create(prependedItems=[a for a in apis if a not in COLLISION_APIS]),
    )
    mesh.SetInfo("apiSchemas", Sdf.TokenListOp.Create(prependedItems=api_items(mesh) + collision))
    for prop in list(spec.properties):
        if prop.name.startswith(("physics:", "physxCollision:", "physxConvexHullCollision:")):
            Sdf.CopySpec(layer, prop.path, layer, mesh.path.AppendProperty(prop.name))
            spec.RemoveProperty(prop)
    moved.append(str(path))


layer.Traverse(Sdf.Path.absoluteRootPath, visit)
layer.Export(dst)
print(f"moved collision APIs to child meshes on {len(moved)} prims -> {dst}")
