"""Convert the G1 URDF used by sonic_release to USD with the Isaac Lab 2.3 importer (Isaac Sim 5.1).

The Isaac Lab 3.0 URDF importer ignores replace_cylinders_with_capsules, so the 2.3 output is
converted once and loaded on 3.0 through manager_env.config.robot.frozen_usd. Run with an Isaac
Lab 2.3 install; also writes the collision-shape inventory per body (collision_shapes.json).

    python gear_sonic/scripts/isaaclab3/freeze_g1_usd.py --out_dir <dir>
"""

import argparse
import collections
import importlib.util
import json
import os

p = argparse.ArgumentParser()
p.add_argument("--sonic_root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
p.add_argument("--out_dir", required=True)
args = p.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

from isaaclab.sim.converters import UrdfConverter  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

# load robots/g1.py by path: importing gear_sonic would pull in the whole training stack
spec = importlib.util.spec_from_file_location(
    "sonic_g1", f"{args.sonic_root}/gear_sonic/envs/manager_env/robots/g1.py"
)
g1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g1)
spawn = g1.G1_CYLINDER_MODEL_12_DEX_CFG.spawn
cfg = spawn.replace(
    asset_path=os.path.join(args.sonic_root, spawn.asset_path),
    usd_dir=args.out_dir,
    usd_file_name="g1_sonic.usd",
    force_usd_conversion=True,
)
usd_path = UrdfConverter(cfg).usd_path

stage = Usd.Stage.Open(usd_path)
shapes = collections.defaultdict(collections.Counter)
for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        body = prim.GetParent()
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        shapes[body.GetName() if body else "?"][prim.GetTypeName()] += 1
total = sum(shapes.values(), collections.Counter())
inventory = {
    "usd": usd_path,
    "total": dict(total),
    "per_body": {k: dict(v) for k, v in sorted(shapes.items())},
}
with open(os.path.join(args.out_dir, "collision_shapes.json"), "w") as f:
    json.dump(inventory, f, indent=1)
print("USD:", usd_path)
print("collision shapes:", dict(total), "bodies:", len(shapes))
app.close()
