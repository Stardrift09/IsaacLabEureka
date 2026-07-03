"""Generate a deterministic init grid for even-coverage data collection.

Builds an npz of full init configs that `PickItUpClean`/`BatchEvalPickItUpClean` consume in
order (via cfg ``init_grid_file``), so collection covers the trained state space on an even
grid instead of re-drawing the random ring. Each entry fully specifies:

  * ``object_xy`` [N,2] : env-local target-object XY (nominal + grid offset, clipped to disk)
  * ``basket_xy`` [N,2] : env-local basket XY (nominal + grid offset)
  * ``robot_qpos`` [N,J]: full robot joint targets = nominal (mean of demos) + sampled arm noise

The spatial axes (object XY, basket XY) are gridded evenly over the curriculum-end disk
(radius = outer_diam/2). The robot's 7 arm joints can't be gridded (7-D), so they are SAMPLED
~N(0, arm_joint_noise) per entry and stored, keeping every init fully reproducible.

No Isaac Sim needed -- reads the same sources as PickItUpClean._compute_nominal_init
(init_state_distribution.npz for object/basket XY means; the demo pkl for the robot mean).

    python scripts/make_init_grid.py --out logs/init_grids/grid_v1.npz \
        --outer_diam 0.24 --n_object 18 --n_basket 18 --mode product --max_count 90000
"""

import argparse
import os

import numpy as np

from isaaclab_eureka.utils import eureka_root_dir, read_pkl


def _disk_lattice(center_xy, radius, n_per_axis):
    """Even square lattice over [-r,r]^2 around center, clipped to the disk. -> [M,2]."""
    if n_per_axis <= 1 or radius <= 0:
        return np.asarray([center_xy], dtype=np.float64)
    g = np.linspace(-radius, radius, n_per_axis)
    xx, yy = np.meshgrid(g, g)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    pts = pts[(pts[:, 0] ** 2 + pts[:, 1] ** 2) <= radius * radius + 1e-12]
    return pts + np.asarray(center_xy, dtype=np.float64)[None, :]


def _nominal(target_object, npz_path):
    """Mean-of-demos nominal: robot qpos (J) + object/basket XY (npz means, fallback demo mean)."""
    root = eureka_root_dir()
    pkl = (f"{root}/libero/trajs/libero90/libero_90_living_room_scene1_pick_up_the_"
           f"{target_object}_and_put_it_in_the_basket_traj_v2.pkl")
    eps = read_pkl(pkl)["franka"]
    joints, obj_xy, bask_xy = [], [], []
    for ep in eps:
        s0 = ep["states"][0]
        jp = dict(s0["franka"]["dof_pos"])
        jp["panda_finger_joint2"] = jp["panda_finger_joint1"]
        joints.append([float(np.asarray(v).reshape(-1)[0]) for v in jp.values()])
        obj_xy.append(s0[target_object]["pos"][:2])
        bask_xy.append(s0["basket"]["pos"][:2])
    robot_nominal = np.mean(np.asarray(joints, dtype=np.float64), axis=0)  # [J]
    obj_c = np.mean(np.asarray(obj_xy, dtype=np.float64), axis=0)
    bask_c = np.mean(np.asarray(bask_xy, dtype=np.float64), axis=0)
    if os.path.isfile(npz_path):
        d = np.load(npz_path, allow_pickle=True)
        if "object_samples" in d.files:
            obj_c = d["object_samples"].mean(0)[:2]
        if "basket_samples" in d.files:
            bask_c = d["basket_samples"].mean(0)[:2]
    return robot_nominal, obj_c, bask_c, len(eps)


def main():
    p = argparse.ArgumentParser(description="Build a deterministic init grid npz.")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--target_object", type=str, default="alphabet_soup")
    p.add_argument("--npz", type=str, default="/home/shaotongchen/lerobot/logs/init_state_distribution.npz")
    p.add_argument("--outer_diam", type=float, default=0.24, help="Disk diameter (m); radius = /2.")
    p.add_argument("--n_object", type=int, default=18, help="Object lattice points per axis (pre-clip).")
    p.add_argument("--n_basket", type=int, default=18, help="Basket lattice points per axis (pre-clip).")
    p.add_argument("--mode", choices=["product", "paired"], default="product",
                   help="product: every object x every basket; paired: zip object/basket (decorrelated).")
    p.add_argument("--max_count", type=int, default=90000, help="Cap on N (random even subsample if exceeded).")
    p.add_argument("--arm_joint_noise", type=float, default=0.05, help="Std (rad) on the 7 arm joints.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    r = 0.5 * args.outer_diam
    robot_nominal, obj_c, bask_c, n_demo = _nominal(args.target_object, args.npz)
    obj_lat = _disk_lattice(obj_c, r, args.n_object)     # [Go,2]
    bask_lat = _disk_lattice(bask_c, r, args.n_basket)   # [Gb,2]

    if args.mode == "product":
        oi, bi = np.meshgrid(np.arange(len(obj_lat)), np.arange(len(bask_lat)))
        oi, bi = oi.ravel(), bi.ravel()
    else:  # paired: object lattice in order; basket via a random permutation so BOTH marginals
           # are fully covered and decorrelated (a fixed stride can alias when sizes share a factor).
        N = max(len(obj_lat), len(bask_lat))
        oi = np.arange(N) % len(obj_lat)
        bperm = rng.permutation(len(bask_lat))
        bi = bperm[np.arange(N) % len(bask_lat)]

    object_xy = obj_lat[oi]
    basket_xy = bask_lat[bi]
    N = object_xy.shape[0]
    if N > args.max_count:  # even random subsample of the lattice product
        keep = rng.choice(N, size=args.max_count, replace=False)
        keep.sort()
        object_xy, basket_xy = object_xy[keep], basket_xy[keep]
        N = args.max_count

    # robot: nominal + sampled arm-joint noise (7-D can't be gridded), fingers fixed
    robot_qpos = np.tile(robot_nominal[None, :], (N, 1))
    robot_qpos[:, :7] += rng.normal(0.0, args.arm_joint_noise, size=(N, 7))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(
        args.out,
        object_xy=object_xy.astype(np.float32),
        basket_xy=basket_xy.astype(np.float32),
        robot_qpos=robot_qpos.astype(np.float32),
        # metadata
        outer_diam=np.float32(args.outer_diam), radius=np.float32(r),
        object_center=obj_c.astype(np.float32), basket_center=bask_c.astype(np.float32),
        mode=args.mode, arm_joint_noise=np.float32(args.arm_joint_noise),
        n_object=args.n_object, n_basket=args.n_basket, n_demo=n_demo, seed=args.seed,
    )
    print(f"[make_init_grid] wrote {N} entries -> {args.out}")
    print(f"  radius={r:.3f} m  object_lattice={len(obj_lat)}  basket_lattice={len(bask_lat)}  mode={args.mode}")
    print(f"  object_center={obj_c.tolist()}  basket_center={bask_c.tolist()}  robot J={robot_qpos.shape[1]}")


if __name__ == "__main__":
    main()
