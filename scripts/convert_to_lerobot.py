"""Convert a recorded replay dataset (from scripts/replay_record.py) into a
LeRobot v3 dataset.

IMPORTANT: run this in a DEDICATED lerobot environment, NOT the `eureka` /
Isaac Sim env. Installing lerobot into the Isaac Sim env pulls numpy>=2 (and
churns av / torchcodec / huggingface-hub), which breaks Isaac Sim. Create a
separate venv:

    python -m venv ~/lerobot-venv && source ~/lerobot-venv/bin/activate
    pip install lerobot imageio imageio-ffmpeg

Then:

    python scripts/convert_to_lerobot.py \
        --in logs/replay_record --repo-id local/pickitup_replay [--max-episodes 2]

Input layout (produced by replay_record.py):
    <in>/meta.json
    <in>/manifest.jsonl                          # one json per episode
    <in>/{successful,unsuccessful}/epNN/agentview.mp4
                                       /wrist.mp4
                                       /arrays.npz  (observation_state, action, timestamp)
                                       /episode_meta.json

Each episode's per-frame `task` is the base task, suffixed with " unsuccessful"
for failed episodes, so success/failure stays distinguishable in the dataset.
"""

import argparse
import json
import os


def load_manifest(in_dir):
    rows = []
    with open(os.path.join(in_dir, "manifest.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", required=True, help="recorded dataset dir")
    parser.add_argument("--repo-id", required=True, help="LeRobot repo id, e.g. local/pickitup")
    parser.add_argument("--root", default=None, help="output root (default: lerobot cache)")
    parser.add_argument("--max-episodes", type=int, default=None, help="cap episodes (debug)")
    args = parser.parse_args()

    import imageio.v3 as iio
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    in_dir = args.in_dir
    meta = json.load(open(os.path.join(in_dir, "meta.json")))
    fps = meta["fps"]
    cams = meta["cameras"]
    h = w = 256  # recorded resolution

    features = {
        "observation.state": {"dtype": "float32", "shape": (meta["state_dim"],), "names": None},
        "action": {"dtype": "float32", "shape": (meta["action_dim"],), "names": None},
    }
    for cam in cams:
        features[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channel"],
        }

    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=fps, features=features, root=args.root,
        robot_type="franka", use_videos=True,
    )

    rows = load_manifest(in_dir)
    if args.max_episodes is not None:
        rows = rows[: args.max_episodes]

    for r in rows:
        ep_dir = os.path.join(in_dir, r["ep_dir"])
        arr = np.load(os.path.join(ep_dir, "arrays.npz"))
        states, actions = arr["observation_state"], arr["action"]
        # decode the camera mp4s back to frame stacks
        cam_stacks = {cam: iio.imread(os.path.join(ep_dir, f"{cam}.mp4")) for cam in cams}
        n = min(len(states), len(actions), *[len(cam_stacks[c]) for c in cams])
        task = r["task"]
        for i in range(n):
            frame = {
                "observation.state": states[i].astype(np.float32),
                "action": actions[i].astype(np.float32),
            }
            for cam in cams:
                frame[f"observation.images.{cam}"] = cam_stacks[cam][i][..., :3].astype(np.uint8)
            ds.add_frame(frame, task=task)
        ds.save_episode()
        print(f"  converted ep{r['episode_index']:02d} ({'ok' if r['success'] else 'fail'}, {n} frames): {task}")

    print(f"done -> {ds.root}")


if __name__ == "__main__":
    main()
