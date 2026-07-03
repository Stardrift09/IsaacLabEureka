"""Recover top-level manifest.jsonl + meta.json for a collect_rsl_rl run that was
stopped before _write_dataset_manifest ran.

Every episode dir already holds a complete episode_meta.json (written per-episode
inside _write_episode_recording). The only missing artifacts are the two
dataset-level files, which are pure aggregates. We rebuild them by scanning all
ep* dirs under successful/ and unsuccessful/ -- byte-compatible with what
replay_pick_it_up_one_env._write_dataset_manifest would have emitted.

    python scripts/recover_manifest.py logs/collect_rsl_rl_v8
"""
import collections
import json
import os
import sys


def main(record_dir):
    eps = []
    for split in ("successful", "unsuccessful"):
        split_dir = os.path.join(record_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for name in os.listdir(split_dir):
            mpath = os.path.join(split_dir, name, "episode_meta.json")
            if not os.path.isfile(mpath):
                continue
            with open(mpath) as f:
                m = json.load(f)
            m["_ep_dir"] = os.path.relpath(os.path.join(split_dir, name), record_dir)
            eps.append(m)

    # collection order == ascending episode_index (ep counter increments per recorded ep)
    eps.sort(key=lambda m: m["episode_index"])
    if not eps:
        sys.exit(f"no episode_meta.json found under {record_dir}")

    # manifest.jsonl : one line per episode, same fields/order as the writer
    with open(os.path.join(record_dir, "manifest.jsonl"), "w") as f:
        for m in eps:
            f.write(json.dumps({
                "episode_index": m["episode_index"],
                "success": bool(m.get("success")),
                "task": m.get("task"),
                "length": m["length"],
                "ep_dir": m["_ep_dir"],
                "videos": m.get("videos"),
                "controller": m.get("controller"),
                "gains": m.get("gains"),
            }) + "\n")

    n = len(eps)
    n_succ = sum(bool(m.get("success")) for m in eps)
    ref = eps[0]
    cams = ref.get("cameras", [])

    # base_task: strip the " unsuccessful" suffix; env is randomized so the object
    # varies per episode. The original would have recorded only the LAST episode's
    # task here; we record the modal (most common) base_task instead and stash the
    # full set under base_task_variants so nothing is lost.
    base_tasks = collections.Counter(
        (m["task"][:-len(" unsuccessful")] if m["task"].endswith(" unsuccessful") else m["task"])
        for m in eps
    )
    base_task = base_tasks.most_common(1)[0][0]

    meta = {
        "fps": ref.get("fps"),
        "cameras": cams,
        "image_keys": [f"observation.images.{c}" for c in cams],
        "state_dim": ref.get("state_dim", 8),
        "action_dim": ref.get("action_dim", 8),
        "state_desc": ref.get("state_desc"),
        "action_desc": ref.get("action_desc"),
        "controller": ref.get("controller"),
        "gains": ref.get("gains"),
        "base_task": base_task,
        "num_episodes": n,
        "num_successful": n_succ,
        "num_unsuccessful": n - n_succ,
        "note": "Convert with scripts/convert_to_lerobot.py in a dedicated lerobot env "
                "(do NOT install lerobot into eureka; it pulls numpy>=2 and breaks Isaac Sim).",
        "recovered": True,
        "base_task_variants": dict(base_tasks),
    }
    with open(os.path.join(record_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[recover] {n} episodes ({n_succ} successful, {n - n_succ} unsuccessful) "
          f"-> {record_dir}")
    print(f"[recover] base_task='{base_task}' ({len(base_tasks)} object/site variants)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logs/collect_rsl_rl_v8")
