#!/usr/bin/env python3
"""Local LLM watcher — polls a directory for request files, calls OpenAI, writes responses.

The watch_dir must be a path that is shared with the BSC compute node, typically an
sshfs mount of the GPFS llm_requests directory:

    mkdir -p /mnt/bsc_llm
    sshfs tum634073@alogin1.bsc.es:/gpfs/scratch/ehpc565/IsaacLabEureka/llm_requests /mnt/bsc_llm
    OPENAI_API_KEY=sk-... python scripts/local_llm_watcher.py /mnt/bsc_llm --interval 30

Multiple concurrent BSC jobs are handled safely — each request has a unique UUID.
"""

import argparse
import json
import time
from pathlib import Path

import openai


def find_pending_requests(watch_dir: Path) -> list:
    pending = []
    for req in watch_dir.rglob("*_request.json"):
        resp = req.parent / req.name.replace("_request.json", "_response.json")
        done = req.parent / req.name.replace("_request.json", "_request.done.json")
        if not resp.exists() and not done.exists():
            pending.append(req)
    return pending


def process_request(req_path: Path, client: openai.OpenAI) -> None:
    data = json.loads(req_path.read_text())
    print(f"[watcher] Processing {req_path.name}  model={data['model']}  n={data['n']}")

    responses = client.chat.completions.create(
        model=data["model"],
        messages=data["messages"],
        temperature=data["temperature"],
        n=data["n"],
    )

    raw_outputs = [choice.message.content for choice in responses.choices]
    resp_path = req_path.parent / req_path.name.replace("_request.json", "_response.json")
    resp_path.write_text(json.dumps({"raw_outputs": raw_outputs}, indent=2))

    done_path = req_path.parent / req_path.name.replace("_request.json", "_request.done.json")
    req_path.rename(done_path)
    print(f"[watcher] Done → {resp_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch for LLM request files and respond via OpenAI.")
    parser.add_argument("watch_dir", help="Directory to watch (sshfs mount or local path).")
    parser.add_argument("--interval", type=int, default=300, help="Poll interval in seconds (default: 30).")
    args = parser.parse_args()

    watch_dir = Path(args.watch_dir)
    if not watch_dir.exists():
        raise SystemExit(f"watch_dir does not exist: {watch_dir}")

    client = openai.OpenAI()
    print(f"[watcher] Watching {watch_dir}  interval={args.interval}s")

    while True:
        try:
            for req in find_pending_requests(watch_dir):
                try:
                    process_request(req, client)
                except Exception as e:
                    print(f"[watcher] Error on {req.name}: {e}")
        except Exception as e:
            print(f"[watcher] Scan error: {e}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
