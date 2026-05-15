# Agent Sandbox Guide

How to let an agent run code without wrecking your host system.

---

## What a sandbox actually does

Five attack surfaces to close:

| Surface | Risk | Mechanism |
|---------|------|-----------|
| Filesystem | Agent overwrites/deletes host files | isolation, read-only mounts, overlayfs |
| Resources | Agent eats all RAM/CPU/disk | cgroups, ulimit |
| Network | Agent calls external APIs, exfiltrates data | network namespace, firewall |
| Processes | Agent spawns infinite children | PID namespace, RLIMIT_NPROC |
| Syscalls | Agent does dangerous kernel calls | seccomp, capabilities |

You don't need to close all five. Pick based on threat level.

---

## Threat levels → what you need

### Level 1 — Trusted agent, prevent accidents

Agent is your own code. You just don't want runaway loops or accidental `rm -rf`.

**Tools:** `ulimit`, `timeout`, git worktree for rollback. No container needed.

```python
import resource, subprocess

# Memory cap: 4 GB
resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
# Process cap: no fork bombs
resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))
# File size cap: 500 MB
resource.setrlimit(resource.RLIMIT_FSIZE, (500 * 1024**2, 500 * 1024**2))

result = subprocess.run(
    ["python", "agent.py"],
    timeout=120,          # watchdog equivalent — kills after 120s
    capture_output=True,
)
```

Git worktree gives you cheap rollback:
```bash
git worktree add /tmp/agent-workspace -b agent-run-$(date +%s)
# agent works inside /tmp/agent-workspace
# diff: git -C /tmp/agent-workspace diff
# undo: git -C /tmp/agent-workspace checkout .
# clean up: git worktree remove /tmp/agent-workspace
```

---

### Level 2 — Container (Docker/Podman)

Agent is semi-trusted or runs user-supplied code. Standard choice.

```bash
docker run --rm \
  --memory=4g \
  --memory-swap=4g \        # swap = memory = no swap
  --cpus=2 \
  --pids-limit=256 \        # no fork bombs
  --network=none \          # no outbound network
  --read-only \             # root FS read-only
  --tmpfs /tmp:size=500m \  # writable scratch
  --tmpfs /root:size=100m \
  -v $(pwd)/workspace:/workspace:rw \
  --security-opt=no-new-privileges \
  my-agent-image \
  python agent.py
```

If agent needs internet (e.g. pip install) but not arbitrary egress:
```bash
# Custom bridge — whitelist only what you need
docker network create --driver bridge \
  --opt com.docker.network.bridge.name=agent-br \
  agent-net

# Then use iptables to restrict egress
iptables -I DOCKER-USER -i agent-br ! -d 172.18.0.0/16 -j DROP
# Whitelist specific IP:
iptables -I DOCKER-USER -i agent-br -d 151.101.0.0/17 -j ACCEPT  # pypi
```

**Dockerfile for agent image:**
```dockerfile
FROM python:3.11-slim

# Drop all capabilities, only add what's needed
RUN useradd -m -u 1000 agent
USER agent
WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "agent.py"]
```

---

### Level 3 — Microvm (Firecracker / gVisor)

Agent runs untrusted code (user submissions, LLM-generated arbitrary code). Container escape is a real concern.

**gVisor** — intercepts all syscalls in userspace. Drop-in with Docker:
```bash
# Install gVisor runsc
sudo apt-get install runsc

# Configure Docker daemon (/etc/docker/daemon.json):
# { "runtimes": { "runsc": { "path": "/usr/bin/runsc" } } }

docker run --runtime=runsc --rm \
  --memory=2g --network=none \
  my-agent-image python agent.py
```

**Firecracker** — real microVM, ~150ms startup. Used by AWS Lambda, Fly.io.
Too low-level to manage directly; use [Kata Containers](https://katacontainers.io/) or managed services (Modal, E2B).

---

## Watchdog and alternatives

### What watchdog does
`watchdog` (the Python lib) watches **filesystem events** — not process health. People confuse the name.

For process health / timeout enforcement you have several options:

| Mechanism | How | Use when |
|-----------|-----|----------|
| `subprocess.run(timeout=N)` | Python kills process after N sec | simple one-shot subprocess |
| `timeout` command | `timeout 60s python agent.py` | shell scripts |
| `SIGALRM` | `signal.alarm(N)` in Python | single-threaded, same process |
| `threading.Timer` | Python timer fires kill | async or multi-step agents |
| systemd `TimeoutSec=` | unit file | long-running services |
| cgroup CPU quota | hard CPU time cap | prevent slow-burn loops |
| Custom watchdog process | separate process polls target | distributed / multi-agent |

**Custom watchdog process (robust):**
```python
import subprocess, time, os, signal

class AgentWatchdog:
    def __init__(self, timeout_sec=120, memory_mb=4096):
        self.timeout = timeout_sec
        self.memory_mb = memory_mb

    def run(self, cmd):
        proc = subprocess.Popen(cmd)
        start = time.time()

        while proc.poll() is None:
            elapsed = time.time() - start
            if elapsed > self.timeout:
                proc.kill()
                raise TimeoutError(f"Agent exceeded {self.timeout}s")

            # Check memory via /proc
            try:
                with open(f"/proc/{proc.pid}/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            kb = int(line.split()[1])
                            if kb > self.memory_mb * 1024:
                                proc.kill()
                                raise MemoryError(f"Agent exceeded {self.memory_mb}MB")
            except FileNotFoundError:
                pass  # process already dead

            time.sleep(1)

        return proc.returncode

dog = AgentWatchdog(timeout_sec=120, memory_mb=2048)
dog.run(["python", "agent.py"])
```

---

## Filesystem rollback strategies

| Strategy | Cost | How |
|----------|------|-----|
| Git worktree | free | `git worktree add`; `git checkout .` to undo |
| tmpfs | free | mount `tmpfs`; destroyed on process exit |
| overlayfs | free | copy-on-write layer; discard upper layer |
| Docker `--rm` | free | container FS deleted on exit |
| LVM snapshot | disk space | snapshot volume before run; rollback = restore |
| btrfs/ZFS snapshot | disk space | `btrfs subvolume snapshot` |

**overlayfs without Docker:**
```bash
# Base dir (read-only original workspace)
BASE=/home/user/project
UPPER=/tmp/agent-upper    # agent writes here
WORK=/tmp/agent-work      # overlayfs internals
MERGED=/tmp/agent-merged  # what agent sees

mkdir -p $UPPER $WORK $MERGED

mount -t overlay overlay \
  -o lowerdir=$BASE,upperdir=$UPPER,workdir=$WORK \
  $MERGED

# Run agent with MERGED as working dir
# Discard changes: rm -rf $UPPER/* (or just delete $UPPER)
# Inspect changes: ls $UPPER
```

---

## Beyond watchdog — full checklist

### Resource controls
```bash
# In /etc/security/limits.conf or via pam_limits
agent  hard  nproc   256
agent  hard  nofile  1024
agent  hard  fsize   524288   # 512MB in KB

# Or inline with bash ulimit:
(
  ulimit -u 256       # max processes
  ulimit -n 1024      # max open files
  ulimit -f 524288    # max file size (KB)
  ulimit -v 4194304   # max virtual memory (KB = 4GB)
  python agent.py
)
```

### seccomp (syscall filter)
```python
# Using python-prctl + seccomp
import prctl
# Or use Docker's built-in seccomp profile (default blocks ~40 dangerous syscalls)
# Custom profile: --security-opt seccomp=policy.json
```

### Drop Linux capabilities
```bash
docker run \
  --cap-drop=ALL \
  --cap-add=NET_BIND_SERVICE \   # only if needed
  ...
```

### AppArmor profile (Ubuntu default Docker uses this)
```
# /etc/apparmor.d/agent-profile
profile agent-profile flags=(attach_disconnected) {
  /workspace/** rw,
  /tmp/** rw,
  /usr/bin/python3 ix,
  deny /etc/** rw,
  deny /home/** rw,
  deny /root/** rw,
}
```

---

## Managed cloud sandboxes (skip the ops)

| Service | What | Use when |
|---------|------|----------|
| [E2B](https://e2b.dev) | API for sandboxed code exec | LLM agent tooling |
| [Modal](https://modal.com) | Serverless GPU/CPU containers | ML workloads |
| [Daytona](https://daytona.io) | Dev environment sandboxes | IDE-like agent work |
| [Codex/OpenAI sandbox] | Built-in for Codex CLI | if using OpenAI |
| AWS Lambda | Firecracker microVMs | short-lived tasks |

E2B example (Python SDK):
```python
from e2b_code_interpreter import Sandbox

with Sandbox() as sbx:
    result = sbx.run_code("import os; print(os.listdir('/'))")
    print(result.text)
# Sandbox auto-destroyed on exit
```

---

## Decision tree

```
Is agent running untrusted user code?
├── Yes → microVM (gVisor or Firecracker/Kata) or E2B
└── No: Is agent semi-trusted (LLM-generated code)?
    ├── Yes → Docker + --network=none + --read-only + --cap-drop=ALL
    └── No: Is agent your own trusted code?
        ├── Need rollback? → git worktree + ulimit + subprocess timeout
        └── Just resource limits? → ulimit + subprocess(timeout=N)
```

---

## Minimal working example (Level 1)

```python
import subprocess, resource, tempfile, os

def run_agent_sandboxed(agent_script: str, workspace: str, timeout_sec=60):
    """Run agent with resource limits and timeout. No container needed."""

    def preexec():
        # 2GB memory
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
        # 256 processes
        resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))
        # 500MB files
        resource.setrlimit(resource.RLIMIT_FSIZE, (500 * 1024**2, 500 * 1024**2))

    try:
        result = subprocess.run(
            ["python", agent_script],
            cwd=workspace,
            timeout=timeout_sec,
            preexec_fn=preexec,
            capture_output=True,
            text=True,
        )
        return result
    except subprocess.TimeoutExpired:
        print(f"Agent killed after {timeout_sec}s")
        return None
```

## Minimal working example (Level 2 — Docker)

```python
import subprocess, pathlib

def run_agent_in_docker(workspace: str, timeout_sec=120):
    workspace = pathlib.Path(workspace).resolve()

    cmd = [
        "docker", "run", "--rm",
        "--memory=2g",
        "--cpus=1",
        "--pids-limit=256",
        "--network=none",
        "--read-only",
        "--tmpfs", "/tmp:size=200m",
        "--security-opt=no-new-privileges",
        "--cap-drop=ALL",
        f"-v{workspace}:/workspace:rw",
        "-w", "/workspace",
        "my-agent-image",
        "python", "agent.py",
    ]

    try:
        return subprocess.run(cmd, timeout=timeout_sec, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", "$(docker ps -q --filter ancestor=my-agent-image)"],
                       shell=True)
        return None
```

---

## Key parameters to tune

| Parameter | Location | Effect |
|-----------|----------|--------|
| `timeout` | subprocess / docker / systemd | wall-clock kill |
| `RLIMIT_AS` | ulimit / cgroup memory.limit | virtual memory cap |
| `RLIMIT_NPROC` | ulimit / cgroup pids.max | fork bomb prevention |
| `--network=none` | Docker | no outbound calls |
| `--read-only` | Docker | immutable root FS |
| `--cap-drop=ALL` | Docker | no privilege escalation |
| overlayfs upper dir | mount | inspect/discard agent writes |
