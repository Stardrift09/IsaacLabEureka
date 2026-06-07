# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Task manager variant that exits IsaacLab after training before running VLM evaluation.

Each Eureka iteration spawns two subprocesses sequentially:
  1. Training subprocess  — creates env, trains, saves checkpoint, closes IsaacLab.
  2. VLM eval subprocess  — creates env, loads checkpoint, calls run_single_traj_and_get_vlm_feedback, closes IsaacLab.

Because the two subprocesses never overlap, the GPU is never simultaneously loaded with
the RL environment *and* the VLM model.
"""

import glob
import inspect
import math
import multiprocessing
import os
import queue as _queue_mod
import socket
import traceback
import types
import time
from datetime import datetime
from typing import Literal

import torch

from isaaclab_eureka.utils import get_freest_gpu


# ---------------------------------------------------------------------------
# Template strings (identical to eureka_task_manager.py)
# ---------------------------------------------------------------------------

TEMPLATE_REWARD_STRING = """
from {module_name} import *
import torch

def _get_rewards(self):
    rewards_oracle = self._get_rewards_oracle()
    rewards_eureka, rewards_dict = self._get_rewards_eureka()
    self._eureka_episode_sums["eureka_total_rewards"] += rewards_eureka
    self._eureka_episode_sums["oracle_total_rewards"] += rewards_oracle
    for key in rewards_dict.keys():
        if key not in self._eureka_episode_sums:
            self._eureka_episode_sums[key] = torch.zeros(self.num_envs, device=self.device)
        self._eureka_episode_sums[key] += rewards_dict[key]
    return rewards_eureka
"""

TEMPLATE_RESET_STRING = """
from {module_name} import *

@torch.inference_mode()
def _reset_idx(self, env_ids):
    if env_ids is None or len(env_ids) == self.num_envs:
        env_ids = torch.arange(self.num_envs, device=self.device)
    extras = dict()
    {success_metric}
    # Channels set above by the success-metric injection (Eureka/success_metric and
    # Eureka/stage_*) are INSTANTANEOUS fractions in [0,1] and are authoritative.
    # The episodic-sum loop below normalizes per-second, which is correct for reward
    # magnitudes but inflates 0/1 indicators ~60x. If an LLM reward dict reuses one of
    # these reserved names as a component, the loop would clobber the correct value
    # with the inflated one (this is what pushed task_score to ~5.3). Guard against it.
    _reserved_metric_keys = set(extras.keys())
    self._reset_idx_original(env_ids)
    if not "log" in self.extras:
        self.extras["log"] = dict()
    for key in self._eureka_episode_sums.keys():
        episodic_sum_avg = torch.mean(self._eureka_episode_sums[key][env_ids])
        log_key = "Eureka/"+key
        if log_key in _reserved_metric_keys:
            log_key = log_key + "_reward"
        extras[log_key] = episodic_sum_avg / self.max_episode_length_s
        self._eureka_episode_sums[key][env_ids] = 0.0
    self.extras["log"].update(extras)
"""


# ---------------------------------------------------------------------------
# Module-level worker functions (must be picklable for multiprocessing spawn)
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> str:
    if device == "cuda":
        device_id = get_freest_gpu()
        return f"cuda:{device_id}"
    return device


def _queue_get_with_liveness(q: multiprocessing.Queue, p: multiprocessing.Process, poll_interval: float = 5.0):
    """Block until q has an item, raising RuntimeError if p dies without putting one."""
    while True:
        try:
            return q.get(timeout=poll_interval)
        except _queue_mod.Empty:
            if not p.is_alive():
                raise RuntimeError(
                    f"Subprocess (pid={p.pid}) died (exit code {p.exitcode}) "
                    "without putting a result in the queue."
                )


def _get_obs_worker(task: str, device: str, env_seed: int, obs_queue: multiprocessing.Queue):
    """One-shot subprocess: import env class (no gym.make), fetch observations source, exit.

    Avoids gym.make entirely — USD stage load and physics init are not needed just to call
    inspect.getsource on the class method.  AppLauncher is still required because the env
    module imports pxr / isaacsim.core which need Isaac Sim running.
    """
    try:
        from isaaclab.app import AppLauncher
        import importlib

        device = _resolve_device(device)
        app_launcher = AppLauncher(headless=True, device=device, enable_cameras=False)
        sim_app = app_launcher.app
        time.sleep(10)

        # pxr / isaacsim.core are now available
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401  — registers all envs in the gym registry

        # Resolve entry_point to a class WITHOUT instantiating the env
        spec = gym.spec(task)
        entry_point = spec.entry_point
        if callable(entry_point):
            env_class = entry_point
        else:
            module_path, class_name = entry_point.rsplit(":", 1)
            module = importlib.import_module(module_path)
            env_class = getattr(module, class_name)

        obs_string = inspect.getsource(env_class._get_observations)
        obs_queue.put(obs_string)

    except Exception as e:
        traceback.print_exc()
        obs_queue.put(f"# ERROR fetching _get_observations source: {e}")

    try:
        sim_app.close()
    except Exception:
        pass


def _training_worker(
    idx: int,
    task: str,
    device: str,
    env_seed: int,
    max_training_iterations: int,
    success_metric_string: str,
    checkpoint_to_resume_from: str | None,
    replay: bool,
    rl_library: str,
    reward_func_string: str,
    result_queue: multiprocessing.Queue,
    use_vlm: bool = True,
):
    """Training subprocess: create env, patch reward, train, close IsaacLab, return result."""
    from isaaclab.app import AppLauncher

    device = _resolve_device(device)
    # Cameras are only needed if VLM will run inside this process.
    # In the sequential manager VLM always runs in a separate subprocess, so this is always False.
    # The flag is kept as a parameter for future flexibility.
    app_launcher = AppLauncher(headless=True, device=device, enable_cameras=use_vlm)
    sim_app = app_launcher.app
    time.sleep(10)
    print(f"[TrainWorker {idx}] App launched")

    # These imports require Isaac Sim to be running
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(task)
    env_cfg.sim.device = device
    env_cfg.seed = env_seed
    env = gym.make(task, cfg=env_cfg)
    print(f"[TrainWorker {idx}] Env created")

    # --- Patch the Eureka reward function into the environment ---
    raw_env = env.unwrapped
    namespace: dict = {}

    raw_env._get_rewards_oracle = raw_env._get_rewards
    raw_env._reset_idx_original = raw_env._reset_idx

    template_reward = TEMPLATE_REWARD_STRING.format(module_name=raw_env.__module__)
    exec(template_reward, namespace)
    setattr(raw_env, "_get_rewards", types.MethodType(namespace["_get_rewards"], raw_env))

    template_reset = TEMPLATE_RESET_STRING.format(
        module_name=raw_env.__module__, success_metric=success_metric_string
    )
    if rl_library == "rl_games":
        template_reset = template_reset.replace("@torch.inference_mode()", "")
    exec(template_reset, namespace)
    setattr(raw_env, "_reset_idx", types.MethodType(namespace["_reset_idx"], raw_env))

    full_reward_src = f"from {raw_env.__module__} import *\nimport torch\n" + reward_func_string
    exec(full_reward_src, namespace)
    setattr(raw_env, "_get_rewards_eureka", types.MethodType(namespace["_get_rewards_eureka"], raw_env))

    raw_env._eureka_episode_sums = {
        "eureka_total_rewards": torch.zeros(raw_env.num_envs, device=raw_env.device),
        "oracle_total_rewards": torch.zeros(raw_env.num_envs, device=raw_env.device),
    }

    # --- Run training ---
    log_dir_path = None
    checkpoint_path = None
    host_name = socket.gethostname()

    try:
        if rl_library == "rsl_rl":
            from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
            from rsl_rl.runners import OnPolicyRunner
            from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

            agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
            agent_cfg.device = device
            agent_cfg.max_iterations = max_training_iterations

            log_root_path = os.path.abspath(
                os.path.join("logs", "rl_runs", "rsl_rl_eureka", agent_cfg.experiment_name)
            )
            log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_Run-{idx}" + "_" + host_name
            if agent_cfg.run_name:
                log_dir += f"_{agent_cfg.run_name}"
            log_dir_path = os.path.join(log_root_path, log_dir)

            if replay:
                raw_env.run_replay(log_dir_path)

            env_wrapped = RslRlVecEnvWrapper(env)
            runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=log_dir_path, device=agent_cfg.device)
            if checkpoint_to_resume_from:
                assert os.path.isfile(checkpoint_to_resume_from), \
                    f"Checkpoint not found: {checkpoint_to_resume_from}"
                runner.load(checkpoint_to_resume_from)
            runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

            # Find the most recently written checkpoint
            checkpoints = glob.glob(os.path.join(log_dir_path, "model_*.pt"))
            if checkpoints:
                checkpoint_path = max(checkpoints, key=os.path.getmtime)
            print(f"[TrainWorker {idx}] Training done. Checkpoint: {checkpoint_path}")

        elif rl_library == "rl_games":
            from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
            from rl_games.common import env_configurations, vecenv
            from rl_games.common.algo_observer import IsaacAlgoObserver
            from rl_games.torch_runner import Runner
            from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

            agent_cfg = load_cfg_from_registry(task, "rl_games_cfg_entry_point")
            agent_cfg["params"]["config"]["max_epochs"] = max_training_iterations
            agent_cfg["params"]["config"]["device"] = device
            agent_cfg["params"]["config"]["device_name"] = device

            log_root_path = os.path.abspath(
                os.path.join("logs", "rl_runs", "rl_games_eureka", agent_cfg["params"]["config"]["name"])
            )
            log_dir = (
                agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
                + f"_Run-{idx}" + host_name
            )
            agent_cfg["params"]["config"]["train_dir"] = log_root_path
            agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
            log_dir_path = os.path.join(log_root_path, log_dir, "summaries")

            if replay:
                raw_env.run_replay(log_dir_path)

            clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
            clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
            env_wrapped = RlGamesVecEnvWrapper(env, device, clip_obs, clip_actions)

            vecenv.register(
                "IsaacRlgWrapper",
                lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
            )
            env_configurations.register(
                "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env_wrapped}
            )
            agent_cfg["params"]["config"]["num_actors"] = env_wrapped.unwrapped.num_envs

            runner = Runner(IsaacAlgoObserver())
            runner.load(agent_cfg)
            runner.reset()
            runner.run({"train": True, "play": False, "sigma": None})
            # rl_games checkpoint discovery not implemented; VLM eval will be skipped
            checkpoint_path = None

        else:
            raise ValueError(f"RL library '{rl_library}' is not supported.")

        result = {"success": True, "log_dir": log_dir_path, "checkpoint_path": checkpoint_path}

    except Exception as e:
        # Carry the full traceback (not just str(e)) so it reaches the slurm log via the
        # main process — after AppLauncher, kit captures this worker's stdout, so the
        # print() below goes only to the kit log. The result dict travels back over the
        # queue and is printed by the parent, which is not under kit's stdout capture.
        result = {"success": False, "exception": traceback.format_exc()}
        print(traceback.format_exc())

    # Put result BEFORE closing sim — sim_app.close() can hang indefinitely.
    result_queue.put(result)
    try:
        env.close()
    except Exception:
        pass
    try:
        sim_app.close()
    except Exception:
        pass


def _vlm_eval_worker(
    task: str,
    device: str,
    checkpoint_path: str,
    output_file: str,
    pic_save_dir: str,
    result_queue: multiprocessing.Queue,
):
    """VLM evaluation subprocess: fresh IsaacLab instance, load policy, call VLM, exit."""
    sim_app = None
    env = None
    try:
        from isaaclab.app import AppLauncher

        device = _resolve_device(device)
        app_launcher = AppLauncher(headless=True, device=device, enable_cameras=True)
        sim_app = app_launcher.app
        time.sleep(10)
        print(f"[VLMEval] App launched, loading checkpoint: {checkpoint_path}")

        # These imports require Isaac Sim to be running
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        env_cfg = parse_env_cfg(task)
        env_cfg.sim.device = device
        env_cfg.scene.num_envs = 1
        env_cfg.camera_sensor_record = True
        env = gym.make(task, cfg=env_cfg)

        agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        agent_cfg.device = device

        env_wrapped = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint_path)

        policy = runner.get_inference_policy(device=env.unwrapped.device)
        policy_nn = runner.alg.policy

        os.makedirs(pic_save_dir, exist_ok=True)

        with torch.inference_mode():
            output_text = env.unwrapped.run_single_traj_and_get_vlm_feedback(
                policy_nn, policy, pic_save_dir
            )

        with open(output_file, "w") as f:
            f.write(output_text)
        print(f"[VLMEval] Output written to: {output_file}")

        # Put result BEFORE closing sim — sim_app.close() can hang indefinitely.
        result_queue.put({"success": True})

    except Exception as e:
        print(f"[VLMEval] Exception: {e}")
        print(traceback.format_exc())
        # Always put a result so the parent's queue.get() unblocks.
        result_queue.put({"success": False, "exception": str(e)})

    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        try:
            if sim_app is not None:
                sim_app.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Manager class
# ---------------------------------------------------------------------------

class EurekaTaskManagerSequentialVLM:
    """Like EurekaTaskManager but exits IsaacLab between RL training and VLM evaluation.

    Each call to :meth:`train` spawns two sequential subprocesses per run:
    1. A *training* subprocess that creates the environment, trains with the
       Eureka reward, saves a checkpoint, then **closes IsaacLab**.
    2. A *VLM eval* subprocess that starts a fresh IsaacLab instance, loads the
       checkpoint, calls ``run_single_traj_and_get_vlm_feedback``, writes the
       output to ``vlm_output.txt``, then exits.

    This prevents the GPU from being simultaneously loaded with the RL
    environment and the VLM model.
    """

    def __init__(
        self,
        task: str,
        checkpoint_to_resume_from: str | None,
        rl_library: Literal["rsl_rl", "rl_games"] = "rsl_rl",
        num_processes: int = 1,
        device: str = "cuda",
        env_seed: int = 42,
        max_training_iterations: int = 100,
        success_metric_string: str = "",
        replay: bool = False,
        use_vlm: bool = True,
    ):
        self._task = task
        self.checkpoint_to_resume_from = checkpoint_to_resume_from
        self._rl_library = rl_library
        self._num_processes = num_processes
        self._device = device
        self._env_seed = env_seed
        self._max_training_iterations = max_training_iterations
        self._success_metric_string = success_metric_string
        self.replay = replay
        self.use_vlm = use_vlm

        # Fetch the observations string via a one-shot subprocess so the main
        # process (Eureka) can build the LLM prompt before any training starts.
        print("[INFO]: Fetching observations method via one-shot subprocess...")
        obs_queue: multiprocessing.Queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=_get_obs_worker, args=(task, device, env_seed, obs_queue))
        p.start()
        # Get result first, then reap — subprocess may hang in sim_app.close().
        self._get_observations_as_string: str = _queue_get_with_liveness(obs_queue, p)
        p.join(timeout=600)
        if p.is_alive():
            print("[INFO]: Obs worker still alive after 600 s, killing.")
            p.terminate()
            p.join(timeout=30)
            if p.is_alive():
                p.kill()
                p.join()
        print("[INFO]: Observations method fetched.")

    @property
    def get_observations_method_as_string(self) -> str:
        """The _get_observations method of the environment as a string."""
        return self._get_observations_as_string

    def close(self):
        """No persistent worker processes to clean up."""
        pass

    def train(self, get_rewards_method_as_string: list[str]) -> list[dict]:
        """Train with the given reward functions, running VLM eval in a separate subprocess.

        Args:
            get_rewards_method_as_string: One reward method string per parallel run.
        Returns:
            List of result dicts with keys ``success``, ``log_dir``, and optionally
            ``exception``.
        """
        if len(get_rewards_method_as_string) != self._num_processes:
            raise ValueError(
                f"Number of reward methods ({len(get_rewards_method_as_string)}) does not match "
                f"number of processes ({self._num_processes})."
            )

        results: list[dict | None] = [None] * self._num_processes

        for idx, reward_str in enumerate(get_rewards_method_as_string):
            # Stagger multiple runs to avoid simultaneous GPU initialisation.
            if idx > 0:
                time.sleep(90 * idx)

            # ------------------------------------------------------------------
            # Step 1: Training subprocess
            # ------------------------------------------------------------------
            train_queue: multiprocessing.Queue = multiprocessing.Queue()
            p = multiprocessing.Process(
                target=_training_worker,
                args=(
                    idx,
                    self._task,
                    self._device,
                    self._env_seed,
                    self._max_training_iterations,
                    self._success_metric_string,
                    self.checkpoint_to_resume_from,
                    self.replay,
                    self._rl_library,
                    reward_str,
                    train_queue,
                    False,  # use_vlm=False: training never calls VLM, so no cameras needed
                ),
            )
            p.start()
            # Get result first (worker puts it before closing sim), then reap the process.
            try:
                result = _queue_get_with_liveness(train_queue, p)
            except RuntimeError as exc:
                print(f"[TrainWorker {idx}] {exc}")
                result = {"success": False, "exception": str(exc)}
            p.join(timeout=600)
            if p.is_alive():
                print(f"[TrainWorker {idx}] Still alive after 600 s, terminating.")
                p.terminate()
                p.join(timeout=30)
                if p.is_alive():
                    p.kill()
                    p.join()

            # ------------------------------------------------------------------
            # Step 2: VLM evaluation subprocess (only when use_vlm=True, rsl_rl, checkpoint exists)
            # ------------------------------------------------------------------
            if self.use_vlm and result.get("success") and result.get("checkpoint_path") and self._rl_library == "rsl_rl":
                vlm_queue: multiprocessing.Queue = multiprocessing.Queue()
                pic_save_dir = os.path.join(result["log_dir"], "pictures")
                vlm_output_file = os.path.join(result["log_dir"], "vlm_output.txt")

                vp = multiprocessing.Process(
                    target=_vlm_eval_worker,
                    args=(
                        self._task,
                        self._device,
                        result["checkpoint_path"],
                        vlm_output_file,
                        pic_save_dir,
                        vlm_queue,
                    ),
                )
                vp.start()
                try:
                    vlm_result = _queue_get_with_liveness(vlm_queue, vp)
                except RuntimeError as exc:
                    print(f"[VLMEval] {exc}")
                    vlm_result = {"success": False, "exception": str(exc)}
                vp.join(timeout=600)
                if vp.is_alive():
                    print("[VLMEval] Still alive after 600 s, terminating.")
                    vp.terminate()
                    vp.join(timeout=30)
                    if vp.is_alive():
                        vp.kill()
                        vp.join()
                if not vlm_result.get("success"):
                    # Ensure the file exists so eureka.py does not crash.
                    vlm_output_file_path = os.path.join(result["log_dir"], "vlm_output.txt")
                    if not os.path.exists(vlm_output_file_path):
                        with open(vlm_output_file_path, "w") as f:
                            f.write("VLM evaluation did not produce output.")

            elif result.get("success") and result.get("log_dir"):
                # VLM skipped (use_vlm=False, rl_games, or no checkpoint) — write placeholder.
                vlm_output_file_path = os.path.join(result["log_dir"], "vlm_output.txt")
                if not os.path.exists(vlm_output_file_path):
                    reason = "VLM disabled." if not self.use_vlm else "VLM evaluation not available for this run."
                    with open(vlm_output_file_path, "w") as f:
                        f.write(reason)

            results[idx] = result

        return results
