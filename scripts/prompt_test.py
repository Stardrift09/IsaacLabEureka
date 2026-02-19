TEMPLATE_RESET_STRING = """
from {module_name} import *

@torch.inference_mode()
def _reset_idx(self, env_ids):
    if env_ids is None or len(env_ids) == self.num_envs:
        env_ids = torch.arange(self.num_envs, device=self.device)
    extras = dict()
    # This needs to happen before self._reset_idx_original(env_ids) because it will reset buffers that might be needed
    {success_metric}
    print(f"success_metric in reset idx: {extras['Eureka/success_metric']}")
    self._reset_idx_original(env_ids)
    if not "log" in self.extras:
        self.extras["log"] = dict()
    for key in self._eureka_episode_sums.keys():
        episodic_sum_avg = torch.mean(self._eureka_episode_sums[key][env_ids])
        extras["Eureka/"+key] = episodic_sum_avg / self.max_episode_length_s
        self._eureka_episode_sums[key][env_ids] = 0.0
    self.extras["log"].update(extras)
"""

success_metric =          """
    self.extras['Eureka/success_metric'] = (self.target_object.data.root_pos_w[env_ids, 2] > 0.4).float().mean()"""

print(TEMPLATE_RESET_STRING.format(module_name="source.isaaclab_eureka.isaaclab_eureka.managers.eureka_task_manager", success_metric=success_metric))