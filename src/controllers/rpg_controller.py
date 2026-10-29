"""RPG controller using MACPro's EpisodeBatch and action selection contract."""

from pathlib import Path

import torch

from components.action_selectors import REGISTRY as action_REGISTRY
from modules.agents.rpg_agent import RPGAgent


class RPGMAC:
    def __init__(self, train_tasks, task2scheme, task2args, main_args, task2decomposer=None):
        self.train_tasks = list(train_tasks)
        self.task2scheme = task2scheme
        self.task2args = task2args
        self.main_args = main_args
        self.task2n_agents = {task: task2args[task].n_agents for task in train_tasks}
        if task2decomposer is None:
            task2decomposer = getattr(main_args, "rpg_decomposers", None)
        if task2decomposer is None:
            task2decomposer = {task: task2args[task].rpg_decomposer for task in train_tasks}
        self.task2decomposer = task2decomposer
        self.surrogate_decomposer = next(iter(task2decomposer.values()))
        self.agent = RPGAgent(task2decomposer, main_args)
        self.action_selector = action_REGISTRY[getattr(main_args, "action_selector", "epsilon_greedy")](main_args)
        self.hidden_states = None
        self.last_entropy = None

    def begin_task(self, task, previous_task=None):
        self.agent.begin_task(task, previous_task)
        self.hidden_states = None
        self.last_entropy = None

    def init_hidden(self, batch_size, task):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(
            batch_size, self.task2n_agents[task], -1)
        self.last_entropy = None

    def forward(self, ep_batch, task, t, test_mode=False, embedding_task=None, **kwargs):
        del test_mode, kwargs
        n_agents = self.task2n_agents[task]
        obs = ep_batch["obs"][:, t].reshape(ep_batch.batch_size * n_agents, -1)
        last_actions = None
        if getattr(self.task2args[task], "obs_last_action", True):
            last_actions = (torch.zeros_like(ep_batch["actions_onehot"][:, t]) if t == 0
                            else ep_batch["actions_onehot"][:, t - 1])
            last_actions = last_actions.reshape(ep_batch.batch_size * n_agents, -1)
        if self.hidden_states is None:
            self.init_hidden(ep_batch.batch_size, task)
        q, hidden, entropy = self.agent(obs, last_actions, self.hidden_states, task, embedding_task)
        self.hidden_states = hidden.reshape(ep_batch.batch_size, n_agents, -1)
        self.last_entropy = entropy.reshape(ep_batch.batch_size, n_agents)
        return q.reshape(ep_batch.batch_size, n_agents, -1)

    def select_actions(self, ep_batch, t_ep, t_env, task, bs=slice(None), test_mode=False,
                       embedding_task=None, **kwargs):
        with torch.no_grad():
            q = self.forward(ep_batch, task, t_ep, test_mode, embedding_task, **kwargs)
            available = ep_batch["avail_actions"][:, t_ep]
            return self.action_selector.select_action(q[bs], available[bs], t_env, test_mode)

    def parameters(self):
        return self.agent.parameters()

    def to(self, device):
        self.agent.to(device)
        self.hidden_states = None
        self.last_entropy = None
        return self

    def cuda(self):
        return self.to("cuda")

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.hidden_states = None
        self.last_entropy = None

    def save_models(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.agent.state_dict(), path / "agent.th")

    def load_models(self, path):
        device = next(self.agent.parameters()).device
        self.agent.load_state_dict(torch.load(Path(path) / "agent.th", map_location=device, weights_only=True))
        self.hidden_states = None
        self.last_entropy = None
