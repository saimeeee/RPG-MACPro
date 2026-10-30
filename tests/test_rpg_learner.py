"""Numerical regression tests for RPG's continual-learning equations."""

from pathlib import Path
import sys
from types import SimpleNamespace as SN

import pytest
import torch as th
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers.rpg_controller import RPGMAC
from learners.rpg_learner import RPGQLearner


class TinyDecomposer:
    """Small entity layout, not a replacement benchmark environment."""
    own_obs_dim = obs_nf_al = obs_nf_en = state_nf_al = state_nf_en = 2
    n_actions_no_attack = 2
    state_last_action = state_timestep_number = False
    timestep_number_state_dim = 0

    def __init__(self, n_agents=2, n_enemies=1):
        self.n_agents, self.n_enemies = n_agents, n_enemies
        self.n_actions = 2 + n_enemies
        self.obs_dim = 2 * (n_agents + n_enemies)
        self.state_dim = 2 * (n_agents + n_enemies)

    def decompose_obs(self, obs):
        tokens = obs.reshape(*obs.shape[:-1], -1, 2)
        return (tokens[..., 0, :],
                list(tokens[..., self.n_agents:, :].unbind(-2)),
                list(tokens[..., 1:self.n_agents, :].unbind(-2)))

    def decompose_state(self, state):
        tokens = state.reshape(*state.shape[:-1], -1, 2)
        return (list(tokens[..., :self.n_agents, :].unbind(-2)),
                list(tokens[..., self.n_agents:, :].unbind(-2)), [], state[..., :0])


def args():
    return SN(rnn_hidden_dim=4, entity_embed_dim=4, attn_embed_dim=2,
              mixing_embed_dim=4, rpg_task_embed_dim=3, rpg_head_hidden_dim=4,
              rpg_hyper_hidden_dim=4, rpg_mixer_heads=2, rpg_alpha_init=0.5,
              rpg_alpha_att=0.02, rpg_lambda_rc=2.0, rpg_lambda_hy=0.3,
              rpg_return_probe_episodes=2, rpg_importance_samples=2,
              action_selector="epsilon_greedy", epsilon_start=1.0,
              epsilon_finish=0.05, epsilon_anneal_time=50, test_greedy=True,
              obs_last_action=True, gamma=0.9, target_update_interval=100,
              lr=0.001, double_q=True)


def batch(decomposer, batch_size=2, length=4):
    scheme = {
        "state": {"vshape": decomposer.state_dim},
        "obs": {"vshape": decomposer.obs_dim, "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (decomposer.n_actions,), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    result = EpisodeBatch(scheme, {"agents": decomposer.n_agents}, batch_size, length,
                          preprocess={"actions": ("actions_onehot", [OneHot(decomposer.n_actions)])})
    data = result.data.transition_data
    data["state"].uniform_(0.1, 1.0)
    data["obs"].uniform_(0.1, 1.0)
    data["avail_actions"].fill_(1)
    data["actions_onehot"][..., 0] = 1
    data["reward"][:, :-1].fill_(0.5)
    data["terminated"][:, -2] = 1
    data["filled"].fill_(1)
    return result


def learner():
    decomposers = {"a": TinyDecomposer(), "b": TinyDecomposer(3, 2)}
    config = args()
    task_args = {name: SN(n_agents=d.n_agents, obs_last_action=True)
                 for name, d in decomposers.items()}
    mac = RPGMAC(list(decomposers), {}, task_args, config, decomposers)
    return RPGQLearner(mac, None, config)


def test_return_discount_freezes_after_first_n_episodes():
    obj = RPGQLearner.__new__(RPGQLearner)
    obj.args, obj.return_samples, obj.gamma_p = args(), [], 0.0
    assert obj.observe_return(0.2) == pytest.approx(0.2)
    assert obj.observe_return(0.8) == pytest.approx(0.5)
    assert obj.observe_return(1.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        obj.observe_return(float("nan"))
    with pytest.raises(ValueError):
        obj.observe_return(-0.1)


def test_importance_uses_squared_individual_gradients_and_global_max():
    obj = RPGQLearner.__new__(RPGQLearner)
    relation = nn.Linear(2, 1)
    with th.no_grad():
        relation.weight.copy_(th.tensor([[2.0, 1.0]]))
        relation.bias.fill_(3.0)
    agent = nn.Module()
    agent.relation_capturer = relation
    agent.hypernetwork = nn.Linear(1, 2)
    agent.generated_parameters = lambda task: agent.hypernetwork(th.tensor([1.0 if task == "a" else 2.0]))
    obj.mac = SN(agent=agent, parameters=agent.parameters, hidden_states=None, last_entropy=None)
    obj.args = args()
    obj.optimiser = th.optim.SGD(agent.parameters(), lr=0.1)
    obj.current_task, obj.completed_tasks = "a", []
    obj.gamma_p, obj.return_samples = 0.0, []
    obj.relation_anchors, obj.relation_importance, obj.hyper_targets = {}, {}, {}
    data = batch(TinyDecomposer())
    data.data.transition_data["reward"][0].fill_(1)
    data.data.transition_data["reward"][1].fill_(-1)

    def analytic_loss(episode, task):
        loss = episode["reward"][0, 0, 0] * (relation.weight.sum() + 2 * relation.bias.sum())
        return loss, loss * 0, {}

    obj._td_loss = analytic_loss
    result = obj.end_task("a", data)
    assert result["importance_max_before_normalization"] == pytest.approx(36)
    importance = obj.relation_importance["a"]
    th.testing.assert_close(importance["weight"], th.tensor([[4 / 36, 1 / 36]]))
    th.testing.assert_close(importance["bias"], th.ones(1))
    old_target = obj.hyper_targets["a"].clone()
    with th.no_grad():
        agent.hypernetwork.weight.add_(1)
    obj.current_task = "b"
    obj.end_task("b", data)
    # Eq. 10 compares to the latest boundary, not the original task-a head.
    assert not th.equal(old_target, obj.hyper_targets["a"])
    th.testing.assert_close(obj.hyper_targets["a"], agent.generated_parameters("a"))
    assert set(obj.relation_anchors) == {"a", "b"}


def test_relation_regularization_sums_all_task_anchors():
    obj = learner()
    named = dict(obj.mac.agent.relation_capturer.named_parameters())
    obj.completed_tasks = ["a", "b"]
    for task, offset in [("a", 1.0), ("b", 2.0)]:
        obj.relation_anchors[task] = {name: p.detach().clone() - offset for name, p in named.items()}
        obj.relation_importance[task] = {name: th.ones_like(p) for name, p in named.items()}
    rc, hy = obj.regularization_losses()
    expected = sum(p.numel() for p in named.values()) * (1 ** 2 + 2 ** 2) * obj.args.rpg_lambda_rc / 2
    assert rc.item() == pytest.approx(expected)
    assert hy.item() == 0


def test_hyper_regularizer_is_sum_of_squared_outputs_averaged_over_old_tasks():
    obj = learner()
    obj.begin_task('a')
    reference = {task: obj.mac.agent.generated_parameters(task).detach().clone()
                 for task in ('a', 'b')}
    obj.hyper_targets = {'a': reference['a'] - 1, 'b': reference['b'] - 2}
    _, loss = obj.regularization_losses()
    size = reference['a'].numel()
    assert loss.item() == pytest.approx(obj.args.rpg_lambda_hy / 2 * size * 5)
    loss.backward()
    assert sum(p.grad.abs().sum().item() for p in obj.mac.agent.hypernetwork.parameters()
               if p.grad is not None) > 0


def test_padding_does_not_change_td_or_attention_loss():
    th.manual_seed(7)
    obj = learner()
    obj.begin_task("a")
    original = batch(obj.task2decomposer["a"])
    padded = batch(obj.task2decomposer["a"], length=7)
    for name, tensor in original.data.transition_data.items():
        padded.data.transition_data[name][:, :4] = tensor
    for name in ["state", "obs", "reward"]:
        padded.data.transition_data[name][:, 4:] = 1000
    padded.data.transition_data["filled"][:, 4:] = 0
    padded.data.transition_data["avail_actions"][:, 4:] = 0
    loss, entropy, _ = obj._td_loss(original, "a")
    padded_loss, padded_entropy, _ = obj._td_loss(padded, "a")
    th.testing.assert_close(loss, padded_loss)
    th.testing.assert_close(entropy, padded_entropy)


@pytest.mark.parametrize("terminal, expected_target", [(False, 4.6), (True, 1.0)])
def test_double_q_masks_actions_and_only_true_terminals_stop_bootstrap(terminal, expected_target):
    class FixedMAC:
        def __init__(self, target=False):
            self.target = target

        def init_hidden(self, batch_size, task):
            pass

        def forward(self, data, task, t):
            self.last_entropy = th.zeros(data.batch_size, 1)
            # Unavailable action zero must never be picked. The online net
            # chooses action two while the target net prefers action one.
            values = [1.0, 0.0, 0.0] if t == 0 else ([1000.0, 20.0, 4.0] if self.target else [100.0, 2.0, 3.0])
            return th.tensor(values).reshape(1, 1, 3)

    obj = RPGQLearner.__new__(RPGQLearner)
    obj.args = args()
    obj.mac, obj.target_mac = FixedMAC(), FixedMAC(target=True)
    obj.task2decomposer = {"a": None}
    obj.mixer = obj.target_mixer = lambda qs, states, decomposer: qs.sum(-1, keepdim=True)
    data = batch(TinyDecomposer(1, 1), batch_size=1, length=2)
    data.data.transition_data["reward"][0, 0] = 1.0
    data.data.transition_data["terminated"][0, 0] = int(terminal)
    data.data.transition_data["avail_actions"][0, 1, 0, 0] = 0
    td, _, details = obj._td_loss(data, "a")
    assert details["target_mean"].item() == pytest.approx(expected_target)
    assert td.item() == pytest.approx((1 - expected_target) ** 2, abs=1e-5)


def test_targets_have_no_gradients_and_task_embedding_freezing():
    obj = learner()
    obj.begin_task("a")
    first = batch(obj.task2decomposer["a"])
    obj.train(first, 1, 1, "a")
    obj.end_task("a", first)
    old_embedding = obj.mac.agent.task_embeddings["t0"].detach().clone()
    obj.begin_task("b", "a")
    obj.observe_return(0.8)
    obj.observe_return(0.6)
    metrics = obj.train(batch(obj.task2decomposer["b"]), 2, 2, "b")
    assert all(th.isfinite(th.tensor(value)) for value in metrics.values())
    th.testing.assert_close(old_embedding, obj.mac.agent.task_embeddings["t0"])
    assert not obj.mac.agent.task_embeddings["t0"].requires_grad
    assert obj.mac.agent.task_embeddings["t1"].requires_grad
    assert all(p.grad is None for p in obj.target_mac.parameters())
    assert all(p.grad is None for p in obj.target_mixer.parameters())


def test_checkpoint_restores_regularizers_targets_and_continued_update(tmp_path):
    th.manual_seed(11)
    original = learner()
    original.begin_task("a")
    first = batch(original.task2decomposer["a"])
    original.train(first, 1, 1, "a")
    original.end_task("a", first)
    original.begin_task("b", "a")
    original.observe_return(0.7)
    original.observe_return(0.9)
    second = batch(original.task2decomposer["b"])
    original.train(second, 2, 2, "b")
    before = original.regularization_losses()
    assert before[0].item() > 0
    assert before[1].item() > 0
    original.progress = {"task_index": 1, "t_env": 20}
    original.save_models(tmp_path)
    expected_metrics = original.train(second, 3, 3, "b")

    restored = learner()
    restored.load_models(tmp_path)
    assert restored.progress == {"task_index": 1, "t_env": 20}
    assert restored.gamma_p == pytest.approx(0.8)
    assert not restored.mac.agent.task_embeddings["t0"].requires_grad
    assert restored.mac.agent.task_embeddings["t1"].requires_grad
    for saved, loaded in zip(before, restored.regularization_losses()):
        th.testing.assert_close(saved, loaded)
    actual_metrics = restored.train(second, 3, 3, "b")
    assert actual_metrics == pytest.approx(expected_metrics)
    for expected, actual in zip(original.params, restored.params):
        th.testing.assert_close(expected, actual)
