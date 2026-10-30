"""Policy identity, gradient and evaluation isolation checks at a task switch."""
import copy
import random
from types import SimpleNamespace as SN

import numpy as np
import pytest
import torch

from test_rpg_learner import learner, batch
from rpg_run import evaluate


def sequence(mac, data, task):
    mac.init_hidden(data.batch_size, task)
    return torch.stack([mac.forward(data, task, t) for t in range(data.max_seq_length)], 1)


def test_begin_task_and_old_evaluation_keep_old_policy_and_embedding_identity():
    torch.manual_seed(81)
    obj = learner()
    obj.begin_task('a')
    data = batch(obj.task2decomposer['a'])
    obj.train(data, 1, 1, 'a')
    obj.end_task('a', data)
    before = sequence(obj.mac, data, 'a').detach()
    reference_head = obj.mac.agent.generated_parameters('a').detach().clone()
    obj.begin_task('b')
    after = sequence(obj.mac, data, 'a').detach()
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    torch.testing.assert_close(reference_head, obj.hyper_targets['a'], rtol=0, atol=0)
    assert obj.mac.agent.active_task == 'b'
    assert not obj.mac.agent.task_embeddings['t0'].requires_grad
    assert obj.mac.agent.task_embeddings['t1'].requires_grad
    # An accidentally selected new embedding must produce detectably different
    # actions/values on this same old-task observation layout.
    obj.mac.init_hidden(data.batch_size, 'a')
    wrong = torch.stack([obj.mac.forward(data, 'a', t, embedding_task='b')
                         for t in range(data.max_seq_length)], 1)
    assert not torch.allclose(after, wrong)


def test_regularizers_restore_displaced_parameters_and_reach_hypernetwork():
    torch.manual_seed(82)
    obj = learner()
    obj.begin_task('a')
    obj.end_task('a', batch(obj.task2decomposer['a']))
    obj.begin_task('b')
    with torch.no_grad():
        for p in obj.mac.agent.relation_capturer.parameters():
            p.add_(.03)
        for p in obj.mac.agent.hypernetwork.parameters():
            p.add_(.03)
    rc, hy = obj.regularization_losses()
    named = list(obj.mac.agent.relation_capturer.named_parameters())
    gradients = torch.autograd.grad(rc, [p for _,p in named])
    for (name, parameter), grad in zip(named, gradients):
        expected = obj.args.rpg_lambda_rc * obj.relation_importance['a'][name] * (
            parameter - obj.relation_anchors['a'][name])
        torch.testing.assert_close(grad, expected)
    hypers = list(obj.mac.agent.hypernetwork.parameters())
    grads = torch.autograd.grad(hy, hypers)
    assert sum(g.square().sum() for g in grads).item() > 0
    # A small step against the hyper-regularizer must reduce generated-head drift.
    with torch.no_grad():
        for p,g in zip(hypers, grads):
            p.sub_(1e-4 * g)
    assert obj.regularization_losses()[1].item() < hy.item()
    assert obj.mac.agent.task_embeddings['t0'].grad is None


@pytest.mark.parametrize('fail', [False, True])
def test_evaluation_routes_old_task_and_restores_all_controller_state(fail):
    obj = learner()
    obj.begin_task('a')
    obj.end_task('a', batch(obj.task2decomposer['a']))
    obj.begin_task('b')
    mac = obj.mac
    mac.init_hidden(2, 'b')
    hidden = mac.hidden_states
    entropy = mac.last_entropy = torch.tensor([123.])
    mac.action_selector.epsilon = .37
    weights = copy.deepcopy(mac.agent.state_dict())
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    rows, called = [], []

    class Runner:
        def __init__(self, task):
            self.task = task

        def run(self, controller, test_mode, embedding_task):
            assert test_mode and not controller.agent.training
            assert embedding_task is None  # Each task uses its own embedding.
            called.append(self.task)
            controller.init_hidden(1, self.task)
            controller.last_entropy = torch.tensor([-1.])
            controller.action_selector.epsilon = 0.
            random.random(); np.random.rand(); torch.rand(2)
            if fail:
                raise RuntimeError('simulator failure')
            return None, {'return': 20., 'success': 1.}

    def call():
        return evaluate(mac, {t:Runner(t) for t in ('a','b')}, ['a','b'], 2,
                        SN(write=lambda filename,row: rows.append(row)), 100, 1)
    if fail:
        with pytest.raises(RuntimeError, match='simulator failure'):
            call()
    else:
        call()
        assert called == ['a','a','b','b']
        assert [(r['task'],r['embedding_task']) for r in rows] == [('a','a'),('b','b')]
    assert mac.hidden_states is hidden and mac.last_entropy is entropy
    assert mac.action_selector.epsilon == .37 and mac.agent.training
    assert mac.agent.active_task == 'b'
    assert random.getstate() == rng[0]
    actual_np = np.random.get_state()
    assert actual_np[0] == rng[1][0] and np.array_equal(actual_np[1],rng[1][1])
    assert actual_np[2:] == rng[1][2:]
    assert torch.equal(torch.get_rng_state(),rng[2])
    for k,v in weights.items():
        if isinstance(v,torch.Tensor):
            torch.testing.assert_close(v,mac.agent.state_dict()[k],rtol=0,atol=0)
        else:
            assert v == mac.agent.state_dict()[k]
