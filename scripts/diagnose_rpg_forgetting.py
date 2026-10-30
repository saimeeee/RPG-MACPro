"""Bounded, isolated task-transition diagnostic; never resume its snapshots.

Reuses the production driver/config/checkpoint. Stops after a small amount of
second-task training and evaluates component restorations on fresh, identically
seeded SMAC instances. Production output directories are never modified.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import rpg_run
from learners.rpg_learner import RPGQLearner


class DiagnosticComplete(Exception):
    pass


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


def rollout_q(mac, batch, task):
    with torch.no_grad():
        mac.init_hidden(batch.batch_size, task)
        return torch.stack([mac.forward(batch, task, t)
                            for t in range(batch.max_seq_length)], dim=1)


def norm(grads):
    return sum(float(g.detach().square().sum()) for g in grads if g is not None) ** .5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--episodes', type=int, default=32)
    parser.add_argument('--relation-scale', type=float, default=1.)
    parser.add_argument('--hyper-scale', type=float, default=1.)
    cli = parser.parse_args()
    if cli.steps < 1 or cli.episodes < 1 or any(not math.isfinite(v) or v <= 0
            for v in (cli.relation_scale, cli.hyper_scale)):
        parser.error('Steps, episodes and diagnostic coefficient multipliers must be positive')
    config = json.loads((cli.checkpoint / 'driver.json').read_text())['config']
    # These explicit overrides are diagnostic interventions, not paper settings
    # and not a continuation of the unchanged full experiment.
    config['rpg_lambda_rc'] *= cli.relation_scale
    config['rpg_lambda_hy'] *= cli.hyper_scale
    initial = torch.load(cli.checkpoint / 'learner.th', map_location='cpu', weights_only=True)
    old_task = initial['completed_tasks'][-1]
    start_step = initial['progress']['total_steps']
    output = cli.output
    report = {'checkpoint': str(cli.checkpoint), 'steps_requested': cli.steps,
              'evaluation_episodes': cli.episodes, 'config': config,
              'diagnostic_coefficient_multipliers': {'relation': cli.relation_scale, 'hyper': cli.hyper_scale},
              'old_task': old_task, 'restoration_evaluations': [], 'gradient_checks': []}

    class ProbeLearner(RPGQLearner):
        def save_report(self):
            (output / 'diagnosis.json').write_text(json.dumps(report, indent=2))

        def old_runner(self, task=None):
            task = task or old_task
            spec_index = [rpg_run.task_name(s) for s in config['tasks']].index(task)
            env, _ = rpg_run.build_env(config['tasks'][spec_index], config['env'],
                                      config['env_args'], config['seed'] + 100000 + 100 * spec_index)
            scheme, groups, preprocess = rpg_run.make_scheme(env.get_env_info())
            return rpg_run.RPGRunner(env, task, scheme, groups, preprocess, self.device)

        def test_old(self, label, keep_batch=False):
            runner = self.old_runner()
            try:
                results = []
                for i in range(cli.episodes):
                    batch, result = runner.run(self.mac, test_mode=True)
                    results.append(result)
                    if keep_batch and i == 0:
                        self.reference_batch = batch[:, :int(batch.max_t_filled().item())]
                        self.reference_q = rollout_q(self.mac, self.reference_batch, old_task).clone()
                row = {'variant': label, 'wins': sum(x['success'] for x in results),
                       'episodes': cli.episodes,
                       'win_rate': float(np.mean([x['success'] for x in results])),
                       'return_mean': float(np.mean([x['return'] for x in results]))}
                report['restoration_evaluations'].append(row)
                print(json.dumps(row), flush=True)
                self.save_report()
            finally:
                runner.env.close()

        def load_models(self, path):
            super().load_models(path)
            state = rng_state()
            try:
                self.old_agent = copy.deepcopy(self.mac.agent.state_dict())
                self.test_old('boundary_before_begin_task', keep_batch=True)
            finally:
                restore_rng(state)
                self.mac.hidden_states = self.mac.last_entropy = None

        def begin_task(self, task, previous_task=None):
            super().begin_task(task, previous_task)
            if task == old_task:
                return
            q = rollout_q(self.mac, self.reference_batch, old_task)
            names = ('relation_capturer.', 'hypernetwork.',
                     'task_embeddings.' + self.mac.agent.task_keys[old_task])
            current = self.mac.agent.state_dict()
            report['transition'] = {
                'old_policy_tensors_equal': all(torch.equal(v, current[k])
                    for k, v in self.old_agent.items()
                    if isinstance(v, torch.Tensor) and k.startswith(names)),
                'old_q_max_abs_difference': (q - self.reference_q).abs().max().item(),
                'embedding_requires_grad': {t: self.mac.agent.task_embeddings[k].requires_grad
                                           for t, k in self.mac.agent.task_keys.items()},
                'anchors_equal_parameters': all(torch.equal(v, dict(
                    self.mac.agent.relation_capturer.named_parameters())[k])
                    for k, v in self.relation_anchors[old_task].items()),
                'head_target_equal': torch.equal(self.hyper_targets[old_task],
                    self.mac.agent.generated_parameters(old_task)),
            }
            importance = torch.cat([v.flatten() for v in self.relation_importance[old_task].values()])
            report['importance'] = {
                'quantiles_0_50_90_99_100': torch.quantile(importance, importance.new_tensor([0,.5,.9,.99,1])).tolist(),
                'sum': importance.sum().item(),
                'by_parameter': {k: {'mean': v.mean().item(), 'max': v.max().item(), 'sum': v.sum().item()}
                                 for k,v in self.relation_importance[old_task].items()}}
            print('TRANSITION', json.dumps(report['transition']), flush=True)
            self.save_report()
            self.mac.hidden_states = self.mac.last_entropy = None
            self.next_gradient_step = start_step + 1000

        def train(self, batch, t_env, episode_num, task):
            if t_env >= self.next_gradient_step:
                td, _, _ = self._td_loss(batch, task)
                rc, hy = self.regularization_losses()
                params = list(self.mac.agent.relation_capturer.parameters())
                hypers = list(self.mac.agent.hypernetwork.parameters())
                td_rc = torch.autograd.grad(td, params, retain_graph=True, allow_unused=True)
                td_hy = torch.autograd.grad(td, hypers, allow_unused=True)
                reg_rc = torch.autograd.grad(self.gamma_p * rc, params, allow_unused=True)
                reg_hy = torch.autograd.grad(hy, hypers, allow_unused=True)
                row = {'step': t_env, 'td_relation_grad_norm': norm(td_rc),
                       'regularizer_relation_grad_norm': norm(reg_rc),
                       'td_hyper_grad_norm': norm(td_hy), 'regularizer_hyper_grad_norm': norm(reg_hy),
                       'gamma_p': self.gamma_p}
                report['gradient_checks'].append(row)
                self.save_report()
                self.next_gradient_step += 2000
            metrics = super().train(batch, t_env, episode_num, task)
            if t_env >= start_step + cli.steps:
                report['final_step'] = t_env
                report['last_update'] = metrics
                current = copy.deepcopy(self.mac.agent.state_dict())
                torch.save({'agent': current, 'mixer': self.mixer.state_dict(),
                            'relation_importance': self.relation_importance,
                            'relation_anchors': self.relation_anchors,
                            'hyper_targets': self.hyper_targets, 'step': t_env},
                           output / 'diagnostic_snapshot.th')
                report['old_embedding_max_abs_change'] = float((current[
                    'task_embeddings.' + self.mac.agent.task_keys[old_task]] - self.old_agent[
                    'task_embeddings.' + self.mac.agent.task_keys[old_task]]).abs().max())
                report['parameter_drift'] = {
                    k: {'rms_change': float((v-self.old_agent[k]).square().mean().sqrt()),
                        'max_change': float((v-self.old_agent[k]).abs().max())}
                    for k,v in current.items() if isinstance(v,torch.Tensor)
                    and k.startswith(('relation_capturer.', 'hypernetwork.'))}
                runner = self.old_runner(task)
                try:
                    results = [runner.run(self.mac, test_mode=True)[1] for _ in range(cli.episodes)]
                    report['new_task_evaluation'] = {
                        'task': task, 'episodes': cli.episodes,
                        'wins': sum(x['success'] for x in results),
                        'win_rate': float(np.mean([x['success'] for x in results]))}
                finally:
                    runner.env.close()
                for label, prefixes in [('current', ()), ('restore_relation', ('relation_capturer.',)),
                                        ('restore_hyper', ('hypernetwork.',)),
                                        ('restore_both', ('relation_capturer.', 'hypernetwork.'))]:
                    state = copy.deepcopy(current)
                    for k in state:
                        if k.startswith(prefixes):
                            state[k] = self.old_agent[k]
                    self.mac.agent.load_state_dict(state)
                    q = rollout_q(self.mac, self.reference_batch, old_task)
                    available = self.reference_batch['avail_actions'].bool()
                    old_actions = self.reference_q.masked_fill(~available, -torch.inf).argmax(-1)
                    actions = q.masked_fill(~available, -torch.inf).argmax(-1)
                    valid = self.reference_batch['filled'].bool().expand_as(actions) & available.any(-1)
                    report.setdefault('fixed_trajectory', {})[label] = {
                        'q_max_abs_difference': float((q-self.reference_q).abs().max()),
                        'greedy_action_disagreement': float((old_actions[valid] != actions[valid]).float().mean())}
                    self.test_old(label)
                self.mac.agent.load_state_dict(current)
                self.save_report()
                raise DiagnosticComplete()
            return metrics

    rpg_run.LEARNER_REGISTRY[config['learner']] = ProbeLearner
    try:
        rpg_run.run_experiment(config, SimpleNamespace(checkpoint=str(cli.checkpoint),
                               output=str(output), zero_shot_embedding=None, evaluate=False))
    except DiagnosticComplete:
        print('DIAGNOSTIC_COMPLETE', output / 'diagnosis.json', flush=True)


if __name__ == '__main__':
    main()
