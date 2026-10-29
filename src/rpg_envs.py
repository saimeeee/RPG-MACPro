"""Environment adapters for RPG without importing MACPro's eager env registry.

LBF uses the unmodified ``lbforaging==2.0.0`` package, not the directional
five-action environment bundled with MACPro. Local observations retain LBF's
partial observability. Coordinates are divided by (height - 1, width - 1),
and all entity levels by players * max_player_level. Missing entities are zero.
The centralized mixer alone receives the global player/food state.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


class LBFDecomposer:
    """MACPro decomposition interface for canonical entity-wise LBF features."""

    def __init__(self, n_agents, max_food, episode_limit=50):
        self.n_agents = int(n_agents)
        self.n_enemies = int(max_food)  # Legacy name: these are food entities.
        self.n_allies = self.n_agents - 1
        self.episode_limit = int(episode_limit)
        self.n_actions = self.n_actions_no_attack = 6
        self.n_interaction_actions = 0
        self.n_actions_move = 4
        self.own_obs_dim = self.obs_nf_al = self.obs_nf_en = 3
        self.state_nf_al = self.state_nf_en = 3
        self.obs_dim = 3 * (self.n_agents + self.n_enemies)
        self.state_dim = self.obs_dim
        self.state_last_action = self.state_timestep_number = False
        self.last_action_state_dim = self.timestep_number_state_dim = 0
        self.ally_state_dim = 3 * self.n_agents
        self.enemy_state_dim = 3 * self.n_enemies

    def decompose_obs(self, obs_input):
        own = obs_input[..., :3]
        allies = [obs_input[..., 3 * i:3 * (i + 1)]
                  for i in range(1, self.n_agents)]
        base = 3 * self.n_agents
        foods = [obs_input[..., base + 3 * i:base + 3 * (i + 1)]
                 for i in range(self.n_enemies)]
        return own, foods, allies

    def decompose_state(self, state_input):
        allies = [state_input[..., 3 * i:3 * (i + 1)]
                  for i in range(self.n_agents)]
        base = 3 * self.n_agents
        foods = [state_input[..., base + 3 * i:base + 3 * (i + 1)]
                 for i in range(self.n_enemies)]
        return allies, foods, [], state_input[..., :0]

    def decompose_obs_mask(self, obs_input):
        entities = obs_input[..., :self.obs_dim].reshape(
            *obs_input.shape[:-1], self.n_agents + self.n_enemies, 3)
        return (entities[..., 1:self.n_agents, 2] > 0,
                entities[..., self.n_agents:, 2] > 0)

    def decompose_state_mask(self, state_input):
        entities = state_input[..., :self.state_dim].reshape(
            *state_input.shape[:-1], self.n_agents + self.n_enemies, 3)
        return (entities[..., :self.n_agents, 2] > 0,
                entities[..., self.n_agents:, 2] > 0)

    def decompose_action_info(self, action_info):
        import torch

        if action_info.shape[-1] != 6:
            raise ValueError("LBF expects six actions: NONE, N, S, W, E, LOAD")
        # LOAD is an ego action. There is no one-action-per-food interface.
        foods = action_info.new_zeros(*action_info.shape[:-1], self.n_enemies)
        interaction = action_info.new_zeros(*action_info.shape[:-1], 1)
        return action_info, foods, torch.cat([action_info, interaction], dim=-1)


class LBFAdapter:
    """Expose official LBF through the episode runner's MultiAgentEnv contract."""

    def __init__(self, task_spec, seed=0):
        if not isinstance(task_spec, dict):
            raise ValueError("LBF tasks must be explicit dictionaries, not Gym IDs")
        try:
            from lbforaging.foraging import ForagingEnv
        except ImportError as exc:
            raise ImportError(
                "RPG LBF requires the official package: pip install lbforaging==2.0.0"
            ) from exc

        self.task_spec = dict(task_spec)
        self.n_agents = int(task_spec["players"])
        self.max_food = int(task_spec["max_food"])
        self.field_size = tuple(int(x) for x in task_spec.get("field_size", [8, 8]))
        self.max_player_level = int(task_spec.get("max_player_level", 3))
        self.sight = int(task_spec.get("sight", max(self.field_size)))
        self.episode_limit = int(task_spec.get("max_episode_steps", 50))
        if (self.n_agents < 1 or self.max_food < 1 or len(self.field_size) != 2
                or min(self.field_size) < 3 or self.max_player_level < 1
                or self.episode_limit < 1 or self.sight < 0):
            raise ValueError("Invalid LBF task sizes, levels, horizon, or sight")
        self._seed = int(seed)
        self._reset_count = 0
        self._steps = 0
        self._obs = None
        self.decomposer = LBFDecomposer(self.n_agents, self.max_food, self.episode_limit)
        self._scale = np.asarray([
            self.field_size[0] - 1, self.field_size[1] - 1,
            self.n_agents * self.max_player_level,
        ], dtype=np.float32)

        # PyPI 2.0.0 uses max_num_food and separate min/max levels. Accept the
        # older spelling too; only adapt argument spelling, never dynamics.
        parameters = inspect.signature(ForagingEnv).parameters
        kwargs = {
            "players": self.n_agents,
            "max_player_level": self.max_player_level,
            "field_size": self.field_size,
            "sight": self.sight,
            "max_episode_steps": self.episode_limit,
            "force_coop": bool(task_spec.get("force_coop", False)),
            "normalize_reward": True,
            "grid_observation": False,
            "observe_agent_levels": True,
            "penalty": 0.0,
        }
        food_arg = "max_num_food" if "max_num_food" in parameters else "max_food"
        kwargs[food_arg] = self.max_food
        for name, value in (("min_player_level", 1), ("min_food_level", 1),
                            ("max_food_level", None)):
            if name in parameters:
                kwargs[name] = value
        self._env = ForagingEnv(**{k: v for k, v in kwargs.items() if k in parameters})
        if any(space.n != 6 for space in self._env.action_space.spaces):
            raise RuntimeError("Unexpected LBF action space; install official lbforaging==2.0.0")

    def _canonical_observations(self, raw_observations):
        result = []
        count = self.n_agents + self.max_food
        for raw in raw_observations:
            raw = np.asarray(raw, dtype=np.float32)
            if raw.size != count * 3:
                raise RuntimeError(
                    f"Unexpected LBF flat observation size {raw.size}; expected {count * 3}. "
                    "The altered MACPro LBF package is not supported."
                )
            # Official LBF orders foods first, then visible players with self first.
            entities = raw.reshape(count, 3)
            entities = np.concatenate([entities[self.max_food:], entities[:self.max_food]])
            visible = entities[:, 2] > 0
            entities = entities / self._scale
            entities[~visible] = 0
            result.append(entities.reshape(-1))
        return np.stack(result).astype(np.float32, copy=False)

    def reset(self):
        # Reproducible sequence of different episodes; do not repeat the same map.
        episode_seed = self._seed + self._reset_count
        self._reset_count += 1
        if "seed" in inspect.signature(self._env.reset).parameters:
            result = self._env.reset(seed=episode_seed)
        else:
            self._env.seed(episode_seed)
            result = self._env.reset()
        raw = result[0] if (isinstance(result, tuple) and len(result) == 2
                            and isinstance(result[1], dict)) else result
        self._obs = self._canonical_observations(raw)
        self._steps = 0
        return self.get_obs(), self.get_state()

    def step(self, actions):
        if self._obs is None:
            raise RuntimeError("Call reset() before step()")
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        if actions.size != self.n_agents or np.any((actions < 0) | (actions >= 6)):
            raise ValueError("Supply one integer action in [0, 5] per LBF player")
        result = self._env.step(actions.tolist())
        if len(result) == 5:
            raw, rewards, terminated, truncated, info = result
        elif len(result) == 4:
            raw, rewards, terminated, info = result
            truncated = False
        else:
            raise RuntimeError("Unsupported LBF step return signature")
        self._steps += 1
        self._obs = self._canonical_observations(raw)
        completed = not np.any(self._env.field)
        timeout = self._steps >= self.episode_limit and not completed
        # LBF v2 reports its own horizon as termination. Correct it for TD masking.
        done = bool(np.all(terminated) or np.all(truncated) or completed or timeout)
        info = dict(info or {})
        info["episode_limit"] = bool((timeout or np.all(truncated)) and not completed)
        info["battle_won"] = bool(completed)
        return float(np.sum(rewards)), done, info

    def get_obs(self):
        if self._obs is None:
            raise RuntimeError("Call reset() before requesting observations")
        return self._obs.copy()

    def get_state(self):
        if self._obs is None:
            raise RuntimeError("Call reset() before requesting state")
        players = np.asarray([
            [p.position[0], p.position[1], p.level] for p in self._env.players
        ], dtype=np.float32)
        foods = np.zeros((self.max_food, 3), dtype=np.float32)
        rows, cols = np.nonzero(self._env.field)
        if len(rows) > self.max_food:
            raise RuntimeError("LBF contains more food entities than the configured capacity")
        foods[:len(rows)] = np.asarray([
            [r, c, self._env.field[r, c]] for r, c in zip(rows, cols)
        ], dtype=np.float32).reshape(-1, 3)
        return (np.concatenate([players, foods]) / self._scale).reshape(-1)

    def get_avail_actions(self):
        # Invalid LBF actions become NONE in the official environment. Keeping the
        # standard six-action space also avoids leaking off-view cells through masks.
        return np.ones((self.n_agents, 6), dtype=np.int32)

    def get_env_info(self):
        return {
            "state_shape": self.decomposer.state_dim,
            "obs_shape": self.decomposer.obs_dim,
            "n_actions": 6,
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }

    def close(self):
        self._env.close()

    def get_checkpoint_state(self):
        """Episode-boundary RNG position; the next reset seeds a fresh episode."""
        return {"seed": self._seed, "reset_count": self._reset_count}

    def load_checkpoint_state(self, state):
        if state["seed"] != self._seed:
            raise ValueError("Environment seed differs from checkpoint")
        self._reset_count = int(state["reset_count"])


def _require_starcraft_installation():
    configured = os.environ.get("SC2PATH")
    candidates = [Path(configured)] if configured else []
    if not configured:
        if sys.platform == "win32":
            for variable in ("ProgramFiles(x86)", "ProgramFiles"):
                if os.environ.get(variable):
                    candidates.append(Path(os.environ[variable]) / "StarCraft II")
        else:
            candidates.extend([Path.cwd() / "3rdparty" / "StarCraftII",
                               Path.home() / "StarCraftII"])
    for candidate in candidates:
        binaries = list(candidate.glob("Versions/Base*/SC2_x64*"))
        binaries += list(candidate.glob("Versions/Base*/SC2.app/Contents/MacOS/SC2"))
        if any(path.is_file() for path in binaries):
            os.environ.setdefault("SC2PATH", str(candidate))
            return candidate
    raise FileNotFoundError(
        "StarCraft II executable is missing. Install StarCraft II and the SMAC maps, "
        "then set SC2PATH to its installation directory (containing Versions/Base*/SC2_x64). "
        "The RPG LBF benchmark can run without StarCraft II."
    )


def build_env(task_spec, env_name, env_args=None, seed=0):
    """Return ``(MultiAgentEnv-compatible environment, MACPro decomposer)``."""
    if env_name == "lbf":
        spec = dict(env_args or {})
        if not isinstance(task_spec, dict):
            raise ValueError("LBF requires an explicit task dictionary")
        spec.update(task_spec)
        env = LBFAdapter(spec, seed=seed)
        return env, env.decomposer
    if env_name in ("sc2", "smac"):
        _require_starcraft_installation()
        smac_root = str(Path(__file__).resolve().parent / "envs" / "smac")
        if smac_root not in sys.path:
            sys.path.insert(0, smac_root)
        try:
            from smac.env import StarCraft2Env
            from modules.decomposers.sc2_decomposer import SC2Decomposer
        except ImportError as exc:
            raise ImportError(
                "SMAC Python dependencies are missing; install requirements-rpg-smac.txt "
                "and the bundled src/envs/smac package."
            ) from exc
        options = dict(env_args or {})
        if isinstance(task_spec, str):
            options["map_name"] = task_spec
        elif isinstance(task_spec, dict):
            options.update({k: v for k, v in task_spec.items() if k != "name"})
            options.setdefault("map_name", task_spec.get("name"))
        else:
            raise ValueError("SMAC task must be a map name or task dictionary")
        options["seed"] = int(seed)
        env = StarCraft2Env(**options)
        # Read effective defaults rather than duplicating SMAC configuration.
        decomp_options = dict(options)
        for key in ("obs_own_health", "obs_all_health", "obs_instead_of_state",
                    "obs_last_action", "obs_pathing_grid", "obs_terrain_height",
                    "obs_timestep_number", "state_last_action", "state_timestep_number"):
            decomp_options[key] = getattr(env, key)
        decomposer = SC2Decomposer(SimpleNamespace(env_args=decomp_options))
        decomposer.n_interaction_actions = decomposer.n_enemies
        return env, decomposer
    raise ValueError(f"Unsupported RPG environment: {env_name!r}")
