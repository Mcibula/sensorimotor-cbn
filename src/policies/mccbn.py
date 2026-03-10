import babybench.utils as bb_utils
import numpy as np
import pandas as pd
import yaml

from modeling import CBNNode, MonteCarloCBN
from modeling.models import SumModel


class CBNExplorationPolicy:
    def __init__(
            self,
            cbn: MonteCarloCBN,
            action_vars: list[str],
            state_vars: list[str],
            sampled_state_vars: list[str],
            fixed_state_vars: list[str],
            history_limit: int = 1000,
            expl_limit: int = 10,
            target_thresh: float = 1e-2,
            verbose: bool = False
    ) -> None:
        assert action_vars
        assert state_vars
        assert sampled_state_vars

        self.cbn: MonteCarloCBN = cbn

        self.act_history: list[dict[str, float]] = []
        self.obs_history: list[dict[str, float]] = []
        self.history_limit: int = history_limit

        self.action_vars: list[str] = action_vars
        self.state_vars: list[str] = state_vars
        self.sampled_vars: list[str] = sampled_state_vars
        self.fixed_vars: list[str] = fixed_state_vars

        self.expl_target: dict[str, float] = {}
        self.expl_steps: int = 0
        self.expl_limit: int = expl_limit
        self.target_thresh: float = target_thresh

        self.verbose: bool = verbose

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = self._parse_state(obs)
        self._observe(obs)

        if self.expl_steps >= self.expl_limit or self._reached_target(obs):
            if self.verbose:
                print('Finished the exploration episode')

            self.expl_target = {}

        if not self.exploring:
            if self.verbose:
                print('Starting new exploration episode...')

            self.expl_target = self.cbn.get_exploration_target(
                state=obs,
                sampled_nodes=self.sampled_vars
            )
            self.expl_steps = 0

        sampling_targets = self.expl_target | {
            node_name: obs[node_name]
            for node_name in self.fixed_vars
        }

        act_distribs = self.cbn.construct_hypothesis(
            sampling_targets=sampling_targets,
            fixed_nodes=self.fixed_vars,
            sampled_nodes=self.sampled_vars,
            readout_nodes=self.action_vars,
            delta_gain=1.25,
            min_results=50,
            verbose=self.verbose
        )

        self.expl_steps += 1

        action = {
            node_name: distrib.rvs(size=1)
            for node_name, distrib in act_distribs.items()
        }
        self.act_history.append(action)
        action = self._vectorize_action(action)

        return action

    @property
    def exploring(self) -> bool:
        return len(self.expl_target) > 0

    def _observe(self, obs: dict[str, float]) -> None:
        self.obs_history.append(obs)

        if len(self.obs_history) - 1 < self.history_limit:
            return

        if self.verbose:
            print('Reached the history limit, refitting the CBN model...')

        obs_history = self.obs_history[1:]
        act_history = self.act_history

        assert len(obs_history) == len(act_history)

        self.obs_history = []
        self.act_history = []

        data = {
            var_name: []
            for var_name in self.action_vars + self.state_vars
        }

        for t in range(len(obs_history)):
            for var_name, var_value in (act_history[t] | obs_history[t]).items():
                data[var_name].append(var_value)

        self.cbn.fit(pd.DataFrame(data))

    def _reached_target(self, obs: dict[str, float]) -> bool:
        error = np.mean([
            np.abs(obs[node_name] - node_target)
            for node_name, node_target in self.expl_target.items()
        ])

        return error <= self.target_thresh

    def _parse_state(self, obs: np.ndarray) -> dict[str, float]:
        return {
            var_name: float(obs[idx])
            for idx, var_name in enumerate(self.state_vars)
        }

    def _vectorize_action(self, action: dict[str, float]) -> np.ndarray:
        return np.array([
            action[var_name]
            for var_name in self.action_vars
        ])


if __name__ == '__main__':
    cbn = MonteCarloCBN(load_file=...)

    with open('./config_selftouch.yml', 'r') as f:
        config = yaml.safe_load(f)

    env = bb_utils.make_env(config)

    policy = CBNExplorationPolicy(
        cbn=cbn,
        action_vars=[
            act_name
            for act_idx in range(env.model.nu)
            if (act_name := env.model.actuator(act_idx).name).startswith('act:')
        ],
        state_vars=[
            ...
        ]
    )
