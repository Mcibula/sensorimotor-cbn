from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from modeling import MonteCarloCBN

if TYPE_CHECKING:
    from scipy.stats import rv_histogram


class CBNExplorationPolicy:
    def __init__(
            self,
            cbn: MonteCarloCBN,
            action_vars: list[str],
            state_vars: list[str],
            sampled_state_vars: list[str] | set[str],
            fixed_state_vars: list[str] | set[str],
            history_limit: int = 1000,
            expl_limit: int = 10,
            target_thresh: float = 1e-2,
            verbose: bool = False
    ) -> None:
        # There must be some action and state variables
        if not action_vars or not state_vars:
            raise ValueError

        # There must be some state variables marked for sampling/exploration
        if not sampled_state_vars:
            raise ValueError

        # All subtype state variables must be included in the state space
        if (
            not set(sampled_state_vars).issubset(set(state_vars))
            or not set(fixed_state_vars).issubset(set(state_vars))
        ):
            raise ValueError

        self.cbn: MonteCarloCBN = cbn

        # Action variables together with the state variables
        # must form the exact variables space of the CBN
        if not set(action_vars + state_vars) == set(self.cbn.nodes.keys()):
            raise ValueError

        self.act_history: list[dict[str, float]] = []
        self.obs_history: list[dict[str, float]] = []
        self.history_limit: int = history_limit

        self.action_vars: list[str] = action_vars
        self.state_vars: list[str] = state_vars
        self.sampled_vars: list[str] | set[str] = sampled_state_vars
        self.fixed_vars: list[str] | set[str] = fixed_state_vars

        self.expl_target: dict[str, float] = {}
        self.expl_steps: int = 0
        self.expl_limit: int = expl_limit
        self.target_thresh: float = target_thresh

        self.verbose: bool = verbose

    def __call__(self, state: dict[str, float]) -> np.ndarray:
        if set(state.keys()) != set(self.state_vars):
            raise ValueError

        self._observe(state)

        if self.expl_steps >= self.expl_limit or self._reached_target(state):
            if self.verbose:
                print('Finished the exploration episode')

            self.expl_target = {}

        if not self.exploring:
            if self.verbose:
                print('Starting new exploration episode...')

            self.expl_target = self.cbn.get_exploration_target(
                state=state,
                sampled_nodes=self.sampled_vars
            )
            self.expl_steps = 0

        sampling_targets = self.expl_target | {
            node_name: state[node_name]
            for node_name in self.fixed_vars
        }

        act_distribs = self.cbn.construct_hypothesis(
            sampling_targets=sampling_targets,
            target_nodes=self.sampled_vars,
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

    def _observe(self, state: dict[str, float]) -> None:
        if set(state.keys()) != set(self.state_vars):
            raise ValueError

        self.obs_history.append(state)

        if len(self.obs_history) - 1 < self.history_limit:
            return

        if self.verbose:
            print('Reached the history limit, refitting the CBN model...')

        obs_history = self.obs_history[1:]
        act_history = self.act_history

        if len(obs_history) != len(act_history):
            raise RuntimeError

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

    def _reached_target(self, state: dict[str, float]) -> bool:
        if not state:
            return False

        error = np.mean([
            np.abs(state[node_name] - node_target)
            for node_name, node_target in self.expl_target.items()
        ])

        return error <= self.target_thresh

    def _vectorize_action(self, action: dict[str, float]) -> np.ndarray:
        return np.array([
            action[var_name]
            for var_name in self.action_vars
        ])
