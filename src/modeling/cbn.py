from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from dowhy import gcm
from scipy import stats

if TYPE_CHECKING:
    from dowhy.gcm.causal_mechanisms import ConditionalStochasticModel, StochasticModel
    from dowhy.gcm.causal_models import StructuralCausalModel
    from networkx import DiGraph


class CBNNode:
    def __init__(
            self,
            name: str,
            causal_mechanism: StochasticModel | ConditionalStochasticModel,
            limits: str | tuple[float, float] = 'infer'
    ) -> None:
        self.name: str = name
        self.causal_mechanism: StochasticModel | ConditionalStochasticModel = causal_mechanism
        self.parents: list['CBNNode'] = []
        self.children: list['CBNNode'] = []

        self.observed_y: np.ndarray | None = None
        self.curiosity: stats.rv_histogram | None = None

        self.infer_limits: bool = limits == 'infer'
        self.limits: tuple[float, float] = (
            (0.0, 0.0)
            if self.infer_limits
            else limits
        )

        self.gen_samples: np.ndarray | None = None

    @property
    def is_root(self) -> bool:
        return not self.parents

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def has_been_fit(self) -> bool:
        return self.observed_y is not None

    @property
    def x_dim(self) -> int:
        return len(self.parents)

    @property
    def domain(self) -> np.ndarray:
        return np.array([
            np.linspace(*p.limits, num=10_000)
            for p in self.parents
        ]).T

    @property
    def codomain(self) -> np.ndarray:
        return np.linspace(*self.limits, num=10_000)

    def _compute_curiosity(self) -> stats.rv_histogram:
        counts, bin_edges = np.histogram(
            self.observed_y,
            bins='auto',
            range=self.limits
        )
        compl_hist = (
            counts.max() - counts,
            bin_edges
        )

        return stats.rv_histogram(compl_hist)

    def bump_curiosity(self, point: float, strength: float = 1.5, width: float = 0.1) -> stats.rv_histogram:
        if self.curiosity is None:
            self.curiosity = self._compute_curiosity()

        if point < self.limits[0] or point > self.limits[1]:
            raise ValueError

        counts, bin_edges = self.curiosity._histogram
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        bump = 1 + (strength - 1) * np.exp(-0.5 * ((bin_centers - point) / width) ** 2)
        boosted_counts = counts * bump

        bin_widths = np.diff(bin_edges)
        area = np.sum(boosted_counts * bin_widths)
        boosted_counts /= area

        return stats.rv_histogram((boosted_counts, bin_edges))

    def observe(self, data: np.ndarray) -> None:
        self.observed_y = (
            data
            if self.observed_y is None
            else np.append(self.observed_y, data)
        )

        if self.infer_limits:
            self.limits = (self.observed_y.min(), self.observed_y.max())

        self.curiosity = self._compute_curiosity()

    def set_parents(self, parents: list['CBNNode']) -> None:
        self.parents = parents

    def set_children(self, children: list['CBNNode']) -> None:
        self.children = children

    def plot_obs_distribution(self) -> None:
        fig, ax1 = plt.subplots()

        c1 = 'tab:blue'
        c2 = 'tab:orange'

        ax1.hist(
            self.observed_y,
            bins='auto',
            range=self.limits,
            color=c1,
        )
        ax1.set_xlabel(f'Value of {self.name}')
        ax1.set_ylabel('Number of observations', color=c1)
        ax1.tick_params(axis='y', labelcolor=c1)

        ax2 = ax1.twinx()

        X = np.linspace(*self.limits, int(1e4))
        ax2.plot(
            X,
            self.curiosity.pdf(X),
            color=c2,
            linewidth=1
        )
        ax2.set_ylabel('Sampling probability density', color=c2)
        ax2.tick_params(axis='y', labelcolor=c2)

        fig.tight_layout()
        plt.show()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.causal_mechanism.draw_samples(X)

    def plot_image(self) -> None:
        if self.x_dim == 0:
            raise ValueError

        X = self.domain
        y = self.predict(X)
        points = np.hstack((X, y))

        dim_reduce = points.shape[1] > 2

        if dim_reduce:
            reducer = umap.UMAP(
                n_neighbors=100,
                n_components=2,
                metric='euclidean',
                min_dist=0.1,
                verbose=False
            )
            points = reducer.fit_transform(points)

        fig, ax = plt.subplots()
        ax.scatter(points[:, 0], points[:, 1], s=1)

        if dim_reduce:
            ax.set_xlabel('Embedded X')
            ax.set_ylabel('Embedded Y')
        else:
            ax.set_xlabel(f'Ground-truth {self.parents[0].name}')
            ax.set_ylabel(f'Predicted {self.name}')

        fig.tight_layout()
        plt.show()

    def set_samples(self, samples: np.ndarray) -> None:
        if self.gen_samples is not None and self.gen_samples.shape != samples.shape:
            raise ValueError

        self.gen_samples = samples

    def propagate_samples(self) -> None:
        pa_samples = np.vstack([
            parent.gen_samples
            for parent in self.parents
        ]).T
        samples = self.predict(pa_samples)[:, 0]

        self.set_samples(samples)

        for child in self.children:
            child.propagate_samples()

    def explore(self, current_state: float | None = None, n_samples: int = 1) -> np.ndarray:
        curiosity = (
            self.curiosity
            if current_state is None
            else self.bump_curiosity(current_state)
        )

        return curiosity.rvs(size=n_samples)


class MonteCarloCBN:
    def __init__(
            self,
            untrained_scm: StructuralCausalModel | None = None,
            limits: dict[str, tuple[float, float]] | None = None,
            load_file: str | None = None
    ) -> None:
        if untrained_scm is None and load_file is None:
            raise ValueError

        if load_file is not None:
            with open(load_file, 'rb') as f:
                data = pickle.load(f)

            self.M: StructuralCausalModel = data['scm']
            self.nodes: dict[str, CBNNode] = data['nodes']

            return

        self.M: StructuralCausalModel = untrained_scm
        self.nodes: dict[str, CBNNode] = {}

        if limits is None:
            limits = {}

        try:
            for node in list(self.G.nodes):
                self.nodes[node] = CBNNode(
                    name=node,
                    causal_mechanism=self.M.causal_mechanism(node),
                    limits=(
                        'infer'
                        if node not in limits
                        else limits[node]
                    )
                )
        except KeyError:
            raise ValueError

        for node_name in self.nodes:
            node = self.nodes[node_name]

            node.set_parents([
                self.nodes[parent]
                for parent in self.G.predecessors(node_name)
            ])
            node.set_children([
                self.nodes[child]
                for child in self.G.successors(node_name)
            ])

    @property
    def G(self) -> DiGraph:
        return self.M.graph

    def _check_same_len(self, nodes: list[str]) -> bool:
        data_len: list[int] = []
        for node in nodes:
            node_data = self.nodes[node].observed_y
            data_len.append(
                node_data.shape[0]
                if node_data is not None
                else 0
            )

        return len(set(data_len)) == 1

    def _reconstruct_data(self, nodes: list[str]) -> pd.DataFrame:
        return pd.DataFrame({
            node: (
                data
                if (data := self.nodes[node].observed_y) is not None
                else []
            )
            for node in nodes
        })

    def fit(self, data: pd.DataFrame) -> None:
        if not self._check_same_len(list(self.nodes)):
            raise ValueError

        new_data = pd.concat([
            data,
            self._reconstruct_data(list(self.nodes))
        ])

        gcm.fit(self.M, new_data)
        samples = self.forward_sample()

        for node_name in self.nodes:
            node = self.nodes[node_name]

            node.observe(data.loc[:, node_name].to_numpy())
            node.set_samples(samples.loc[:, node_name].to_numpy())

    def fit_node(self, node: str, data: pd.DataFrame) -> None:
        rel_nodes = [
            parent.name
            for parent in self.nodes[node].parents
        ] + [node]
        data = data.loc[:, rel_nodes]

        if not self._check_same_len(rel_nodes):
            raise ValueError

        new_data = pd.concat([
            data,
            self._reconstruct_data(rel_nodes)
        ])

        gcm.fitting_sampling.fit_causal_model_of_target(self.M, node, new_data)

        for node_name in rel_nodes:
            self.nodes[node_name].observe(data.loc[:, node_name].to_numpy())

        self.nodes[node].propagate_samples()

    def forward_sample(self, n_samples: int = int(1e7)) -> pd.DataFrame:
        return gcm.draw_samples(
            causal_model=self.M,
            num_samples=n_samples
        )

    def predict(
            self,
            interventions: dict[str, float],
            aggregator: Callable[[np.ndarray, ...], np.ndarray] | None = None,
            n_samples: int = int(1e5)
    ) -> np.ndarray:
        samples = gcm.interventional_samples(
            causal_model=self.M,
            interventions={
                node_name: lambda x: node_value
                for node_name, node_value in interventions.items()
            },
            num_samples_to_draw=n_samples
        )

        if aggregator is not None:
            samples = aggregator(samples)

        return samples

    def get_exploration_target(
            self,
            state: dict[str, float],
            sampled_nodes: list[str]
    ) -> dict[str, float]:
        sampling_targets = {
            node_name: self.nodes[node_name].explore(
                current_state=state[node_name],
                n_samples=1
            )[0]
            for node_name in sampled_nodes
        }

        return sampling_targets

    def construct_hypothesis(
            self,
            sampling_targets: dict[str, float],
            fixed_nodes: list[str],
            sampled_nodes: list[str],
            readout_nodes: list[str],
            init_delta: float = 0.1,
            delta_gain: float = 2.0,
            min_results: int = 100,
            verbose: bool = False
    ) -> dict[str, stats.rv_histogram]:
        if verbose:
            print(f'Starting backpropagated rejection sampling with delta = {init_delta:.2f}')

        delta = init_delta
        iteration = 0

        while True:
            if verbose:
                print(f'Iteration {iteration + 1} (delta = {delta:.2f}): ', end='')

            init_node = sampled_nodes[0]
            mask = np.abs(self.nodes[init_node].gen_samples - sampling_targets[init_node]) <= delta
            sample_ids = np.where(mask)[0]

            for node_name in sampled_nodes[1:] + fixed_nodes:
                if len(sample_ids) < min_results:
                    break

                rel_samples = self.nodes[node_name].gen_samples[sample_ids]

                mask = np.abs(rel_samples - sampling_targets[node_name]) <= delta
                sample_ids = sample_ids[mask]

            if len(sample_ids) >= min_results:
                if verbose:
                    print(f'Success. Found {len(sample_ids)} valid samples.')

                break

            if verbose:
                print(f'Failed. Found {len(sample_ids)} valid samples, repeating.')

            delta *= delta_gain
            iteration += 1

        readout_distribs = {}
        for node_name in readout_nodes:
            node = self.nodes[node_name]
            samples = node.gen_samples[sample_ids]

            hist = np.histogram(
                samples,
                bins='auto',
                range=node.limits
            )
            readout_distribs[node_name] = stats.rv_histogram(hist)

        return readout_distribs

    def explore(
            self,
            state: dict[str, float],
            fixed_nodes: list[str],
            sampled_nodes: list[str],
            readout_nodes: list[str],
            init_delta: float = 0.1,
            delta_gain: float = 2.0,
            min_results: int = 100,
            verbose: bool = False
    ) -> dict[str, stats.rv_histogram]:
        sampling_targets = self.get_exploration_target(state, sampled_nodes) | {
            node_name: state[node_name]
            for node_name in fixed_nodes
        }

        readout_distribs = self.construct_hypothesis(
            sampling_targets,
            fixed_nodes,
            sampled_nodes,
            readout_nodes,
            init_delta,
            delta_gain,
            min_results,
            verbose
        )

        return readout_distribs

    def save(self, filename: str) -> None:
        data = {
            'scm': self.M,
            'nodes': self.nodes
        }

        with open(filename, 'wb') as f:
            pickle.dump(data, f)
