from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from dowhy import gcm
from dowhy.gcm.causal_mechanisms import StochasticModel
from scipy import stats

if TYPE_CHECKING:
    from dowhy.gcm.causal_mechanisms import ConditionalStochasticModel
    from dowhy.gcm.causal_models import StructuralCausalModel
    from networkx import DiGraph


class CBNNode:
    """
    Causal Bayesian network node supporting
    probabilistic learning and inference operations
    """

    def __init__(
            self,
            name: str,
            causal_mechanism: StochasticModel | ConditionalStochasticModel,
            limits: Literal['infer'] | tuple[float, float] = 'infer'
    ) -> None:
        """
        Initialize a node instance

        :param name: Natural-language name of the node
        :param causal_mechanism: Causal learning/inference model to be used in the node
        :param limits: Min-max range of the node's output (i.e., co-domain / y-values / targets);
                       if set to `'infer'`, the range will be inferred from observed data
        """

        self.name: str = name
        self.causal_mechanism: StochasticModel | ConditionalStochasticModel = causal_mechanism
        self.parents: list[CBNNode] = []
        self.children: list[CBNNode] = []

        # Observation accumulator
        self.observed_y: np.ndarray | None = None
        self.curiosity: stats.rv_histogram | None = None

        self.infer_limits: bool = limits == 'infer'
        self.limits: tuple[float, float] = (
            (0.0, 0.0)
            if self.infer_limits
            else limits
        )

        # Generative distribution
        self.gen_samples: np.ndarray | None = None

    @property
    def is_root(self) -> bool:
        """
        Check if the node is a root (orphaned)
        """

        return not self.parents

    @property
    def is_leaf(self) -> bool:
        """
        Check if the node is a leaf (childless)
        """

        return not self.children

    @property
    def has_been_fit(self) -> bool:
        """
        Check if the node has already observed any data (has been fit)
        """

        return self.observed_y is not None

    @property
    def x_dim(self) -> int:
        """
        Input dimensionality (number of input nodes / features)
        """

        return len(self.parents)

    @property
    def domain(self) -> np.ndarray:
        """
        Numerical domain of the node
        """

        return np.array([
            np.linspace(*p.limits, num=10_000)
            for p in self.parents
        ]).T

    @property
    def codomain(self) -> np.ndarray:
        """
        Numerical co-domain of the node
        """

        return np.linspace(*self.limits, num=10_000)

    def _compute_curiosity(self) -> stats.rv_histogram:
        r"""
        Compute the curiosity distribution based on epistemic uncertainty.
        I.e., the observation density :math:`\hat{p}_\text{obs}` is inverted as
        .. math::
            p_\text{inv} (x) \triangleq \frac{p_{\max} - \hat{p}_\text{obs}(x)}{p_{\max} V - 1},
        where :math:`p_{\max} = \sup_{x \in \mathcal{X}} \hat{p}_\text{obs}(x)` is the maximum density
        of the observation distribution, and :math:`V = \sum_{m=1}^M \lvert B_m \rvert` denotes the total
        continuous width of the support :math:`\mathcal{X}`, with :math:`\lvert B_m \rvert` being
        the width of a histogram bin :math:`B_m`.

        :return: The inverted histogram distribution :math:`P_\text{inv}`
        """

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
        r"""
        Generate curiosity / intrinsic motivation distribution by inverting
        the observation distribution and boosting the local neighborhood around `point`
        using the Gaussian bump function to avoid large discontinuities in the exploration.

        :param point: Point to boost the neighborhood around. This is supposed to be the current
                      point observation, so the intrinsic motivation primarily forces the agent
                      to explore near the current state
        :param strength: Amplification factor of the locality bias
        :param width: Standard deviation of the bump
        :return: Curiosity / intrinsic motivation distribution modulated by the current state
        """

        # If no curiosity distribution is cached yet, compute a new one
        if self.curiosity is None:
            self.curiosity = self._compute_curiosity()

        # Check whether the observed point is within the range
        if point < self.limits[0] or point > self.limits[1]:
            raise ValueError

        counts, bin_edges = self.curiosity._histogram
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Boost the distribution around the point
        bump = 1 + (strength - 1) * np.exp(-0.5 * ((bin_centers - point) / width) ** 2)
        boosted_counts = counts * bump

        # Normalize
        bin_widths = np.diff(bin_edges)
        area = np.sum(boosted_counts * bin_widths)
        boosted_counts /= area

        return stats.rv_histogram((boosted_counts, bin_edges))

    def observe(self, data: np.ndarray) -> None:
        """
        Observe and record new outcomes of this node

        :param data: Vector batch of the observed scalar outcomes
        """

        self.observed_y = (
            data
            if self.observed_y is None
            else np.append(self.observed_y, data)
        )

        # Refresh the data limits with respect to new data
        if self.infer_limits:
            self.limits = (self.observed_y.min(), self.observed_y.max())

        # Recompute curiosity distribution
        self.curiosity = self._compute_curiosity()

    def set_parents(self, parents: list[CBNNode]) -> None:
        """
        Set the parent nodes of this node

        :param parents: Parent `CBNNode` instances
        """

        self.parents = parents

    def set_children(self, children: list[CBNNode]) -> None:
        """
        Set the children nodes of this node

        :param children: Children `CBNNode` instances
        """

        self.children = children

    def plot_obs_distribution(self) -> None:
        """
        Plot the observational distribution overlaid by the curiosity density
        """

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
        r"""
        Run this node's causal mechanism to predict outcome data
        samples from the parent samples

        :param X: :math:`N` parent samples :math:`\mathbb{R}^{N \times d}`,
                  with :math:`d` equal to the number of parents
        :return: :math:`N` scalar outcome predictions as :math:`\mathbb{R}^N`
                 element-wise corresponding to `X`
        """

        if self.is_root or isinstance(self.causal_mechanism, StochasticModel):
            raise RuntimeError

        return self.causal_mechanism.draw_samples(X)

    def plot_image(self) -> None:
        """
        Scatter plot the predictions based on the whole domain of the node.
        If the node has more than 1 parent and thus the scatter plot would be >=3D,
        the dimensional reduction to 2D is performed using the UMAP method
        """

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
        """
        Record a new generative distribution of this node by providing the new samples

        :param samples: Samples to assemble the generative distribution from
        """

        if (
                self.gen_samples is not None
                and self.gen_samples.shape != samples.shape
        ):
            raise ValueError

        self.gen_samples = samples

    def propagate_samples(self) -> None:
        """
        Recursively perform prediction on this node and its children
        based on this node's parents' generative distributions
        """

        pa_samples = np.vstack([
            parent.gen_samples
            for parent in self.parents
        ]).T
        samples = self.predict(pa_samples)[:, 0]

        self.set_samples(samples)

        for child in self.children:
            child.propagate_samples()

    def explore(self, current_state: float | None = None, n_samples: int = 1) -> np.ndarray:
        """
        Sample new targets / outcomes of this node based on the curiosity distribution.
        If `current_state` (the current target) is given, the curiosity density will be
        increased in the Gaussian neighborhood around it.

        :param current_state: Scalar target of this node to construct local bias in the curiosity density around;
                              if `None`, the curiosity distribution is sampled without modulation
        :param n_samples: Number of targets to sample from the curiosity distribution
        :return: Array of sampled targets of this node
        """

        curiosity = (
            self.curiosity
            if current_state is None
            else self.bump_curiosity(current_state)
        )

        return curiosity.rvs(size=n_samples)


class MonteCarloCBN:
    """
    Framework for an untrained existing structural causal model (a causal Bayesian network)
    supporting Monte Carlo sampling and meta-learning
    """

    def __init__(
            self,
            untrained_scm: StructuralCausalModel | None = None,
            limits: dict[str, tuple[float, float]] | None = None,
            load_file: str | None = None
    ) -> None:
        """
        Initialize the framework with an existing untrained structural causal model

        :param untrained_scm: Existing untrained SCM encoding the causal relationships
                              between variables
        :param limits: Ranges of variables in the `untrained_scm` as `(min, max)` tuples.
                       If range is not specified for some variable, the range will be inferred
                       from data during the continual fitting process
        :param load_file: File path of a pickle file containing an SCM; if given, the framework
                          will be initialized from the structures in the file instead of using
                          `untrained_scm` and `limits`
        """

        if untrained_scm is None and load_file is None:
            raise ValueError

        # Priority load file with pre-existing structures, if given
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
            # Initialize the network nodes for each SCM variable
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

        # Copy the topology of the SCM to individual nodes
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
        """
        Causal graph of the SCM / CBN;
        by definition, this will be a directed acyclic graph (DAG)
        """

        return self.M.graph

    def _check_same_len(self, nodes: list[str]) -> bool:
        """
        Check whether all the specified nodes have observed the same number
        of data samples. This check is meant to force the preservation of
        the row-wise sample correspondence across the nodes

        :param nodes: Nodes to check
        :return: `True` if all the specified nodes have observed
                 the same amount of data, `False` otherwise
        """

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
        """
        Construct Pandas DataFrame containing all the observed data samples
        from the specified nodes

        :param nodes: Nodes to extract data from
        :return: DataFrame containing all the observed data samples
                 with the nodes organized in columns
        """

        return pd.DataFrame({
            node: (
                data
                if (data := self.nodes[node].observed_y) is not None
                else []
            )
            for node in nodes
        })

    def fit(self, data: pd.DataFrame) -> None:
        """
        Incrementally fit the whole network on new data. As the backend framework does not
        support incremental fitting, provided new `data` is concatenated with the previous
        samples and each node is refitted on the merged data.

        :param data: New data to be used for training. Columns of the DataFrame
                     must correspond to the graph variables and all the variables
                     must be contained.
        """

        # Check if the stored data samples from all the nodes match
        if not self._check_same_len(list(self.nodes)):
            raise ValueError

        # Concatenate the new data with the stored samples
        # to facilitate a pseudo-incremental fitting
        new_data = pd.concat([
            data,
            self._reconstruct_data(list(self.nodes))
        ])

        # Refit the SCM on the merged dataset and regenerate the samples
        # to construct a new generative distribution for each estimator (node)
        gcm.fit(self.M, new_data)
        samples = self.forward_sample()

        # Add new data to each node (incrementally this time),
        # instantiate new generative distributions, and recompute curiosity
        for node_name in self.nodes:
            node = self.nodes[node_name]

            node.observe(data.loc[:, node_name].to_numpy())
            node.set_samples(samples.loc[:, node_name].to_numpy())

    def fit_node(self, node: str, data: pd.DataFrame) -> None:
        """
        Incrementally fit a single node of the network

        :param node: Name of the target node to refit
        :param data: DataFrame with new data samples; must include data
                     for all the parents of the node and the target node itself
        """

        # Select data for the relevant nodes
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

        # Refit the causal mechanism of the target node
        # (i.e., the mapping between the parents and the node)
        gcm.fitting_sampling.fit_causal_model_of_target(self.M, node, new_data)

        # Add new data to the parents and the target node
        for node_name in rel_nodes:
            self.nodes[node_name].observe(data.loc[:, node_name].to_numpy())

        # Propagate samples from the target node onward
        self.nodes[node].propagate_samples()

    def forward_sample(self, n_samples: int = int(1e7)) -> pd.DataFrame:
        """
        Recursively draw sample from the distribution of each node.
        Start with sampling the root nodes and propagate the samples downstream
        to condition causally dependent nodes

        :param n_samples: Number of samples to draw for each node
        :return: DataFrame with `n_samples` data points for each node
        """

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
        """
        Perform interventional inference on the model by computing the interventional
        distributions across the network and sampling them afterwards. The observational data
        are sampled from the generative distributions

        :param interventions: Fixed values to set the specific nodes to
        :param aggregator: Optional function to post-process the interventional samples by
        :param n_samples: Number of samples to draw for each node distribution
        :return: DataFrame with `n_samples` data points for each node
        """

        samples = gcm.interventional_samples(
            causal_model=self.M,
            interventions={
                node_name: lambda _: node_value
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
            sampled_nodes: list[str] | set[str]
    ) -> dict[str, float]:
        """
        Generate scalar target values for each of the `sampled_nodes`
        based on the current state

        :param state: Values of all the nodes in `sampled_nodes`
        :param sampled_nodes: Target nodes to generate exploration values for
        :return: Exploration target values for all the nodes specified by `sampled_nodes`
        """

        for node_name in sampled_nodes:
            if node_name not in state:
                raise ValueError

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
            target_nodes: list[str] | set[str],
            readout_nodes: list[str] | set[str],
            init_delta: float = 0.1,
            delta_gain: float = 2.0,
            min_results: int = 100,
            verbose: bool = False
    ) -> dict[str, stats.rv_histogram]:
        r"""
        Generate hypothetical distributions based on the sampling targets.
        Target nodes' generative distributions are filtered to match the sampling
        targets and the corresponding data points are read out from the readout nodes.
        Hence, this process actually finds probable (as per the current knowledge
        of the estimators within the network) values of the readout variables,
        which should cause the target variables to approximately result in values
        specified by `sampling_targets`.

        If at least `min_results` matching data points are not found, the matching tolerance
        :math:`\delta` will be iteratively increased by `delta_gain` until the sufficient
        number of data points is reached

        :param sampling_targets: Values to set the target nodes to; must specify values
                                 for all the variables in `target_nodes`
        :param target_nodes: Nodes to sample first and match them with `sampling_targets`
        :param readout_nodes: Nodes to read out the values from, which should cause
                              the target nodes to result in the values specified by `sampling_targets`
        :param init_delta: Initial tolerance of the absolute difference between the desired target values
                           of the target nodes and their sampled data points
        :param delta_gain: Multiplication factor by which :math:`\delta` will be increased
                           after every failed iteration of rejection sampling
        :param min_results: Minimum number of matching data points for the rejection sampling
                            iteration to be successful
        :param verbose: If `True`, progress of the sampling process will be reported
        :return: Histogram hypothetical distributions of the readout nodes
        """

        if verbose:
            print(f'Starting backpropagated rejection sampling with delta = {init_delta:.2f}')

        if isinstance(target_nodes, set):
            target_nodes = list(target_nodes)

        if isinstance(readout_nodes, set):
            readout_nodes = list(readout_nodes)

        target_nodes: list[str]
        readout_nodes: list[str]

        delta = init_delta
        iteration = 0

        while True:
            if verbose:
                print(f'Iteration {iteration + 1} (delta = {delta:.2f}): ', end='')

            # Initialize the sample filter
            # IDs of the matching samples will be further checked and in case the corresponding
            # samples do not match for other variables, they will be rejected
            init_node = target_nodes[0]
            mask = np.abs(self.nodes[init_node].gen_samples - sampling_targets[init_node]) <= delta
            sample_ids = np.where(mask)[0]

            # Check the match for the rest of the target variables
            for node_name in target_nodes[1:]:
                # If the sample selection is already too small,
                # perform another iteration with increased tolerance
                if len(sample_ids) < min_results:
                    break

                # Get the cached data points sampled from the node's generative distribution
                rel_samples = self.nodes[node_name].gen_samples[sample_ids]

                # Filter out suboptimal samples
                mask = np.abs(rel_samples - sampling_targets[node_name]) <= delta
                sample_ids = sample_ids[mask]

            # End the sampling process if enough data points were found
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
            # Get the corresponding data points for the readout variables
            node = self.nodes[node_name]
            samples = node.gen_samples[sample_ids]

            # Construct a histogram distribution from the collected data points
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
            explored_nodes: list[str],
            readout_nodes: list[str],
            init_delta: float = 0.1,
            delta_gain: float = 2.0,
            min_results: int = 100,
            verbose: bool = False
    ) -> dict[str, stats.rv_histogram]:
        r"""
        Perform a Monte Carlo exploration using this model.
        For the nodes specified by `explored_nodes`, new exploration target values
        will be computed to expand the current observational distribution of these nodes.
        Afterwards, the hypothetical distributions of the readout nodes will be computed,
        which should causally yield the set exploration targets in the explored nodes

        :param state: The current state of the system's variables
        :param fixed_nodes: Nodes which will be set to the values specified by `state`
                            instead of having exploration targets generated
        :param explored_nodes: Nodes which will have new exploration targets generated
        :param readout_nodes: Nodes for which hypothetical distributions will be computed
        :param init_delta: Initial tolerance of the absolute difference between the desired target values
                           of the target nodes and their sampled data points
        :param delta_gain: Multiplication factor by which :math:`\delta` will be increased
                           after every failed iteration of rejection sampling
        :param min_results: Minimum number of matching data points for the rejection sampling
                            iteration to be successful
        :param verbose: If `True`, progress of the hypothesis construction will be reported
        :return: Histogram hypothetical distributions of the readout nodes
        """

        # Generate new exploration targets for `explored_nodes`
        sampling_targets = self.get_exploration_target(state, explored_nodes) | {
            node_name: state[node_name]
            for node_name in fixed_nodes
        }

        readout_distribs = self.construct_hypothesis(
            sampling_targets=sampling_targets,
            target_nodes=explored_nodes + fixed_nodes,
            readout_nodes=readout_nodes,
            init_delta=init_delta,
            delta_gain=delta_gain,
            min_results=min_results,
            verbose=verbose
        )

        return readout_distribs

    def save(self, filename: str) -> None:
        """
        Save this framework to a pickle file to be loaded by the constructor later

        :param filename: Filename to save to
        """

        data = {
            'scm': self.M,
            'nodes': self.nodes
        }

        with open(filename, 'wb') as f:
            pickle.dump(data, f)
