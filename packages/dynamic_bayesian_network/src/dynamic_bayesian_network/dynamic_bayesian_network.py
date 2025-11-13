import torch
from common.statistics import generate_random_probability_matrix, is_probability_matrix
from common.torch_settings import TorchSettings


class Node:
    def __repr__(self):
        return self._name if self._name else super().__repr__()

    def __init__(
        self,
        cpt: torch.Tensor,
        is_sequential: bool,
        prior: torch.Tensor | None = None,
        name=None,
    ):
        if not cpt | is_probability_matrix():
            raise ValueError("The CPT should sum to 1 along the last dimension.")

        if is_sequential:
            if prior is None:
                raise ValueError("Prior should be specified for a sequential node")

            if not prior | is_probability_matrix():
                raise ValueError("The prior should sum to 1 along the last dimension.")
        else:
            if prior:
                raise ValueError("Prior should only be specified for sequential nodes.")

        self.num_states: int = cpt.shape[-1]
        self._cpt = cpt
        self._is_sequential = is_sequential
        self._prior = prior
        self._name = name

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def is_sequential(self) -> bool:
        return self._is_sequential

    @property
    def cpt(self) -> torch.Tensor:
        return self._cpt

    @cpt.setter
    def cpt(self, cpt: torch.Tensor):
        self._cpt = cpt

    @property
    def prior(self) -> torch.Tensor:
        if not self._is_sequential:
            raise RuntimeError("Only sequential nodes have a prior.")

        assert self._prior

        return self._prior

    @prior.setter
    def prior(self, prior: torch.Tensor):
        if not self._is_sequential:
            raise RuntimeError("Only sequential nodes have a prior.")

        self._prior = prior

    @classmethod
    def random(
        cls,
        cpt_size,
        is_sequential: bool,
        torch_settings: TorchSettings,
        prior_size=None,
        name: str | None = None,
    ):
        cpt = generate_random_probability_matrix(cpt_size, torch_settings)

        if is_sequential:
            if not prior_size:
                raise ValueError("Prior_size should be specified for a sequential node")

            prior = generate_random_probability_matrix(prior_size, torch_settings)

            return Node(cpt=cpt, is_sequential=True, prior=prior, name=name)
        else:
            return Node(cpt=cpt, is_sequential=False, name=name)


class DynamicBayesianNetwork:
    def __init__(
        self,
        nodes: list[Node],
        parents: dict[Node, list[Node]],
        sequential_parents: dict[Node, list[Node]],
    ):
        self._nodes = nodes
        self._parents = parents
        self._sequential_parents = sequential_parents

    @property
    def nodes(self):
        return self._nodes
