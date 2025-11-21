# from dynamic_bayesian_network.builder import DynamicBayesianNetworkBuilder
# from dynamic_bayesian_network.dynamic_bayesian_network import Node
# from dynamic_bayesian_network.unroller import unroll


# class TestUnroller:
#     # def __init__(self):
#     #     self.torch_settings = TorchSettings()

#     def test_hmm_mapping(self, torch_settings):
#         # Assign
#         builder = DynamicBayesianNetworkBuilder()

#         Q = Node.random(
#             cpt_size=(2, 2),
#             is_sequential=True,
#             torch_settings=torch_settings,
#             prior_size=(2),
#             name="Q",
#         )
#         Y = Node.random(
#             cpt_size=(2, 3),
#             is_sequential=False,
#             torch_settings=torch_settings,
#             name="Y",
#         )

#         builder.add_node(Q, sequential_parents=Q)
#         builder.add_node(Y, parents=Q)

#         dbn = builder.build()

#         # Act
#         _, mapping = unroll(dbn, sequence_length=3)

#         # Assert
#         assert mapping.keys() == {0, 1, 2}

#         for sub_mapping in mapping.values():
#             assert sub_mapping.keys() == {Q, Y}

#         all_nodes = [[node for node in sub_mapping.values()] for sub_mapping in mapping.values()]
#         # assert len(all_nodes) == len(set(all_nodes))

# from dynamic_bayesian_network.dynamic_bayesian_network import DynamicBayesianNetwork
# import dynamic_bayesian_network


# def test_hoi():
#     # dbn = DynamicBayesianNetwork([], {}, {})

#     # assert dbn is None

#     # assert 1 == 1
#     print("DOEI")
#     print(dir(dynamic_bayesian_network))
#     assert dynamic_bayesian_network.__version__

from unittest import TestCase

import torch
from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.samplers.torch_sampler import TorchBayesianNetworkSampler
from common.torch_settings import TorchSettings
from scipy import stats


class TestHoi(TestCase):
    def get_torch_settings(self) -> TorchSettings:
        return TorchSettings(
            device="cpu",
            dtype="float64",
        )

    def setUp(self):
        self.num_samples = 10000
        self.alpha = 0.001

    def test_hoi(self):
        # Assign
        device = self.get_torch_settings().device
        dtype = self.get_torch_settings().dtype

        p_true = torch.tensor([1 / 5, 4 / 5], device=device, dtype=dtype)
        Q = Node(p_true, name="Q")

        nodes = [Q]
        parents = {
            Q: [],
        }
        network = BayesianNetwork(nodes, parents)

        # Act
        sut = TorchBayesianNetworkSampler(network, torch_settings=self.get_torch_settings())

        samples = sut.sample(self.num_samples, nodes)

        # Assert
        samples = samples.cpu()

        expected = p_true.cpu() * self.num_samples
        actual0 = (samples == 0).sum()
        actual1 = (samples == 1).sum()

        _, p = stats.chisquare([actual0, actual1], expected)

        self.assertGreater(p, self.alpha)
