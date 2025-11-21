from dynamic_bayesian_network.builder import DynamicBayesianNetworkBuilder
from dynamic_bayesian_network.dynamic_bayesian_network import Node
from dynamic_bayesian_network.unroller import unroll


class TestUnroller:
    def test_hmm_mapping(self, torch_settings):
        # Assign
        builder = DynamicBayesianNetworkBuilder()

        Q = Node.random(
            cpt_size=(2, 2),
            is_sequential=True,
            torch_settings=torch_settings,
            prior_size=(2),
            name="Q",
        )
        Y = Node.random(
            cpt_size=(2, 3),
            is_sequential=False,
            torch_settings=torch_settings,
            name="Y",
        )

        builder.add_node(Q, sequential_parents=Q)
        builder.add_node(Y, parents=Q)

        dbn = builder.build()

        # Act
        _, mapping = unroll(dbn, sequence_length=3)

        # Assert
        assert mapping.keys() == {0, 1, 2}

        for sub_mapping in mapping.values():
            assert sub_mapping.keys() == {Q, Y}

        all_nodes = [node for sub_mapping in mapping.values() for node in sub_mapping.values()]
        assert len(all_nodes) == len(set(all_nodes))
