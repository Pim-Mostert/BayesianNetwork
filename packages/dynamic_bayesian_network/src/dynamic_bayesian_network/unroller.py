from bayesian_network import bayesian_network as bn
from bayesian_network.builder import BayesianNetworkBuilder

from dynamic_bayesian_network.dynamic_bayesian_network import DynamicBayesianNetwork


def unroll(
    dynamic_network: DynamicBayesianNetwork,
    sequence_length: int,
):
    builder = BayesianNetworkBuilder()
    nodes_map = {}

    ### t = 0
    # Add nodes
    nodes_map[0] = {}
    for dynamic_node in dynamic_network.nodes:
        cpt = dynamic_node.prior if dynamic_node.is_sequential else dynamic_node.cpt

        node = bn.Node(
            cpt=cpt,
            name=f"{dynamic_node.name}_t0",
        )

        nodes_map[0][dynamic_node] = node

        builder.add_node(node)

    # Set parents
    for dynamic_node in dynamic_network.nodes:
        node = nodes_map[0][dynamic_node]
        parents = [nodes_map[0][parent] for parent in dynamic_network.parents_of(dynamic_node)]

        builder.set_parents(node=node, parents=parents)

    ### t > 0
    for i in range(1, sequence_length):
        # Add nodes
        nodes_map[i] = {}

        for dynamic_node in dynamic_network.nodes:
            node = bn.Node(
                cpt=dynamic_node.cpt,
                name=f"{dynamic_node.name}_t{i}",
            )

            nodes_map[i][dynamic_node] = node

            builder.add_node(node)

        # Set parents
        for dynamic_node in dynamic_network.nodes:
            node = nodes_map[i][dynamic_node]

            # Parents
            parents = [nodes_map[i][parent] for parent in dynamic_network.parents_of(dynamic_node)]
            sequential_parents = [
                nodes_map[i - 1][sequential_parent]
                for sequential_parent in dynamic_network.sequential_parents_of(dynamic_node)
            ]

            all_parents = [*sequential_parents, *parents]
            builder.set_parents(node=node, parents=all_parents)

    ### Build and return
    network = builder.build()

    return network, nodes_map
