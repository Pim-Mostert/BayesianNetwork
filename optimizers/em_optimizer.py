class EmOptimizer:
    def __init__(self, bayesian_network, inference_machine_factory):
        self.bayesian_network = bayesian_network
        self.inference_machine_factory = inference_machine_factory

    def optimize(self, evidence, num_iterations, iteration_callback):
        for iteration in range(num_iterations):
            inference_machine = self.inference_machine_factory(self.bayesian_network)
            ll = inference_machine.enter_evidence(evidence)

            p_conditionals = self._e_step(inference_machine)

            self._m_step(p_conditionals)

            iteration_callback(ll, iteration)

    def _e_step(self, inference_machine):
        p_conditionals = {}

        for node in self.bayesian_network.nodes:
            parents = self.bayesian_network.parents[node]

            p_conditionals[node] = inference_machine.infer(parents + [node])

        return p_conditionals

    def _m_step(self, p_conditionals):
        for node in self.bayesian_network.nodes:
            cpt = p_conditionals[node].sum(axis=0)

            cpt /= cpt.sum(axis=-1, keepdims=True)

            node.cpt = cpt.numpy()
