import random
from typing import Callable, Optional

from bayesian_network.bayesian_network import BayesianNetwork
from bayesian_network.interfaces import IInferenceMachine
from bayesian_network.optimizers.em_optimizer import EmOptimizer, EmOptimizerSettings


class EmBatchOptimizerSettings(EmOptimizerSettings):
    def __init__(
        self,
        num_iterations=10,
        mini_batch_size=100,
        learning_rate=0.001,
        mini_batch_callback=None,
        iteration_callback=None,
    ):
        super().__init__(num_iterations, iteration_callback)
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.mini_batch_callback = mini_batch_callback


class EmBatchOptimizer(EmOptimizer):
    def __init__(
        self,
        bayesian_network: BayesianNetwork,
        inference_machine_factory: Callable[[BayesianNetwork], IInferenceMachine],
        settings: Optional[EmBatchOptimizerSettings] = None,
    ):
        self.bayesian_network = bayesian_network
        self.inference_machine_factory = inference_machine_factory
        self.settings = settings or EmBatchOptimizerSettings()

    def optimize(self, evidence):
        # Shuffle
        random_indices = list(range(len(evidence)))
        random.shuffle(random_indices)

        # Split the data into minibatches
        # DIT WERKT NIET. EEN PROBLEEM IS OOK DAT IK VEEL TE WEINIG DATAPUNTEN HEB OM MINIBATCHES TE TESTEN
        mini_batches = [
            evidence[random_indices[i : (i + self.settings.mini_batch_size)]]
            for i in range(0, len(evidence), self.settings.mini_batch_size)
        ]

        for i, mini_batch in enumerate(mini_batches):
            super().optimize(mini_batch)

            if self.settings.mini_batch_callback:
                self.settings.mini_batch_callback(i, len(mini_batches))

    # def _iterate(self, evidence):
    #     for iteration in range(self.settings.num_iterations):
    #         # Construct inference machine and enter evidence
    #         inference_machine = self.inference_machine_factory(
    #             self.bayesian_network
    #         )
    #         inference_machine.enter_evidence(evidence)
    #         ll = inference_machine.log_likelihood()

    #         # E-step
    #         p_conditionals = self._e_step(inference_machine)

    #         # M-step
    #         self._m_step(p_conditionals)

    #         # User feedback
    #         if self.settings.iteration_callback:
    #             self.settings.iteration_callback(
    #                 iteration, ll, self.bayesian_network
    #             )

    # def _e_step(
    #     self, inference_machine: IInferenceMachine
    # ) -> List[torch.Tensor]:
    #     # List[torch.Tensor((observations x parent1 x parent2 x ... x child))]
    #     p_all = inference_machine.infer_nodes_with_parents(
    #         self.bayesian_network.nodes
    #     )

    #     # Average over observations
    #     p_conditionals = [p.mean(dim=0) for p in p_all]

    #     return p_conditionals

    # def _m_step(self, p_conditionals: List[torch.Tensor]):
    #     for node, p_conditional in zip(
    #         self.bayesian_network.nodes, p_conditionals
    #     ):
    #         # Normalize to conditional probability distribution
    #         cpt = p_conditional / p_conditional.sum(dim=-1, keepdim=True)

    #         # Update node
    #         alpha = self.settings.learning_rate
    #         node.cpt = (1 - alpha) * node.cpt + alpha * cpt
