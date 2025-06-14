# %% Imports
import matplotlib.pyplot as plt
import torch
import torchvision

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.evidence import Evidence, EvidenceBatches
from bayesian_network.inference_machines.spa_v3.spa_inference_machine import (
    SpaInferenceMachine,
    SpaInferenceMachineSettings,
)
from bayesian_network.optimizers.common import (
    OptimizationEvaluator,
    OptimizationEvaluatorSettings,
    OptimizerLogger,
)
from bayesian_network.optimizers.em_batch_optimizer import (
    EmBatchOptimizer,
    EmBatchOptimizerSettings,
)

import logging

logging.basicConfig(level=logging.INFO)

# %% Configuration
torch_settings = TorchSettings(
    device="cpu",
    dtype="float64",
)

# %% Load data
mnist = torchvision.datasets.MNIST(
    "./experiments/mnist",
    train=True,
    # transform=transforms.ToTensor(),
    download=True,
)
data = mnist.data / 255

height, width = data.shape[1:3]
num_features = height * width
num_observations = data.shape[0]

# Morph into evidence structure
data = data.reshape([num_observations, num_features])

gamma = 0.001
evidence = Evidence(
    [torch.stack([1 - x, x]).T for x in data.T * (1 - gamma) + gamma / 2],
    torch_settings,
)

# Make selection
evidence = evidence[:1000]

# %% Define network
num_classes = 10

# Create network
Q = Node(
    torch.ones(
        (num_classes),
        device=torch_settings.device,
        dtype=torch_settings.dtype,
    )
    / num_classes,
    name="Q",
)
mu = (
    torch.rand(
        (height, width, num_classes),
        device=torch_settings.device,
        dtype=torch_settings.dtype,
    )
    * 0.2
    + 0.4
)
mu = torch.stack([1 - mu, mu], dim=3)
Ys = [Node(mu[iy, ix], name=f"Y_{iy}x{ix}") for iy in range(height) for ix in range(width)]

nodes = [Q] + Ys
parents = {Y: [Q] for Y in Ys}
parents[Q] = []

network = BayesianNetwork(nodes, parents)


# %% Fit network
logger = OptimizerLogger()

evaluator = OptimizationEvaluator(
    OptimizationEvaluatorSettings(iteration_interval=1),
    inference_machine_factory=lambda network: SpaInferenceMachine(
        settings=SpaInferenceMachineSettings(
            torch_settings=torch_settings,
            num_iterations=3,
            average_log_likelihood=False,
        ),
        bayesian_network=network,
        observed_nodes=Ys,
        num_observations=evidence.num_observations,
    ),
    evidence=evidence,
)

em_batch_optimizer_settings = EmBatchOptimizerSettings(
    num_iterations=50,
    learning_rate=0.01,
)

batches = EvidenceBatches(evidence, 50)

em_optimizer = EmBatchOptimizer(
    bayesian_network=network,
    inference_machine_factory=lambda network: SpaInferenceMachine(
        settings=SpaInferenceMachineSettings(
            torch_settings=torch_settings,
            num_iterations=3,
            average_log_likelihood=True,
        ),
        bayesian_network=network,
        observed_nodes=Ys,
        num_observations=batches.batch_size,
    ),
    settings=em_batch_optimizer_settings,
    logger=logger,
    evaluator=evaluator,
)
em_optimizer.optimize(batches)

# %% Plot

Ys = network.nodes[1:]
w = torch.stack([y.cpt.cpu() for y in Ys])

plt.figure()
for i in range(0, 10):
    plt.subplot(4, 3, i + 1)
    plt.imshow(w[:, i, 1].reshape(28, 28))
    plt.colorbar()
    plt.clim(0, 1)
