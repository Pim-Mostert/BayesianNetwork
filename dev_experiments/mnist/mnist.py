# %% Imports
import logging

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.evidence import EvidenceLoader
from bayesian_network.inference_machines.spa_v3.spa_inference_machine import (
    SpaInferenceMachine,
    SpaInferenceMachineSettings,
)
from bayesian_network.optimizers.common import (
    BatchEvaluator,
    BatchEvaluatorSettings,
    OptimizerLogger,
)
from bayesian_network.optimizers.em_batch_optimizer import (
    EmBatchOptimizer,
    EmBatchOptimizerSettings,
)

logging.basicConfig(level=logging.INFO)

# %% Configuration
torch_settings = TorchSettings(
    device="cpu",
    dtype="float64",
)

# %% Load data
gamma = 0.001

mnist = torchvision.datasets.MNIST(
    "./dev_experiments/mnist",
    train=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * (1 - gamma) + gamma / 2),
            transforms.Lambda(lambda x: x.flatten()),
        ]
    ),
    download=True,
)
mnist_subset = Subset(mnist, range(0, 1000))
height, width = 28, 28

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


evaluator_batch_size = 1000
evaluator = BatchEvaluator(
    settings=BatchEvaluatorSettings(
        iteration_interval=10,
        torch_settings=torch_settings,
    ),
    inference_machine_factory=lambda network: SpaInferenceMachine(
        settings=SpaInferenceMachineSettings(
            torch_settings=torch_settings,
            num_iterations=3,
            average_log_likelihood=True,
        ),
        bayesian_network=network,
        observed_nodes=Ys,
        num_observations=evaluator_batch_size,
    ),
    data_loader=DataLoader(
        dataset=mnist_subset,
        batch_size=evaluator_batch_size,
    ),
)


batch_size = 100
evidence_loader = EvidenceLoader(
    DataLoader(
        dataset=mnist_subset,
        batch_size=batch_size,
        shuffle=True,
    ),
    torch_settings=torch_settings,
)


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
        num_observations=batch_size,
    ),
    settings=EmBatchOptimizerSettings(
        learning_rate=0.002,
        num_epochs=25,
    ),
    logger=logger,
    evaluator=evaluator,
)

em_optimizer.optimize(evidence_loader)

# %% Plot

Ys = network.nodes[1:]
w = torch.stack([y.cpt.cpu() for y in Ys])

plt.figure()
for i in range(0, 10):
    plt.subplot(4, 3, i + 1)
    plt.imshow(w[:, i, 1].reshape(28, 28))
    plt.colorbar()
    plt.clim(0, 1)

# %%

plt.figure()
plt.plot(*logger.log_likelihoods, label="Train")
plt.plot(*evaluator.log_likelihoods, label="Eval")
plt.legend()

# %%
