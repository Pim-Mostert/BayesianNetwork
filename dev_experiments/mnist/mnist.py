# %% Imports
import logging

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from bayesian_network.bayesian_network import BayesianNetworkBuilder, Node
from bayesian_network.common.statistics import generate_random_probability_matrix
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.evidence import Evidence, EvidenceLoader
from bayesian_network.inference_machines.spa_v3.spa_inference_machine import (
    SpaInferenceMachine,
    SpaInferenceMachineSettings,
)
from bayesian_network.optimizers.common import (
    BatchEvaluator,
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

NUM_EPOCHS = 1
LEARNING_RATE = 0.1
REGULARIZATION = 0.01
BATCH_SIZE = 100

# %% Load data
mnist = torchvision.datasets.MNIST(
    "./dev_experiments/mnist",
    train=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.flatten()),
        ]
    ),
    download=True,
)
mnist_subset = Subset(mnist, range(0, 10000))
height, width = 28, 28

iterations_per_epoch = len(mnist_subset) / BATCH_SIZE
assert int(iterations_per_epoch) == iterations_per_epoch, (
    "len(mnist_subset) / BATCH_SIZE should be an integer"
)
iterations_per_epoch = int(iterations_per_epoch)

# %% Define network

num_classes = 10
num_features = 2

Q = Node(
    torch.ones(
        (num_classes),
        device=torch_settings.device,
        dtype=torch_settings.dtype,
    )
    / num_classes,
    name="Q",
)
builder = BayesianNetworkBuilder().add_node(Q)

F1 = Node.random((num_classes, num_features), torch_settings, "F1")
F2 = Node.random((num_classes, num_features), torch_settings, "F2")
builder.add_node(F1, parents=Q).add_node(F2, parents=Q)

Ys = []
for iy in range(height):
    for ix in range(width):
        node = Node(
            generate_random_probability_matrix((num_features, num_features, 2), torch_settings),
            name=f"Y_{iy}x{ix}",
        )
        builder.add_node(node, parents=[F1, F2])
        Ys.append(node)

network = builder.build()

# %% Fit network
logger = OptimizerLogger()


def transform(batch: torch.Tensor) -> Evidence:
    return Evidence(
        [
            torch.stack(
                [
                    1 - x,
                    x,
                ],
                dim=1,
            )
            for x in batch.T
        ],
        torch_settings,
    )


def wrapper(num_observations):
    def inference_machine_factory(network):
        return SpaInferenceMachine(
            settings=SpaInferenceMachineSettings(
                torch_settings=torch_settings,
                num_iterations=10,
                allow_loops=True,
                average_log_likelihood=True,
            ),
            bayesian_network=network,
            observed_nodes=Ys,
            num_observations=num_observations,
        )

    return inference_machine_factory


# nx.draw_networkx(wrapper(100)(network).factor_graph.G)

evaluator_batch_size = 1000
evaluator = BatchEvaluator(
    inference_machine_factory=wrapper(evaluator_batch_size),
    evidence_loader=EvidenceLoader(
        DataLoader(
            dataset=mnist_subset,
            batch_size=evaluator_batch_size,
        ),
        transform=transform,
    ),
    should_evaluate=lambda epoch, iteration: (
        (iteration == 0)
        # or (iteration == int(iterations_per_epoch / 2))
        or (epoch == (NUM_EPOCHS - 1) and (iteration == iterations_per_epoch - 1))
    ),
)


evidence_loader = EvidenceLoader(
    DataLoader(
        dataset=mnist_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    ),
    transform=transform,
)

em_optimizer = EmBatchOptimizer(
    bayesian_network=network,
    inference_machine_factory=wrapper(BATCH_SIZE),
    settings=EmBatchOptimizerSettings(
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        regularization=REGULARIZATION,
    ),
    logger=logger,
    evaluator=evaluator,
)

em_optimizer.optimize(evidence_loader)


# %% Plot log_likelihood
train_iterations = [log.epoch * iterations_per_epoch + log.iteration for log in logger.logs]
train_values = [log.ll for log in logger.logs]
eval_iterations = [
    epoch * iterations_per_epoch + iteration
    for epoch, iteration in evaluator.log_likelihoods.keys()
]
eval_values = list(evaluator.log_likelihoods.values())

plt.figure()
plt.plot(train_iterations, train_values, label="Train")
plt.plot(eval_iterations, eval_values, label="Eval")
plt.xlabel("Iterations")
plt.ylabel("Average log-likelihood")
plt.legend()

# %% Plot

w = torch.stack([y.cpt.cpu() for y in Ys])[:, :, 0, :]

plt.figure()
for i in range(0, w.shape[1]):
    plt.subplot(13, 8, i + 1)
    plt.imshow(w[:, i, 1].reshape(28, 28))
    plt.colorbar()
    plt.clim(0, 1)

# %%
