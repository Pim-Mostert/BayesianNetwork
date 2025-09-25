# %% Imports
import logging

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from bayesian_network.bayesian_network import BayesianNetwork, Node
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

BATCH_SIZE = 50
LEARNING_RATE = 0.1

TRUE_MEANS_NOISE = 0

NUM_EPOCHS = 1

LENGTH_SUBSET = 60000


# %% Load data
gamma = 0.001

mnist = torchvision.datasets.MNIST(
    "./dev_experiments/mnist",
    train=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.flatten()),
            transforms.Lambda(lambda x: x * (1 - gamma) + gamma / 2),
        ]
    ),
    download=True,
)
mnist_subset = Subset(mnist, range(0, LENGTH_SUBSET))
height, width = 28, 28
num_classes = 10

iterations_per_epoch = len(mnist_subset) / BATCH_SIZE
assert int(iterations_per_epoch) == iterations_per_epoch, (
    "len(mnist) / BATCH_SIZE should be an integer"
)

# %% True means

data_loader = DataLoader(
    dataset=mnist,
    batch_size=1000,
    shuffle=False,
)

# To store sums and counts per class
sums = torch.zeros(num_classes, height * width)
counts = torch.zeros(num_classes)

# Iterate over batches
for images, labels in data_loader:
    for i in range(num_classes):
        mask = labels == i
        sums[i] += images[mask].sum(dim=0)
        counts[i] += mask.sum()

# Compute means
mu_true = sums / counts[:, None]

# %% Define network

Q = Node(
    torch.ones(
        (num_classes),
        device=torch_settings.device,
        dtype=torch_settings.dtype,
    )
    / num_classes,
    name="Q",
)
noise = torch.rand(
    (num_classes, height * width),
    device=torch_settings.device,
    dtype=torch_settings.dtype,
)

mu = (1 - TRUE_MEANS_NOISE) * mu_true + TRUE_MEANS_NOISE * noise
mu = torch.stack([1 - mu, mu], dim=2)

Ys = [
    Node(mu[:, iy + ix * height], name=f"Y_{iy}x{ix}")
    for iy in range(height)
    for ix in range(width)
]
nodes = [Q] + Ys
parents = {Y: [Q] for Y in Ys}
parents[Q] = []

network = BayesianNetwork(nodes, parents)

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


evaluator_batch_size = 2000
evaluator = BatchEvaluator(
    inference_machine_factory=lambda network: SpaInferenceMachine(
        settings=SpaInferenceMachineSettings(
            torch_settings=torch_settings,
            num_iterations=4,
            average_log_likelihood=True,
        ),
        bayesian_network=network,
        observed_nodes=Ys,
        num_observations=evaluator_batch_size,
    ),
    evidence_loader=EvidenceLoader(
        DataLoader(
            dataset=mnist_subset,
            batch_size=evaluator_batch_size,
        ),
        transform=transform,
    ),
    should_evaluate=lambda epoch, iteration: (iteration % 100) == 0,
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
    inference_machine_factory=lambda network: SpaInferenceMachine(
        settings=SpaInferenceMachineSettings(
            torch_settings=torch_settings,
            num_iterations=4,
            average_log_likelihood=True,
        ),
        bayesian_network=network,
        observed_nodes=Ys,
        num_observations=BATCH_SIZE,
    ),
    settings=EmBatchOptimizerSettings(
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
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

epochs = [epoch for epoch, iteration in evaluator.log_likelihoods.keys()]
train_values = [log.ll for log in logger.logs if log.iteration == 0]
eval_values = list(evaluator.log_likelihoods.values())

plt.figure()
plt.plot(epochs, train_values, label="Train")
plt.plot(epochs, eval_values, label="Eval")
plt.xlabel("Epochs")
plt.legend()

# %%

iterations = [log.iteration + log.epoch * 100 for log in logger.logs]
train_values = [log.ll for log in logger.logs]

plt.figure()
plt.plot(iterations, train_values, label="Train")
plt.xlabel("Iterations")
plt.legend()

# %%

batch_size = 2000
evidence_loader = EvidenceLoader(
    DataLoader(
        dataset=mnist_subset,
        batch_size=batch_size,
    ),
    transform=transform,
)


# SpaInferenceMachineV1 geeft 't probleem ook
def inference_machine_factory(network):
    return SpaInferenceMachine(
        settings=SpaInferenceMachineSettings(
            torch_settings=torch_settings,
            num_iterations=4,
            average_log_likelihood=True,
        ),
        bayesian_network=network,
        observed_nodes=Ys,
        num_observations=batch_size,
    )


inference_machine = inference_machine_factory(network)

for evidence in iter(evidence_loader):
    inference_machine.enter_evidence(evidence)
    ll = inference_machine.log_likelihood()
    ll *= len(evidence) / evidence_loader.num_observations
    print(ll)

# %%

with open("network.pickle", "wb") as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(network, f, pickle.HIGHEST_PROTOCOL)
# %%
