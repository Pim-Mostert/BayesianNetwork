# %%

import pickle

from torch.utils.data import DataLoader
import torchvision

from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.evidence import EvidenceLoader
from bayesian_network.inference_machines.spa_v3.spa_inference_machine import (
    SpaInferenceMachine,
    SpaInferenceMachineSettings,
)


import torch
from torchvision import transforms

from bayesian_network.inference_machines.evidence import Evidence

# %%

with open("network.pickle", "rb") as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    network = pickle.load(f)

Ys = network.nodes[1:]

# %%

torch_settings = TorchSettings(
    device="cpu",
    dtype="float64",
)

# %%

gamma = 0.001
mnist = torchvision.datasets.MNIST(
    "./experiments/mnist",
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.flatten()),
            transforms.Lambda(lambda x: x * (1 - gamma) + gamma / 2),
        ]
    ),
)

height, width = 28, 28
num_classes = 10

# %%


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
        torch_settings=torch_settings,
    )


batch_size = 2000
evidence_loader = EvidenceLoader(
    DataLoader(
        dataset=mnist,
        batch_size=batch_size,
    ),
    transform=transform,
)


Q = network.nodes[0]
inference_machine = SpaInferenceMachine(
    settings=SpaInferenceMachineSettings(
        torch_settings=torch_settings,
        num_iterations=4,
        average_log_likelihood=True,
    ),
    bayesian_network=network,
    observed_nodes=Ys,
    num_observations=batch_size,
)

for evidence in iter(evidence_loader):
    inference_machine.enter_evidence(evidence)
    ll = inference_machine.log_likelihood()
    ll *= len(evidence) / evidence_loader.num_observations
    print(ll)

# %%
