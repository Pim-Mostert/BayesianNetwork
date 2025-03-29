# %% Imports
import torch
import torchvision
from torchvision.transforms import transforms

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.torch_sum_product_algorithm_inference_machine import (
    TorchSumProductAlgorithmInferenceMachine,
)
from bayesian_network.interfaces import IInferenceMachine
from bayesian_network.optimizers.em_optimizer import EmOptimizer, EmOptimizerSettings

# %% Configuration
torch_settings = TorchSettings(
    device=torch.device("cpu"),
    dtype=torch.float64,
)

gamma = 0.001

# %% Load data
mnist = torchvision.datasets.MNIST(
    "./experiments/mnist",
    download=True,
    train=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.flatten()),
            transforms.Lambda(lambda x: x * (1 - gamma) + gamma / 2),
        ]
    ),
)

# %% Define network
height = 28
width = 28
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

num_iterations = 10


# %% Fit network
def inference_machine_factory(
    bayesian_network: BayesianNetwork,
) -> IInferenceMachine:
    return TorchSumProductAlgorithmInferenceMachine(
        bayesian_network=bayesian_network,
        observed_nodes=Ys,
        torch_settings=torch_settings,
        num_iterations=3,
        num_observations=len(mnist),
        callback=lambda *args: None,
    )


def callback(ll, iteration, duration):
    print(f"Finished iteration {iteration}/{num_iterations} - ll: {ll}" " - it took: {duration} s")


em_optimizer = EmOptimizer(
    network,
    inference_machine_factory,
    settings=EmOptimizerSettings(
        num_iterations=num_iterations,
        iteration_callback=callback,
    ),
)
em_optimizer.optimize(mnist)
