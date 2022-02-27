import numpy as np
import torch

import common as cmn
import matplotlib.pyplot as plt

from common.statistics import generate_random_probability_matrix

torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# True network
from common.utilities import Empty, Cfg
from inference_engines.torch_naive_inference_machine import TorchNaiveInferenceMachine
from model.bayesian_network import BayesianNetwork
from model.nodes import CPTNode
from optimizers.em_optimizer import EmOptimizer
from samplers.torch_sampler import TorchSampler

node0_1 = CPTNode(np.array([1/5, 4/5], dtype=np.float64))
node0_2 = CPTNode(np.array([[0.2, 0.8], [0.3, 0.7]], dtype=np.float64))
node0_3 = CPTNode(np.array([[[0.4, 0.6], [0.5, 0.5]], [[0.5, 0.5], [0.8, 0.2]]], dtype=np.float64))
node0_4_1 = CPTNode(np.array([[[[0, 1], [1, 0]], [[1, 0], [1, 0]]], [[[1, 0], [1, 0]], [[1, 0], [1, 0]]]], dtype=np.float64))
node0_4_2 = CPTNode(np.array([[[[1, 0], [0, 1]], [[1, 0], [1, 0]]], [[[1, 0], [1, 0]], [[1, 0], [1, 0]]]], dtype=np.float64))
node0_4_3 = CPTNode(np.array([[[[1, 0], [1, 0]], [[0, 1], [1, 0]]], [[[1, 0], [1, 0]], [[1, 0], [1, 0]]]], dtype=np.float64))
node0_4_4 = CPTNode(np.array([[[[1, 0], [1, 0]], [[1, 0], [0, 1]]], [[[1, 0], [1, 0]], [[1, 0], [1, 0]]]], dtype=np.float64))
node0_4_5 = CPTNode(np.array([[[[1, 0], [1, 0]], [[1, 0], [1, 0]]], [[[0, 1], [1, 0]], [[1, 0], [1, 0]]]], dtype=np.float64))
node0_4_6 = CPTNode(np.array([[[[1, 0], [1, 0]], [[1, 0], [1, 0]]], [[[1, 0], [0, 1]], [[1, 0], [1, 0]]]], dtype=np.float64))
node0_4_7 = CPTNode(np.array([[[[1, 0], [1, 0]], [[1, 0], [1, 0]]], [[[1, 0], [1, 0]], [[0, 1], [1, 0]]]], dtype=np.float64))
node0_4_8 = CPTNode(np.array([[[[1, 0], [1, 0]], [[1, 0], [1, 0]]], [[[1, 0], [1, 0]], [[1, 0], [0, 1]]]], dtype=np.float64))

nodes0 = [node0_1, node0_2, node0_3, node0_4_1, node0_4_2, node0_4_3, node0_4_4, node0_4_5, node0_4_6, node0_4_7, node0_4_8]
parents0 = {
    node0_1: [],
    node0_2: [node0_1],
    node0_3: [node0_1, node0_2],
    node0_4_1: [node0_1, node0_2, node0_3],
    node0_4_2: [node0_1, node0_2, node0_3],
    node0_4_3: [node0_1, node0_2, node0_3],
    node0_4_4: [node0_1, node0_2, node0_3],
    node0_4_5: [node0_1, node0_2, node0_3],
    node0_4_6: [node0_1, node0_2, node0_3],
    node0_4_7: [node0_1, node0_2, node0_3],
    node0_4_8: [node0_1, node0_2, node0_3],
}
network0 = BayesianNetwork(nodes0, parents0)

# Sample
num_samples = 10000

cfg = Cfg()
cfg.device = torch_device
sampler = TorchSampler(cfg, network0)
samples = sampler.sample(num_samples, [node0_4_1, node0_4_2, node0_4_3, node0_4_4, node0_4_5, node0_4_6, node0_4_7, node0_4_8])

# Optimize
node1_1 = CPTNode(generate_random_probability_matrix((2)))
node1_2 = CPTNode(generate_random_probability_matrix((2, 2)))
node1_3 = CPTNode(generate_random_probability_matrix((2, 2, 2)))
node1_4_1 = CPTNode(generate_random_probability_matrix((2, 2, 2, 2)))
node1_4_2 = CPTNode(generate_random_probability_matrix((2, 2, 2, 2)))
node1_4_3 = CPTNode(generate_random_probability_matrix((2, 2, 2, 2)))
node1_4_4 = CPTNode(generate_random_probability_matrix((2, 2, 2, 2)))
node1_4_5 = CPTNode(generate_random_probability_matrix((2, 2, 2, 2)))
node1_4_6 = CPTNode(generate_random_probability_matrix((2, 2, 2, 2)))
node1_4_7 = CPTNode(generate_random_probability_matrix((2, 2, 2, 2)))
node1_4_8 = CPTNode(generate_random_probability_matrix((2, 2, 2, 2)))

nodes1 = [node1_1, node1_2, node1_3, node1_4_1, node1_4_2, node1_4_3, node1_4_4, node1_4_5, node1_4_6, node1_4_7, node1_4_8]
parents1 = {
    node1_1: [],
    node1_2: [node1_1],
    node1_3: [node1_1, node1_2],
    node1_4_1: [node1_1, node1_2, node1_3],
    node1_4_2: [node1_1, node1_2, node1_3],
    node1_4_3: [node1_1, node1_2, node1_3],
    node1_4_4: [node1_1, node1_2, node1_3],
    node1_4_5: [node1_1, node1_2, node1_3],
    node1_4_6: [node1_1, node1_2, node1_3],
    node1_4_7: [node1_1, node1_2, node1_3],
    node1_4_8: [node1_1, node1_2, node1_3],
}
network1 = BayesianNetwork(nodes1, parents1)

cfg = Cfg()
cfg.device = torch_device
optimizer = EmOptimizer(
    network1,
    lambda bayesian_network: TorchNaiveInferenceMachine(
        cfg,
        bayesian_network,
        [node1_4_1, node1_4_2, node1_4_3, node1_4_4, node1_4_5, node1_4_6, node1_4_7, node1_4_8]))

evidence = samples

num_iterations = 40

log = Empty()
log.ll = np.zeros((num_iterations))
log.cpt1 = np.zeros((num_iterations, 2, 2, 2, 2))
log.cpt2 = np.zeros((num_iterations, 2, 2, 2, 2))
log.cpt3 = np.zeros((num_iterations, 2, 2, 2, 2))
log.cpt4 = np.zeros((num_iterations, 2, 2, 2, 2))
log.cpt5 = np.zeros((num_iterations, 2, 2, 2, 2))
log.cpt6 = np.zeros((num_iterations, 2, 2, 2, 2))
log.cpt7 = np.zeros((num_iterations, 2, 2, 2, 2))
log.cpt8 = np.zeros((num_iterations, 2, 2, 2, 2))

def callback(ll, iteration):
    log.ll[iteration] = ll
    log.cpt1[iteration] = node1_4_1.cpt.copy()
    log.cpt2[iteration] = node1_4_2.cpt.copy()
    log.cpt3[iteration] = node1_4_3.cpt.copy()
    log.cpt4[iteration] = node1_4_4.cpt.copy()
    log.cpt5[iteration] = node1_4_5.cpt.copy()
    log.cpt6[iteration] = node1_4_6.cpt.copy()
    log.cpt7[iteration] = node1_4_7.cpt.copy()
    log.cpt8[iteration] = node1_4_8.cpt.copy()
    print(f'Finished iteration {iteration}/{num_iterations} - ll: {ll}')

optimizer.optimize(
    evidence,
    num_iterations,
    callback)

plt.figure()
plt.subplot(9, 1, 1); plt.plot(log.ll)
plt.subplot(9, 1, 2); plt.plot(log.cpt1.reshape((num_iterations, -1))); plt.ylim([0, 1])
plt.subplot(9, 1, 3); plt.plot(log.cpt2.reshape((num_iterations, -1))); plt.ylim([0, 1])
plt.subplot(9, 1, 4); plt.plot(log.cpt3.reshape((num_iterations, -1))); plt.ylim([0, 1])
plt.subplot(9, 1, 5); plt.plot(log.cpt4.reshape((num_iterations, -1))); plt.ylim([0, 1])
plt.subplot(9, 1, 6); plt.plot(log.cpt5.reshape((num_iterations, -1))); plt.ylim([0, 1])
plt.subplot(9, 1, 7); plt.plot(log.cpt6.reshape((num_iterations, -1))); plt.ylim([0, 1])
plt.subplot(9, 1, 8); plt.plot(log.cpt7.reshape((num_iterations, -1))); plt.ylim([0, 1])
plt.subplot(9, 1, 9); plt.plot(log.cpt8.reshape((num_iterations, -1))); plt.ylim([0, 1])

pass