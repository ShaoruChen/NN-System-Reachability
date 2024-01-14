import torch
import torch.nn as nn

import os
import sys

script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_directory)

project_dir = os.path.dirname(script_directory)
project_dir = os.path.dirname(project_dir)
sys.path.append(project_dir)

import warnings
warnings.simplefilter("always")
import numpy as np

from pympc.geometry.polyhedron import Polyhedron
import nn_reachability.utilities as ut
import matplotlib.pyplot as plt
from nn_reachability.nn_models import SequentialModel
from nn_reachability.utilities import nn_dyn_approx
import argparse

def Rayleigh_Duffing(x, dt = 0.05):
    y_1 = x[0] + dt*x[1]
    y_2 = x[1] + dt*(-0.1*x[0] - 0.5*x[1] - x[0]**3 - x[1]**3)
    y = np.array([y_1, y_2])
    return y

def nn_dynamics(n=2):
    # neural network dynamics with randomly generated weights
    model = nn.Sequential(
        nn.Linear(n, 50),
        nn.ReLU(),
        nn.Linear(50, 40),
        nn.ReLU(),
        nn.Linear(40, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, n)
    )
    return model

if __name__ == '__main__':
    # compare reachable set over-approximations through the recursive and one-shot methods using linear programming

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    nn_system_file_name = os.path.join(script_directory, 'nn_models', 'duffing_nn_model_0.pt')

    if args.train:
        '''train a neural network to approximate the given dynamics'''
        # domain of the system
        x_min = np.array([-10.0, -10.0])
        x_max = np.array([10.0, 10.0])
        domain = Polyhedron.from_bounds(x_min, x_max)
        nn_model = nn_dynamics(2)
        sampling_param = 50
        nn_system = nn_dyn_approx(Rayleigh_Duffing, nn_model, domain, sampling_param,
                                 batch_size = 10, num_epochs = 400, random_sampling = False, save_model_path = nn_system_file_name)

    # finite step reachability analysis
    torch.set_grad_enabled(False)
    nn_system = nn_dynamics(2)
    nn_system.load_state_dict(torch.load(nn_system_file_name))

    x0 = torch.zeros(2)
    nx = 2

    # inital set is given by A_input x <= b_input
    A_input = np.vstack((np.eye(nx), -np.eye(nx)))
    b_input = np.array([-2.5, -2.5, 3.0, 3.0])

    # select the set of bounding hyperplanes
    c_output = ut.unif_normal_vecs(nx, n = 4)

    horizon = 5

    # recursive reachability analysis
    print('Recursive reachability analysis \n')
    output_bds_list_recur = []
    solver_time_recur_list = []
    A_input_recur, b_input_recur = A_input, b_input
    pre_act_bds_list_recur = []
    for i in range(horizon):
        base_nn_system = SequentialModel(nn_system, 1)
        bounds_list, solver_time_recur = base_nn_system.output_bounds_LP(A_input_recur, b_input_recur, c_output,
                                                                         file_name=None)
        pre_act_bds_list_recur = pre_act_bds_list_recur + bounds_list[:-1]
        output_bds = bounds_list[-1]
        output_bds_list_recur.append(output_bds)
        A_input_recur, b_input_recur = output_bds['A'], output_bds['b']
        solver_time_recur_list.append(solver_time_recur)

    # one-shot reachability analysis
    print('One-shot reachability analysis \n')
    output_bds_list_oneshot = []
    solver_time_oneshot_list = []
    seq_nn_system = SequentialModel(nn_system, 1)
    for i in range(horizon):
        print('step {} \n'.format(i+1))
        seq_nn_system.reset_horizon(i+1)
        bounds_list, solver_time_oneshot = seq_nn_system.output_bounds_LP(A_input, b_input, c_output, file_name=None)
        output_bds = bounds_list[-1]
        output_bds_list_oneshot.append(output_bds)
        solver_time_oneshot_list.append(solver_time_oneshot)

    pre_act_bds_list_oneshot = bounds_list[:-1]

    solver_time_oneshot_list = [sum(item) for item in solver_time_oneshot_list]
    solver_time_recur_list = [sum(item) for item in solver_time_recur_list]

    print(f'solver time comparison: one_shot: {sum(solver_time_oneshot_list)}, recursive: {sum(solver_time_recur_list)}')

    plt.figure()
    solver_time_seq_accumulated_iter = [sum(solver_time_recur_list[:i+1]) for i in range(horizon)]
    plt.semilogy(solver_time_oneshot_list, 'ro-', label = 'one-shot')
    plt.semilogy(solver_time_seq_accumulated_iter,'bs-.', label = 'recursive')
    plt.title('solver time comparison')
    plt.xlabel('step')
    plt.ylabel('solver time [sec]')
    plt.legend()

    # construct polyhedra from the bounds
    oneshot_poly_list = ut.bounds_list_to_polyhedron_list(output_bds_list_oneshot)
    recur_poly_list = ut.bounds_list_to_polyhedron_list(output_bds_list_recur)

    plt.figure()
    domain = Polyhedron(A_input, b_input)
    init_states = ut.uniform_random_sample_from_Polyhedron(domain, 30)
    traj_list = ut.simulate_NN_system(nn_system, init_states, step=horizon - 1)

    ut.plot_multiple_traj_tensor_to_numpy(traj_list, color = 'gray', linewidth = 0.5, alpha = 0.5)
    domain.plot(fill=False, ec='k', linestyle='--', linewidth=2)

    for i in range(horizon - 1):
        oneshot_poly_list[i].plot(fill=False, ec='tab:red', linestyle='-', linewidth=2)
        recur_poly_list[i].plot(fill=False, ec='tab:blue', linestyle='-.', linewidth=2)
    oneshot_poly_list[horizon-1].plot(fill=False, ec='tab:red', linestyle='-', linewidth=2, label='one-shot')
    recur_poly_list[horizon-1].plot(fill=False, ec='tab:blue', linestyle='-.', linewidth=2, label='recursive')

    plt.xlabel(r'$x_1$', fontsize = 18)
    plt.ylabel(r'$x_2$', fontsize = 18)
    plt.grid()
    plt.legend(fontsize = 18)

    # compare layerwise bounds
    ut.compare_layerwise_bounds(pre_act_bds_list_recur, pre_act_bds_list_oneshot)

    plt.show()
