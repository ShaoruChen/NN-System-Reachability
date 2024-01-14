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

import nn_reachability.utilities as ut
import numpy as np
from pympc.geometry.polyhedron import Polyhedron

import matplotlib.pyplot as plt
from nn_reachability.nn_models import SequentialModel
from nn_reachability.nn_models import recursive_output_Lp_bounds_LiRPA, output_Lp_bounds_LiRPA

import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    # load the trained nn model
    nn_system_file_name = os.path.join(script_directory, 'nn_models', 'duffing_nn_model_0.pt')

    nn_system = nn_dynamics(2)
    nn_system.load_state_dict(torch.load(nn_system_file_name))

    x0_lb = torch.tensor([[-3.0, -3.0]]).to(torch.device('cpu'))
    x0_ub = torch.tensor([[-2.5, -2.5]]).to(torch.device('cpu'))

    # one-shot reachability analysis
    print('One-shot reachability analysis \n')

    oneshot_horizon = 8
    method = 'backward'
    box = {'lb': x0_lb[0].numpy(), 'ub': x0_ub[0].numpy()}
    one_shot_bounds_list = [box]
    start_time = time.time()
    for i in range(oneshot_horizon):
        nn_model = SequentialModel(nn_system, i + 1)
        output_lb, output_ub = output_Lp_bounds_LiRPA(nn_model, x0_lb, x0_ub, method=method)
        box = {'lb': output_lb[0].detach().numpy(), 'ub': output_ub[0].detach().numpy()}
        one_shot_bounds_list.append(box)
    one_shot_time = time.time() - start_time

    # recursive reachability analysis
    print('Recursive reachability analysis \n')
    recursive_horizon = 3
    base_nn_system = SequentialModel(nn_system, 1)
    start_time = time.time()
    recursive_bounds_list = recursive_output_Lp_bounds_LiRPA(base_nn_system, x0_lb, x0_ub, recursive_horizon, method=method)
    recursive_time = time.time() - start_time
    recursive_bounds_list = [{'lb': item['lb'][0].detach().numpy(), 'ub': item['ub'][0].detach().numpy()} for item in
                        recursive_bounds_list]

    print(f'one-shot analysis: {one_shot_time} s, recursive analysis: {recursive_time} s.')

    # add IBP baseline
    base_nn_system = SequentialModel(nn_system, 1)
    recursive_bounds_IBP = recursive_output_Lp_bounds_LiRPA(base_nn_system, x0_lb, x0_ub, recursive_horizon, method='IBP')
    recursive_bounds_IBP = [{'lb': item['lb'][0].detach().numpy(), 'ub': item['ub'][0].detach().numpy()} for item in
                       recursive_bounds_IBP]
    recursive_poly_IBP = ut.bounds_list_to_polyhedron_list(recursive_bounds_IBP)

    one_shot_poly_list = ut.bounds_list_to_polyhedron_list(one_shot_bounds_list)
    recursive_poly_list = ut.bounds_list_to_polyhedron_list(recursive_bounds_list)

    # plot reachable set over-approximations
    plt.figure()
    input_set = Polyhedron.from_bounds(x0_lb[0].numpy(), x0_ub[0].numpy())
    init_states = ut.uniform_random_sample_from_Polyhedron(input_set, 30)
    traj_list = ut.simulate_NN_system(nn_system, init_states, step=oneshot_horizon-1)

    ut.plot_multiple_traj_tensor_to_numpy(traj_list, color = 'gray', linewidth = 0.5, alpha = 0.5)
    input_set.plot(fill=False, ec='k', linestyle='--', linewidth=2)

    # plot one-shot
    oneshot_plot_horizon = oneshot_horizon
    for i in range(oneshot_plot_horizon-1):
        one_shot_poly_list[i + 1].plot(fill=False, ec='tab:red', linestyle='-', linewidth=2)
    one_shot_poly_list[oneshot_plot_horizon].plot(fill=False, ec='tab:red', linestyle='-', linewidth=2, label = 'one-shot')

    recursive_plot_horizon = 3
    for i in range(recursive_plot_horizon-1):
        recursive_poly_list[i + 1].plot(fill=False, ec='tab:blue', linestyle='-.', linewidth=2)
    recursive_poly_list[recursive_plot_horizon].plot(fill=False, ec='tab:blue', linestyle='-.', linewidth=2, label = 'recursive')

    # plt.title('Reachable set over-approximations')
    plt.xlabel(r'$x_1$', fontsize = 18)
    plt.ylabel(r'$x_2$', fontsize = 18)
    plt.grid()
    plt.legend(fontsize = 18)

    plt.show()
