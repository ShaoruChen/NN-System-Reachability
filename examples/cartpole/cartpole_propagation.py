import os
import sys

script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_directory)

project_dir = os.path.dirname(script_directory)
project_dir = os.path.dirname(project_dir)
sys.path.append(project_dir)

import numpy as np
import torch
from torch.autograd.functional import jacobian
from pympc.geometry.polyhedron import Polyhedron
import torch.nn as nn
from nn_reachability.utilities import nn_dyn_approx
from scipy.io import savemat, loadmat
from torch.utils.data import Dataset, DataLoader
from nn_reachability.utilities import SystemDataSet, train_nn_and_save, bounds_list_to_polyhedron_list
from nn_reachability.nn_models import LiRPA_bound_NNDS
import matplotlib.pyplot as plt
import nn_reachability.utilities as ut
import time
import argparse



class CartPole:
    def __init__(self, mp, mc, len, dt = 0.1):
        self.mp = mp
        self.mc = mc
        self.g = 9.81
        self.len = len

        nx, nu = 4, 1
        self.nx, self.nu = nx, nu

        # state = [p, pdot, theta, thetadot]
        self.state = np.zeros(nx)
        self.u = np.zeros(1)
        self.dt = dt

        self.Ad = None
        self.Bd = None


    def vector_field(self, state_u):
        p, pdot, theta, thetadot, u  = state_u[0], state_u[1], state_u[2], state_u[3], state_u[4]

        vec = np.zeros(self.nx)
        vec[0] = pdot
        vec[2] = thetadot

        denom = self.len*(4/3 - self.mp*np.cos(theta)**2/(self.mc + self.mp))
        thetaddot = (self.g*np.sin(theta) + np.cos(theta)*(-u-self.mp*self.len*thetadot**2*np.sin(theta))/(self.mp + self.mc))/denom

        vec[1] = (u + self.mp*self.len*(thetadot**2*np.sin(theta) - thetaddot*np.cos(theta)))/(self.mp + self.mc)
        vec[3] = thetaddot

        return vec

    def vector_field_torch(self, state, u):
        # rewrite the cartpole dynamics in torch
        p, pdot, theta, thetadot = state[0], state[1], state[2], state[3]

        vec = torch.zeros(self.nx)
        vec[0] = pdot
        vec[2] = thetadot

        denom = self.len * (4 / 3 - self.mp * torch.cos(theta) ** 2 / (self.mc + self.mp))
        thetaddot = (self.g * torch.sin(theta) + torch.cos(theta) * (-u - self.mp * self.len * thetadot ** 2 * torch.sin(theta)) / (
                    self.mp + self.mc)) / denom

        vec[1] = (u + self.mp * self.len * (thetadot ** 2 * torch.sin(theta) - thetaddot * torch.cos(theta))) / (
                    self.mp + self.mc)
        vec[3] = thetaddot
        return vec

    def discrete_time_linear_dyn(self):
        state = torch.tensor([0.0, 0.0, 0.0, 0.0])
        u = torch.tensor([0.0])
        Ac, Bc = jacobian(self.vector_field_torch, (state, u))

        dt = self.dt
        Ac, Bc = Ac.numpy(), Bc.numpy()
        Id = np.eye(nx)
        Ad = Id + Ac*dt
        Bd = Bc*dt

        self.Ad, self.Bd = Ad, Bd
        return Ad, Bd

    def discrete_dyn(self, state_u):
        x = state_u[:self.nx]
        u = state_u[self.nx:]
        x_next = x + self.vector_field(state_u)*self.dt
        return x_next

    def res_dyn_labels(self, state_u):
        x_next = self.discrete_dyn(state_u)
        x = state_u[:self.nx]
        u = state_u[self.nx:]
        if self.Ad is None or self.Bd is None:
            self.discrete_time_linear_dyn()

        Ad, Bd = self.Ad, self.Bd
        res = x_next - x@Ad.T - u@Bd.T
        return res

class CartPoleNNSystem(nn.Module):
    # CartPoleNNSystem represents an uncertain NN approximation of the cartpole dynamics
    # The closed-loop system is given by x_+ = Ax + Bu + res(x, u) + w where res(x, u) is a NN residual dynamics
    # and the controller is given by u = pi(x)

    def __init__(self, A, B, res_dyn, controller, eps_w = None):
        super(CartPoleNNSystem, self).__init__()
        self.A, self.B = A, B
        self.res_dyn = res_dyn
        self.controller = controller
        self.eps_w = eps_w
        self.nx = 4

    def forward(self, x):
        if x.dim() ==1:
            x = x.unsqueeze(0)

        if self.eps_w is not None:
            w = (2*torch.rand(x.size())-1)*self.eps_w
        else:
            w = torch.zeros(x.size())

        u = self.controller(x)
        res = self.res_dyn(torch.cat((x, u), axis = 1))
        x_next = x@self.A.T + u@self.B.T + res + w
        x_next = x_next.squeeze(0)
        return x_next

def cartpole_res_dyn():
    # neural network dynamics with randomly generated weights
    model = nn.Sequential(
        nn.Linear(5, 50),
        nn.Tanh(),
        nn.Linear(50, 4)
    )
    return model

def nn_controller():
    model = nn.Sequential(
        nn.Linear(4, 100),
        nn.ReLU(),
        nn.Linear(100, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    return model

if __name__ == "__main__":
    # compare reachable set over-approximations through the recursive and one-shot methods using backward bound propagation
    # the uncertain closed-loop dynamics of the form x_+ = Ax + Bu + res(x, u) + w where res(x, u) is a NN residual dynamics is considered

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_residual_dyn', action='store_true')
    parser.add_argument('--train_controller', action='store_true')
    args = parser.parse_args()

    mp, mc, len = 0.1, 0.25, 0.2
    dt = 0.05
    nx = 4
    cartpole = CartPole(mp, mc, len, dt = dt)
    cartpole.discrete_time_linear_dyn()
    Ad, Bd = cartpole.Ad, cartpole.Bd

    nn_system_file_path = os.path.join(script_directory, 'cartpole_res.pt')
    controller_path = os.path.join(script_directory, 'cartpole_nn_controller.pt')

    if args.train_residual_dyn:
        '''train a neural network to approximate the given dynamics'''
        # domain of the system
        x_u_min = np.array([-5.0, -5.0, -np.pi, -2*np.pi, -4.0])
        x_u_max = np.array([5.0, 5.0, np.pi, 2*np.pi, 4.0])
        domain = Polyhedron.from_bounds(x_u_min, x_u_max)
        nn_model = cartpole_res_dyn()
        sampling_param = 10000
        res_nn = nn_dyn_approx(cartpole.res_dyn_labels, nn_model, domain, sampling_param,
                                 batch_size = 30, num_epochs = 200, random_sampling = True, save_model_path = nn_system_file_path)

    res_model = cartpole_res_dyn()
    res_model.load_state_dict(torch.load(nn_system_file_path))

    # train a nn controller from imitation learning
    if args.train_controller:
        controller = nn_controller()
        # labels from the matlab data of nonlinear MPC
        X_train = loadmat('X_train_final_temp.mat')
        y_train = loadmat('y_train_final_temp.mat')

        X_train = X_train['X_train_nnmpc_temp']
        y_train = y_train['y_train_nnmpc_temp']

        X_train = X_train.astype(np.double)
        y_train = y_train.astype(np.double)

        batch_size = 30
        num_epochs = 200

        train_data_set = SystemDataSet(X_train, y_train)
        train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
        controller_nn = train_nn_and_save(train_loader, controller, num_epochs=num_epochs, l1=None,
                                     pred_step=0, lr=1e-4, decay_rate=1.0, clr=None, path=controller_path)

    # connect the nn
    controller = nn_controller()
    controller.load_state_dict(torch.load(controller_path))

    nx = 4
    nu = 1

    # cartpole closed loop system
    # range of cartpole states are bounded by [10; pi; 10; 10]
    A, B = torch.from_numpy(Ad), torch.from_numpy(Bd)
    A, B = A.to(torch.float32), B.to(torch.float32)
    eps_w = torch.tensor([[0.02, 0.02, 0.02, 0.02]])

    x_0_lb = torch.tensor([[0.0,  -0.2, np.pi/12, -0.15]])
    x_0_ub = torch.tensor([[0.3,  -0.1, np.pi/12 + 0.1, -0.05]])

    # define the disturbance range
    w_lb = -eps_w
    w_ub = eps_w

    # compare one-shot and recursive reachability analysis frameworks
    # one-shot reachability analysis
    horizon = 3
    start_time = time.time()
    oneshot_bounds_list = []
    for i in range(horizon):
        N = i + 1
        output_lb, output_ub = LiRPA_bound_NNDS(A, B, res_model, controller, N, x_0_lb, x_0_ub, w_lb, w_ub)
        bds = {'lb': output_lb, 'ub': output_ub}
        oneshot_bounds_list.append(bds)
    oneshot_runtime = time.time() - start_time

    # recursive reachability analysis
    recursive_bounds_list = []
    x_lb, x_ub = x_0_lb, x_0_ub
    start_time = time.time()
    for i in range(horizon):
        N = i + 1
        output_lb, output_ub = LiRPA_bound_NNDS(A, B, res_model, controller, 1, x_lb, x_ub, w_lb, w_ub)
        bds = {'lb': output_lb, 'ub': output_ub}
        recursive_bounds_list.append(bds)

        x_lb = output_lb
        x_ub = output_ub
    recursive_runtime = time.time() - start_time

    print(f'runtime of the one-shot metho: {oneshot_runtime}, runtime of the recursive method: {recursive_runtime}')

    # transfer bounds to numpy list
    oneshot_bounds_list_numpy = [{'lb': item['lb'].squeeze(0).detach().numpy(), 'ub': item['ub'].squeeze(0).detach().numpy()} for item in oneshot_bounds_list]
    recursive_bounds_list_numpy = [{'lb': item['lb'].squeeze(0).detach().numpy(), 'ub': item['ub'].squeeze(0).detach().numpy()} for item in recursive_bounds_list]

    oneshot_poly_list = bounds_list_to_polyhedron_list(oneshot_bounds_list_numpy)
    recur_poly_list = bounds_list_to_polyhedron_list(recursive_bounds_list_numpy)

    # plot the reachable set over-approximations
    # view the projections onto specified dimensions
    for plot_dim in [[0,1],[2,3]]:
        plt.figure()
        cartpole_nn = CartPoleNNSystem(A, B, res_model, controller, eps_w=eps_w.squeeze(0))
        domain = Polyhedron.from_bounds(x_0_lb.squeeze(0).detach().numpy(), x_0_ub.squeeze(0).detach().numpy())
        init_states = ut.uniform_random_sample_from_Polyhedron(domain, 50)
        traj_list = ut.simulate_NN_system(cartpole_nn.forward, init_states, step=horizon - 1)

        ut.plot_multiple_traj_tensor_to_numpy(traj_list, dim=plot_dim, color='gray', linewidth=0.5, alpha = 0.5)
        domain.plot(residual_dimensions=plot_dim, fill=False, ec='k', linestyle='--', linewidth=2)

        # plot bounding boxes
        ut.plot_poly_list(oneshot_poly_list[:-1], residual_dimensions =plot_dim, fill=False, ec='tab:red', linestyle='-', linewidth=2)
        ut.plot_poly_list(recur_poly_list[:-1], residual_dimensions =plot_dim, fill=False, ec='tab:blue', linestyle='-.', linewidth=2)

        oneshot_poly_list[-1].plot(residual_dimensions =plot_dim, fill=False, ec='tab:red', linestyle='-', linewidth=2, label = 'one-shot')
        recur_poly_list[-1].plot(residual_dimensions =plot_dim, fill=False, ec='tab:blue', linestyle='-.', linewidth=2, label = 'recurisve')

        plt.xlabel(f'$x_{plot_dim[0]+1}$', fontsize = 18)
        plt.ylabel(f'$x_{plot_dim[1]+1}$', fontsize = 18)

        plt.show()


