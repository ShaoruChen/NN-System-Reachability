import torch
import torch.nn as nn

import cvxpy as cp
import numpy as np
import copy
import time

from pympc.geometry.polyhedron import Polyhedron
import warnings
warnings.simplefilter("always")

import sys
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from tqdm import tqdm

class InterConnectedModel(nn.Module):
    def __init__(self, A, B, res_dyn, controller, linear_dyn_layer, linear_layers, N = 1):
        # linear_dyn_layer is a nn.Linear that computes Ax + Bu
        # linear_layers include nn.Linear layers that extract variables [x_0; w_0; w_1; ... ; w_N-1] from
        # the concatenated inputs

        super(InterConnectedModel, self).__init__()
        self.nx = A.size(0)
        self.nu = B.size(1)
        self.A = A
        self.B = B
        self.res_dyn = res_dyn
        self.controller = controller
        # size of norm bounded additive disturbances
        # rolling horizon
        self.N = N
        self.linear_layers = linear_layers
        self.linear_dyn_layer = linear_dyn_layer

    def forward(self, input):
        # input = [[x_0; w_0; w_1; ... ; w_N-1]]
        N = self.N

        # extract starting state
        layer = self.linear_layers[0]
        x = layer(input)

        for i in range(N):
            layer = self.linear_layers[i + 1]
            w = layer(input)
            nn_input = self.controller(x)
            x_next = self.linear_dyn_layer(torch.cat((x, nn_input), axis = 1)) + self.res_dyn(torch.cat((x, nn_input), axis = 1)) + w
            x = x_next
        return x

def LiRPA_bound_NNDS(A, B, res_net, control_net, N, x_lb, x_ub, w_lb, w_ub, method = 'backward'):
    nx = A.size(0)
    nu = B.size(1)

    # construct linear dynamics layer
    linear_dyn_layer = nn.Linear(nx + nu, nx)
    linear_dyn_layer.weight.data = torch.cat((A, B), axis=1)
    linear_dyn_layer.bias.data = torch.zeros(nx)

    # construct linear layers that extract x and w variables
    linear_layers = []
    for i in range(N + 1):
        linear_layers.append(nn.Linear((N + 1) * nx, nx))
    linear_layers = nn.ModuleList(linear_layers)

    ind_mat = torch.kron(torch.eye(N + 1), torch.eye(nx))
    for i in range(N + 1):
        linear_layers[i].weight.data = ind_mat[i * nx:(i + 1) * nx, :]
        linear_layers[i].bias.data = torch.zeros(nx)

    cl_system = InterConnectedModel(A, B, res_net, control_net, linear_dyn_layer= linear_dyn_layer,
                                                        linear_layers=linear_layers,  N = N)

    input_lb = torch.cat((x_lb, w_lb.repeat(1, N)), axis=1)
    input_ub = torch.cat((x_ub, w_ub.repeat(1, N)), axis=1)

    output_lb, output_ub = output_Lp_bounds_LiRPA(cl_system, input_lb, input_ub, method= method)
    return output_lb, output_ub


class SequentialModel(nn.Module):
    # SequentialModel is a wrapper of NN dynamics rolled out for a finite horizon
    def __init__(self, nn_model, N):
        # nn_model: torch.nn.Sequential, represents a feedforward NN
        # N: horizon
        super(SequentialModel, self).__init__()
        self.base_model = nn_model
        self.horizon = N
        num_hidden_layers = find_number_of_hidden_layers(nn_model)
        self.num_activation_layers = num_hidden_layers*N
        self.base_num_activation_layers = num_hidden_layers

        self.layer_list = list(nn_model)*N
        self.pre_act_bounds_list = []

    def reset_horizon(self, N):
        self.horizon = N
        self.num_activation_layers = self.base_num_activation_layers*N
        self.layer_list = list(self.base_model) * N

    def forward(self, x):
        for i in range(self.horizon):
            x = self.base_model(x)
        return x

    def pre_activation_bounds_LP(self, bounds_list, layer_num, A, b, c = None):
        # Find pre-activation bounds on the layer_num-th activation layer with a polytopic input set Ax <= b.
        # c = None: finds entrywise lower and upper bounds on the pre-activation variable y.
        #     otherwise, we compute a polyhedral over-approximation cy <= d of the pre-activation variable y.
        # The index layer_num starts from 1.
        # When layer_num is greater than the total number of activation functions in the NN, this function
        # returns over-approximations on the output of the NN.

        assert len(bounds_list) >= layer_num - 1

        x = {}
        y = {}

        x0 = torch.zeros(A.shape[1])

        act_layer_count = 0
        layer_count = 0
        for layer in self.layer_list:
            if isinstance(layer, nn.ReLU):
                act_layer_count += 1
                if act_layer_count >= layer_num:
                    break
            dim_input = x0.shape[0]
            x0 = layer(x0)
            dim_output = x0.shape[0]

            x[layer_count] = cp.Variable(dim_input)
            y[layer_count] = cp.Variable(dim_output)
            layer_count += 1

        constr = [A@x[0] <= b]

        act_layer_count = 0
        layer_count = 0
        for layer in self.layer_list:
            if isinstance(layer, nn.ReLU):
                act_layer_count += 1
                # detect termination
                if act_layer_count >= layer_num:
                    break

                bound = bounds_list[act_layer_count-1]

                lb = bound['lb']
                ub = bound['ub']
                # triangle relaxation of ReLU
                constr += [y[layer_count] >= x[layer_count]]
                constr += [y[layer_count] >= 0]
                constr += [y[layer_count][k] == x[layer_count][k] for k in range(x[layer_count].shape[0]) if lb[k] >= 0]
                constr += [y[layer_count][k] == 0 for k in range(x[layer_count].shape[0]) if ub[k] < 0]
                constr += [y[layer_count][k] <= ub[k]/(ub[k]-lb[k])*(x[layer_count][k] - lb[k]) for k in range(x[layer_count].shape[0])
                           if (ub[k] >0 and lb[k] < 0)]

            if isinstance(layer, nn.Linear):
                weight = layer.weight.detach().numpy()
                bias = layer.bias.detach().numpy()
                constr += [ y[layer_count] == weight @ x[layer_count] + bias ]

            if layer_count > 0:
                constr += [ x[layer_count] == y[layer_count-1] ]

            layer_count += 1

        dim_output = y[layer_count-1].shape[0]
        c_vec = cp.Parameter(dim_output)
        obj = c_vec @ y[layer_count - 1]
        prob = cp.Problem(cp.Minimize(obj), constr)

        print('CVXPY model solving ...')
        total_solver_time = 0
        running_start = time.time()
        if c is None:
            id_mat = np.eye(dim_output)

            lb_vec = np.zeros(dim_output)
            for i in tqdm(range(dim_output), desc='output_lb'):
            # for i in range(dim_output):
                obj_vec = id_mat[i]
                c_vec.value = obj_vec
                prob.solve(solver=cp.GUROBI, verbose=False)
                total_solver_time += prob.solver_stats.solve_time
                lb_vec[i] = obj.value

            ub_vec = np.zeros(dim_output)
            for i in tqdm(range(dim_output), desc='output_ub'):
            # for i in range(dim_output):
                obj_vec = -id_mat[i]
                c_vec.value = obj_vec
                prob.solve(solver=cp.GUROBI, verbose=False)
                total_solver_time += prob.solver_stats.solve_time
                ub_vec[i] = -obj.value
            running_time = time.time() - running_start

            output_bd = {'lb': lb_vec, 'ub': ub_vec}
            diags = {'total_solver_time': total_solver_time, 'dim_output': dim_output,
                     'running_time': running_time}
            return output_bd, diags
        else:
            # compute the polytopic overapproximation given by c
            num_output_constr = c.shape[0]
            output_vec = np.zeros(num_output_constr)
            for i in range(num_output_constr):
                # note that we flip the sign of objective vector to make it a maximization problem
                c_vec.value = -c[i]
                prob.solve(solver=cp.GUROBI, verbose=False)
                output_vec[i] = -obj.value
                solver_time = prob.solver_stats.solve_time
                total_solver_time += solver_time

            output_bd = {'A': c, 'b': output_vec}
            running_time = time.time() - running_start
            diags = {'total_solver_time': total_solver_time, 'dim_output': dim_output,
                 'running_time': running_time}
            return output_bd, diags

    def output_bounds_LP(self, A, b, c = None, file_name = None):
        # compute the pre-activation bounds through LP sequentially and then compute the bound on the NN output
        # input set to NN: Ax <= b
        # bound on output of NN: cy <= d

        if file_name is None:
            file_name = 'layerwise_LP_bds'

        total_num_activation_layers = self.num_activation_layers

        diags_list = []
        time_sum = 0
        solver_time_list = []

        if self.horizon > 1:
            # reuse pre-activation bounds computed from previous steps if horizon > 1
            pre_act_bds = self.pre_act_bounds_list
        else:
            pre_act_bds = []

        # number of activation layers whose pre-activation bounds are already available
        num_existing_layers = len(pre_act_bds)

        for i in range(num_existing_layers, total_num_activation_layers):
            print('activation layer number {}'.format(i))
            # compute pre-activation bounds
            layer_bd, diags = self.pre_activation_bounds_LP(pre_act_bds, i+1, A, b)
            pre_act_bds.append(layer_bd)
            diags_list.append(diags)
            time_sum += diags['total_solver_time']
            solver_time_list.append(diags['total_solver_time'])

        self.pre_act_bounds_list = pre_act_bds

        bounds_list = copy.copy(pre_act_bds)
        # compute bounds on the NN output
        print('output layer')
        layer_bd, diags = self.pre_activation_bounds_LP(bounds_list, total_num_activation_layers + 1, A, b, c = c)
        bounds_list.append(layer_bd)
        diags_list.append(diags)
        time_sum += diags['total_solver_time']
        solver_time_list.append(diags['total_solver_time'])

        data_to_save = {'pre_act_bds': bounds_list, 'diags': diags_list, 'solver_time': time_sum}
        torch.save(data_to_save, file_name + '.pt')
        return bounds_list, solver_time_list

# LP-based one-shot or recursive reachability analysis of a feedforward NNDS nn_system
# Only box bounds are considered so far
# The LP-based verification problem is solved by gurobi or ADMM
def Gurobi_reachable_set(nn_system, x0_lb, x0_ub, horizon, method = 'one_shot'):
    # LP-based one-shot and recursive reachability analysis of nn_system

    nx = nn_system.nx

    # convert input set to Ax  <= b
    A_input = np.vstack((np.eye(nx), -np.eye(nx)))

    input_lb = x0_lb.to(torch.device('cpu')).numpy()
    input_ub = x0_ub.to(torch.device('cpu')).numpy()

    b_input = np.concatenate((input_ub, -input_lb)).flatten()

    if method == 'one_shot':
        print('One-shot reachability analysis \n')
        output_bds_list_seq = []
        solver_time_seq_list = []
        seq_nn_system = SequentialModel(nn_system, 1)
        for i in range(horizon):
            print('step {} \n'.format(i + 1))
            seq_nn_system.reset_horizon(i + 1)
            bounds_list, solver_time_seq = seq_nn_system.output_bounds_LP(A_input, b_input, None, file_name=None)
            output_bds = bounds_list[-1]
            output_bds_list_seq.append(output_bds)
            solver_time_seq_list.append(solver_time_seq)

        pre_act_bds_list_seq = bounds_list[:-1]
        result = {'output_bds': output_bds_list_seq, 'pre_act_bds': pre_act_bds_list_seq, 'solver_time': solver_time_seq_list, 'method':method}

    elif method == 'recursive':
        print('Recursive reachability analysis \n')
        output_bds_list_iter = []
        solver_time_iter_list = []
        A_input_iter, b_input_iter = A_input, b_input
        pre_act_bds_list_iter = []
        for i in range(horizon):
            base_nn_system = SequentialModel(nn_system, 1)
            bounds_list, solver_time_iter = base_nn_system.output_bounds_LP(A_input_iter, b_input_iter, None,
                                                                                file_name=None)
            pre_act_bds_list_iter = pre_act_bds_list_iter + bounds_list[:-1]
            output_bds = bounds_list[-1]
            output_bds_list_iter.append(output_bds)
            output_lb, output_ub = output_bds['lb'], output_bds['ub']
            output_box = Polyhedron.from_bounds(output_lb, output_ub)
            A_input_iter, b_input_iter = output_box.A, output_box.b

            solver_time_iter_list.append(solver_time_iter)
        result = {'output_bds': output_bds_list_iter, 'pre_act_bds': pre_act_bds_list_iter,
                  'solver_time': solver_time_iter_list, 'method': method}
    else:
        raise NotImplementedError
    return output_bds, result

# LiRPA based analysis: only box bounds are considered
def output_Lp_bounds_LiRPA(nn_model, lb, ub, method = 'backward'):

    center = (lb + ub)/2
    radius = (ub - lb)/2

    model = BoundedModule(nn_model, center)
    ptb = PerturbationLpNorm(norm=np.inf, eps=radius)
    # Make the input a BoundedTensor with perturbation
    my_input = BoundedTensor(center, ptb)
    # Regular forward propagation using BoundedTensor works as usual.
    prediction = model(my_input)
    # Compute LiRPA bounds
    output_lb, output_ub = model.compute_bounds(x=(my_input,), method=method)
    return output_lb, output_ub


def recursive_output_Lp_bounds_LiRPA(nn_model, lb_0, ub_0, horizon, method = 'backward'):
    input_lb, input_ub = lb_0, ub_0
    box = {'lb': lb_0, 'ub': ub_0}
    bounds_list =[box]
    ori_model = nn_model
    for i in range(horizon):
        output_lb, output_ub = output_Lp_bounds_LiRPA(ori_model, input_lb, input_ub, method = method)
        box = {'lb': output_lb, 'ub': output_ub}
        bounds_list.append(box)
        input_lb, input_ub = output_lb.detach(), output_ub.detach()

    return bounds_list

def preactivation_bounds_of_sequential_nn_model_LiRPA(nn_model, horizon, lb_0, ub_0, method = 'backward'):
    # compute pre-activation bounds from LiRPA
    nn_layer_list = list(nn_model)*horizon
    num_act_layers = find_number_of_hidden_layers(nn_model)
    pre_act_bds = preactivation_bounds_from_LiRPA(nn_layer_list, lb_0, ub_0, method)
    return pre_act_bds, num_act_layers, horizon

def preactivation_bounds_from_LiRPA(layer_list, lb_0, ub_0, method = 'backward'):
    # find the index of activation layers
    ind_list = []
    for i in range(len(layer_list)):
        if isinstance(layer_list[i], nn.ReLU):
            ind_list.append(i)

    pre_act_bds = []
    for index in ind_list:
        net = nn.Sequential(*layer_list[:index])
        output_lb, output_ub = output_Lp_bounds_LiRPA(net, lb_0, ub_0, method = method)
        bds = {'lb': output_lb, 'ub': output_ub}
        pre_act_bds.append(bds)

    return pre_act_bds

def pre_act_bds_tensor_to_numpy(pre_act_bds):
    # The tensors are of size [1, n] in pre_act_bds
    pre_act_bds_numpy = [{'lb':item['lb'][0].numpy(), 'ub': item['ub'][0].numpy()} for item in pre_act_bds]
    return pre_act_bds_numpy


# def repeat_nn_model(nn_model, N):
#     # repeat nn_model N times and concatenate them in one nn model
#     nn_model_layers = list(nn_model)*N
#     seq_nn_model = nn.Sequential(*(nn_model_layers))
#     return seq_nn_model

def extract_linear_layers(nn_model):
    #extract the linear layers of a FC NN
    linear_layer_list = []
    for i, layer in enumerate(nn_model):
        if isinstance(layer, nn.Linear):
            linear_layer_list.append(layer)
    return linear_layer_list


def find_number_of_hidden_layers(nn_model):
    # find the number of hidden layers in a FC NN
    H = 0
    for i, layer in enumerate(nn_model):
        if isinstance(layer, nn.ReLU):
            H += 1
    return H

def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)


