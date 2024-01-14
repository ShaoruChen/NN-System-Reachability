import numpy as np
from pympc.geometry.polyhedron import Polyhedron
from pympc.plot import plot_state_space_trajectory

from pympc.optimization.programs import linear_program
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import pickle

import warnings
warnings.simplefilter("always")

def generate_training_data(dyn_fcn, X, sample_param, file_name = None, random_sampling = False):
    # sample the dyn_fcn over the domain X
    # if random_sampling = False, grid sampling is applied and sample_param denotes how many
    # points to sample along each dimension (e.g., sample_param = 5 for a 3D system => 5x5x5 = 125 samples in total)
    # if random_sampling = True, random sampling is applied and sample_param denotes how many samples to generate in total.
    # if filename is not None, load the existing file
    if file_name is not None:
        data = load_pickle_file(file_name)
        input_samples = data['train_data']
        labels = data['label_data']
    else:
        if random_sampling:
            input_samples = uniform_random_sample_from_Polyhedron(X, sample_param)
        else:
            input_samples = grid_sample_from_Polyhedron(X, sample_param)
        labels = sample_vector_field(dyn_fcn, input_samples)
    return input_samples, labels


def sample_vector_field(dyn_fcn, samples):
    num_samples, nx = samples.shape

    sample = samples[0]
    output = dyn_fcn(sample)
    ny = output.shape[0]

    labels = np.zeros((1,ny))
    for i in tqdm(range(num_samples), desc = 'sample_vector_filed'):
    # for i in range(num_samples):
        x_input = samples[i]
        y = dyn_fcn(x_input)
        labels = np.vstack((labels, y))

    labels = labels[1:,:]
    return labels


def uniform_random_sample_from_Polyhedron(X, N):
    # uniformly grid sample from the Polyhedron X with N_dim grid points on each dimension
    nx = X.A.shape[1]
    lb, ub = find_bounding_box(X)
    box = [[lb[i], ub[i]] for i in range(len(lb))]
    box_grid_samples = random_unif_sample_from_box(box, N)
    idx_set = [X.contains(box_grid_samples[i, :]) for i in range(box_grid_samples.shape[0])]
    valid_samples = box_grid_samples[idx_set]
    return valid_samples

def grid_sample_from_Polyhedron(X, N_dim, epsilon=None, residual_dim=None):
    # uniformly grid sample from the Polyhedron X with N_dim grid points on each dimension
    nx = X.A.shape[1]
    if residual_dim is not None:
        X = X.project_to(residual_dim)
    lb, ub = find_bounding_box(X)
    box_grid_samples = grid_sample_from_box(lb, ub, N_dim, epsilon)
    idx_set = [X.contains(box_grid_samples[i, :]) for i in range(box_grid_samples.shape[0])]
    valid_samples = box_grid_samples[idx_set]

    if residual_dim is not None:
        aux_samples = np.zeros((valid_samples.shape[0], 1))
        for i in range(nx):
            if i in residual_dim:
                aux_samples = np.hstack((aux_samples, valid_samples[:, i].reshape(-1, 1)))
            else:
                aux_samples = np.hstack((aux_samples, np.zeros((valid_samples.shape[0], 1))))

        aux_samples = aux_samples[:, 1:]
        return aux_samples

    return valid_samples

# random uniform sample from a box
def random_unif_sample_from_box(bounds_list, N):
    # box_list = [[lb_1, ub_1], [lb_2, ub_2], ..., [lb_n, ub_n]] where lb_i and ub_i denotes the box range in the i-th dim.
    # sample a total of N points randomly from the box described by bounds_list
    box_list = [[item[0], item[1] - item[0]] for item in bounds_list]
    nx = len(box_list)
    rand_matrix = np.random.rand(N, nx)
    samples = np.vstack([rand_matrix[:, i] * box_list[i][1] + box_list[i][0] for i in range(nx)])
    samples = samples.T
    return samples

def find_bounding_box(X):
    # find the smallest box that contains Polyhedron X
    A = X.A
    b = X.b

    nx = A.shape[1]

    lb_sol = [linear_program(np.eye(nx)[i], A, b) for i in range(nx)]
    lb_val = [lb_sol[i]['min'] for i in range(nx)]

    ub_sol = [linear_program(-np.eye(nx)[i], A, b) for i in range(nx)]
    ub_val = [-ub_sol[i]['min'] for i in range(nx)]

    return lb_val, ub_val


def grid_sample_from_box(lb, ub, Ndim, epsilon=None):
    # generate uniform grid samples from a box {lb <= x <= ub} with Ndim samples on each dimension
    nx = len(lb)
    assert nx == len(ub)

    if epsilon is not None:
        lb = [lb[i] + epsilon for i in range(nx)]
        ub = [ub[i] - epsilon for i in range(nx)]

    grid_samples = grid_sample(lb, ub, Ndim, nx)
    return grid_samples


def grid_sample(lb, ub, Ndim, idx):
    # generate samples using recursion
    nx = len(lb)
    cur_idx = nx - idx
    lb_val = lb[cur_idx]
    ub_val = ub[cur_idx]

    if idx == 1:
        cur_samples = np.linspace(lb_val, ub_val, Ndim)
        return cur_samples.reshape(-1, 1)

    samples = grid_sample(lb, ub, Ndim, idx - 1)
    n_samples = samples.shape[0]
    extended_samples = np.tile(samples, (Ndim, 1))

    cur_samples = np.linspace(lb_val, ub_val, Ndim)
    new_column = np.kron(cur_samples.reshape(-1, 1), np.ones((n_samples, 1)))

    new_samples = np.hstack((new_column, extended_samples))
    return new_samples


'''
neural network training
'''

def nn_dyn_approx(system_dynamics, nn_model, domain, sampling_param, batch_size = 10, num_epochs = 100, random_sampling = False, save_model_path = None):
    # train a nn_model that approximates system_dynamics by sampling from a bounded domain
    x_train_samples, y_train_samples = generate_training_data(system_dynamics, domain, sampling_param,
                                                              file_name = None, random_sampling = random_sampling)
    sample_data = {'train_data': x_train_samples, 'label_data': y_train_samples}
    pickle_file(sample_data, 'training_data_set')

    train_data_set = SystemDataSet(x_train_samples, y_train_samples)
    train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)

    if save_model_path is None:
        path = 'torch_nn_model'
    else:
        path = save_model_path

    nn_model = train_nn_and_save(train_loader, nn_model, num_epochs= num_epochs, l1=None,
                              pred_step=0, lr=1e-4, decay_rate=1.0, clr=None, path=path)
    return nn_model

class torch_nn_model(nn.Module):
    def __init__(self, nn_dims):
        super(torch_nn_model, self).__init__()
        self.dims = nn_dims
        self.L = len(nn_dims) - 2
        self.linears = nn.ModuleList([nn.Linear(nn_dims[i], nn_dims[i+1]) for i in range(len(nn_dims)-1)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i in range(self.L):
            x = F.relu(self.linears[i](x))

        x = self.linears[-1](x)

        return x

# custom data set
class SystemDataSet(Dataset):
    def __init__(self, x_samples, y_samples):
        x_samples = torch.from_numpy(x_samples)
        y_samples = torch.from_numpy(y_samples)
        nx = x_samples.size(-1)
        ny = y_samples.size(-1)
        self.nx = nx
        self.ny = ny

        # we sample trajectories
        x_samples, y_samples = x_samples.type(torch.float32), y_samples.type(torch.float32)
        x_samples = x_samples.unsqueeze(1)
        y_samples = y_samples.unsqueeze(1)

        self.x_samples = x_samples
        self.y_samples = y_samples

    def __len__(self):
        return len(self.x_samples)

    def __getitem__(self, index):
        target = self.y_samples[index]
        data_val = self.x_samples[index]
        return data_val, target


def criterion(pred_traj, label_traj):
    batch_size = pred_traj.size(0)
    step = pred_traj.size(1)
    label_step = label_traj.size(1)
    if step > label_step:
        warnings.warn('prediction step mismatch')

    slice_step = min(step, label_step)

    label_traj_slice = label_traj[:, :slice_step, :]
    pred_traj_slice = pred_traj[:, :slice_step, :]

    # label_traj_slice_norm = torch.unsqueeze(torch.linalg.norm(label_traj_slice, 2, dim = 2), 2)
    # label_traj_slice = label_traj_slice/label_traj_slice_norm
    # pred_traj_slice = pred_traj_slice/label_traj_slice_norm
    # err = torch.linalg.norm(label_traj_slice.reshape(-1) - pred_traj_slice.reshape(-1), 2)**2/(batch_size*step)

    # err = 0.0
    # for i in range(batch_size):
    #     err += torch.linalg.norm(label_traj_slice[i].reshape(-1) - pred_traj_slice[i].reshape(-1), np.inf)/(torch.linalg.norm(label_traj_slice[i].reshape(-1), np.inf) + 1e-4)
    #
    # err = err/batch_size
    #
    #
    err = torch.norm(label_traj_slice.reshape(-1) - pred_traj_slice.reshape(-1), 2)/(batch_size*slice_step)

    # err = torch.linalg.norm(label_traj_slice.reshape(-1) - pred_traj_slice.reshape(-1), 2)**2/(pred_traj_slice.reshape(-1).size(0))

    return err

def torch_train_nn(nn_model, dataloader, l1 = None, epochs = 30, step = 5, lr = 1e-4, decay_rate = 1.0, clr = None):

    if clr is None:
        optimizer = optim.Adam(nn_model.parameters(), lr= lr)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay_rate, last_epoch=-1)
        lr_scheduler = lambda t: lr
        cycle = 1
        update_rate = 1
    else:
        lr_base = clr['lr_base']
        lr_max = clr['lr_max']
        step_size = clr['step_size']
        cycle = clr['cycle']
        update_rate = clr['update_rate']
        optimizer = optim.Adam(nn_model.parameters(), lr= lr_max)
        lr_scheduler = lambda t: np.interp([t], [0, step_size, cycle], [lr_base, lr_max, lr_base])[0]

    lr_test = {}
    cycle_loss = 0.0
    cycle_count = 0

    nn_model.train()

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        lr = lr_scheduler((epoch//update_rate)%cycle) # change learning rate every two epochs
        optimizer.param_groups[0].update(lr=lr)

        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # forward + backward + optimize
            batch_size = inputs.size(0)
            x = inputs
            y = nn_model(x)
            traj = y
            for _ in range(step):
                x = y
                y = nn_model(x)
                traj = torch.cat((traj, y), 1)

            loss_1 = criterion(traj, labels)

            # add l1 regularization
            if l1 is not None:
                l1_regularization = 0.0
                for param in nn_model.parameters():
                    '''attention: what's the correct l1 regularization'''
                    l1_regularization += torch.linalg.norm(param.view(-1), 1)
                loss = loss_1 + l1*l1_regularization
            else:
                loss = loss_1

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss_1.item()

            cycle_loss += loss_1.item()

            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.6f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        if (epoch + 1) % update_rate == 0:
            lr_test[cycle_count] = cycle_loss / update_rate / len(dataloader)
            print('\n [%d, %.4f] cycle loss: %.6f' % (cycle_count, lr, lr_test[cycle_count]))

            cycle_count += 1
            cycle_loss = 0.0

        # scheduler.step()
    pickle_file(lr_test, 'lr_test_temp')

    print('finished training')
    save_torch_nn_model_dict(nn_model, 'torch_nn_model_dict_temp')
    return nn_model


def train_nn_and_save(dataloader, nn_model, num_epochs= 30, l1 = None, pred_step = 5, lr = 1e-4, decay_rate = 1.0, clr = None, path = 'torch_nn_model_temp'):
    nn_model = torch_train_nn(nn_model, dataloader, l1 = l1, epochs = num_epochs, step = pred_step, lr = lr, decay_rate = decay_rate, clr = clr)
    save_torch_nn_model_dict(nn_model, path)
    return nn_model

def load_torch_nn_model(nn_model, model_param_name):
    nn_model.load_state_dict(torch.load(model_param_name))
    return nn_model

def save_torch_nn_model_dict(nn_model, path):
    torch.save(nn_model.state_dict(), path)

'''
simulate closed-loop system with NN components.
NN structure and data are from pytorch.
'''

def load_pickle_file(file_name):
    with open(file_name, 'rb') as config_dictionary_file:
        data = pickle.load(config_dictionary_file)
    return data

def pickle_file(data, file_name):
    with open(file_name, 'wb') as config_dictionary_file:
          pickle.dump(data, config_dictionary_file)

def generate_training_data_traj(dyn_fcn, X, N_dim, step = 0):
    init_states = grid_sample_from_Polyhedron(X, N_dim)
    x_samples, y_samples = generate_traj_samples(dyn_fcn, init_states, step)
    return x_samples, y_samples


def generate_traj_samples(dyn_fcn, init_states, step = 0):
    N = init_states.shape[0]
    nx = init_states.shape[1]

    traj_list = []
    for i in range(N):
        x = init_states[i]
        traj = x
        for j in range(step+1):
            x_next = dyn_fcn(x)
            traj = np.vstack((traj, x_next))
            x = x_next
        traj_list.append(traj)
    x_samples = [traj[0:1, :] for traj in traj_list]
    y_samples = [traj[1:, :] for traj in traj_list]

    x_samples = np.stack(x_samples, axis = 0)
    y_samples = np.stack(y_samples, axis = 0)

    return x_samples, y_samples


def plot_multiple_traj(x_traj_list, **kwargs):
    num_traj = len(x_traj_list)
    for i in range(num_traj):
        plot_state_space_trajectory(x_traj_list[i], **kwargs)

def simulate_NN_system(nn_model, init_states, step = 0):
    # init_states is a list of initial conditions in numpy

    N = init_states.shape[0]
    nx = init_states.shape[1]

    traj_list = []
    for i in range(N):
        x = torch.from_numpy(init_states[i])
        x = x.type(torch.float32)
        # x = x.type(nn_model[0].weight.dtype)

        traj = x.unsqueeze(0)
        for j in range(step+1):
            x_next = nn_model(x)
            traj = torch.cat((traj, x_next.unsqueeze(0)))
            x = x_next
        traj_list.append(traj)
    return traj_list

def plot_multiple_traj_tensor_to_numpy(traj_list, **kwargs):
    num_traj = len(traj_list)
    for i in range(num_traj):
        plot_state_space_trajectory(traj_list[i].detach().numpy(), **kwargs)


def bounds_list_to_polyhedron_list(bounds_list):
    N = len(bounds_list)
    poly_list = []

    if 'A' in bounds_list[0].keys():
        for i in range(N):
            X = Polyhedron(bounds_list[i]['A'], bounds_list[i]['b'])
            poly_list.append(X)
    elif 'lb' in bounds_list[0].keys():
        for i in range(N):
            X = Polyhedron.from_bounds(bounds_list[i]['lb'], bounds_list[i]['ub'])
            poly_list.append(X)

    return poly_list

def plot_poly_list(poly_list, **kwargs):
    N = len(poly_list)
    for i in range(N):
        poly_list[i].plot(**kwargs)

def unif_normal_vecs(nx, n = 4):
    # sample n uniformly rotated normal vectors in the 2D plane
    # we only consider nx = 2
    assert nx == 2
    theta = 2*np.pi/n
    rotation_mat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    c = np.zeros((1,nx))
    c[0][0] = 1.0

    vec = c.flatten()
    for i in range(n-1):
        vec = rotation_mat@vec
        c = np.vstack((c, vec.reshape(1,-1)))
    return c


def compare_layerwise_bounds(pre_act_bds_list_iter, pre_act_bds_list_seq):
    # analyze the gap between pre-activation bounds
    lb_list_iter = []
    ub_list_iter = []
    lb_list_seq = []
    ub_list_seq = []
    for i in range(len(pre_act_bds_list_seq)):
        lb_list_iter = lb_list_iter + pre_act_bds_list_iter[i]['lb'].tolist()
        ub_list_iter = ub_list_iter + pre_act_bds_list_iter[i]['ub'].tolist()
        lb_list_seq = lb_list_seq + pre_act_bds_list_seq[i]['lb'].tolist()
        ub_list_seq = ub_list_seq + pre_act_bds_list_seq[i]['ub'].tolist()

    lb_index = sorted(range(len(lb_list_seq)), key=lambda k: lb_list_seq[k])
    ub_index = sorted(range(len(ub_list_seq)), key=lambda k: ub_list_seq[k])

    lbs_iter_sorted = [lb_list_iter[k] for k in lb_index]
    ubs_iter_sorted = [ub_list_iter[k] for k in ub_index]
    lbs_seq_sorted = [lb_list_seq[k] for k in lb_index]
    ubs_seq_sorted = [ub_list_seq[k] for k in ub_index]

    plt.figure()
    plt.plot(lbs_iter_sorted, 'b-.', label='lbs recursive')
    plt.plot(ubs_iter_sorted, 'b-', label='ubs recursive')
    plt.plot(lbs_seq_sorted, 'r-.', label='lbs one-shot')
    plt.plot(ubs_seq_sorted, 'r-', label='ubs one-shot')
    plt.legend()
    plt.ylabel(r'pre-activation bounds')
    # truncation = 50
    # plt.figure()
    # plt.semilogy(lbs_iter_sorted[-truncation:], 'b-.', label='lbs recursive')
    # plt.semilogy(lbs_seq_sorted[-truncation:], 'r-.', label='lbs one-shot')
    #
    # plt.figure()
    # plt.semilogy(ubs_iter_sorted[-truncation:], 'b-', label='ubs recursive')
    # plt.semilogy(ubs_seq_sorted[-truncation:], 'r-', label='ubs one-shot')

