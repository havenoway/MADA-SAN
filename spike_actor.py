import torch
import torch.nn as nn
import math
import random


"""
Parameters for SNN
"""

ENCODER_REGULAR_VTH = 0.999
NEURON_VTH = 0.5
NEURON_CDECAY = 1 / 2
NEURON_VDECAY = 3 / 4
SPIKE_PSEUDO_GRAD_WINDOW = 0.5

"""
Parameters for C-LIF
"""
dt = 3e-3
tau_m = 30e-3
tau_s = tau_m / 4.
beta = tau_m / tau_s
V0_param = (1 / (beta - 1)) * (beta ** (beta / (beta - 1)))

decay_s = math.exp(-(dt / tau_s))  # decay constants，~0.9
decay_m = math.exp(-(dt / tau_m))  # ~0.6.



class PseudoEncoderSpikeRegular(torch.autograd.Function):
    """ Pseudo-gradient function for spike - Regular Spike for encoder """
    @staticmethod
    def forward(ctx, input):
        return input.gt(ENCODER_REGULAR_VTH).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class PseudoEncoderSpikePoisson(torch.autograd.Function):
    """ Pseudo-gradient function for spike - Poisson Spike for encoder """
    @staticmethod
    def forward(ctx, input):
        return torch.bernoulli(input).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class PopSpikeEncoderRegularSpike(nn.Module):
    """ Learnable Population Coding Spike Encoder with Regular Spike Trains """

    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
        """
        :param obs_dim: observation dimension
        :param pop_dim: population dimension
        :param spike_ts: spike timesteps
        :param mean_range: mean range
        :param std: std
        :param device: device
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.pop_dim = pop_dim
        self.encoder_neuron_num = obs_dim * pop_dim
        self.spike_ts = spike_ts
        self.device = device
        self.pseudo_spike = PseudoEncoderSpikeRegular.apply
        # Compute evenly distributed mean and variance
        tmp_mean = torch.zeros(1, obs_dim, pop_dim)
        delta_mean = (mean_range[1] - mean_range[0]) / (pop_dim - 1)
        for num in range(pop_dim):
            tmp_mean[0, :, num] = mean_range[0] + delta_mean * num
        tmp_std = torch.zeros(1, obs_dim, pop_dim) + std
        self.mean = nn.Parameter(tmp_mean)
        self.std = nn.Parameter(tmp_std)

    def forward(self, obs, batch_size):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: pop_spikes
        """
        obs = obs.view(-1, self.obs_dim, 1)
        # Receptive Field of encoder population has Gaussian Shape
        pop_act = torch.exp(-(1. / 2.) * (obs - self.mean).pow(2) /
                            self.std.pow(2)).view(-1, self.encoder_neuron_num)
        pop_volt = torch.zeros(
            batch_size, self.encoder_neuron_num, device=self.device)
        pop_spikes = torch.zeros(
            batch_size, self.encoder_neuron_num, self.spike_ts, device=self.device)
        # Generate Regular Spike Trains
        for step in range(self.spike_ts):
            pop_volt = pop_volt + pop_act
            pop_spikes[:, :, step] = self.pseudo_spike(pop_volt)
            pop_volt = pop_volt - pop_spikes[:, :, step] * ENCODER_REGULAR_VTH
        return pop_spikes


class PopSpikeEncoderPoissonSpike(PopSpikeEncoderRegularSpike):
    """ Learnable Population Coding Spike Encoder with Poisson Spike Trains """

    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
        """
        :param obs_dim: observation dimension
        :param pop_dim: population dimension
        :param spike_ts: spike timesteps
        :param mean_range: mean range
        :param std: std
        :param device: device
        """
        super().__init__(obs_dim, pop_dim, spike_ts, mean_range, std, device)
        self.pseudo_spike = PseudoEncoderSpikePoisson.apply

    def forward(self, obs, batch_size):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: pop_spikes
        """
        obs = obs.view(-1, self.obs_dim, 1)
        # Receptive Field of encoder population has Gaussian Shape
        pop_act = torch.exp(-(1. / 2.) * (obs - self.mean).pow(2) /
                            self.std.pow(2)).view(-1, self.encoder_neuron_num)
        pop_spikes = torch.zeros(
            batch_size, self.encoder_neuron_num, self.spike_ts, device=self.device)
        # Generate Poisson Spike Trains
        for step in range(self.spike_ts):
            pop_spikes[:, :, step] = self.pseudo_spike(pop_act)
        return pop_spikes


class PopSpikeDecoder(nn.Module):
    """ Population Coding Spike Decoder """

    def __init__(self, act_dim, pop_dim, output_activation=nn.Tanh):
        """
        :param act_dim: action dimension
        :param pop_dim:  population dimension
        :param output_activation: activation function added on output
        """
        super().__init__()
        self.act_dim = act_dim
        self.pop_dim = pop_dim
        self.decoder = nn.Conv1d(act_dim, act_dim, pop_dim, groups=act_dim)
        self.output_activation = output_activation()

    def forward(self, pop_act):
        """
        :param pop_act: output population activity
        :return: raw_act
        """
        pop_act = pop_act.view(-1, self.act_dim, self.pop_dim)
        raw_act = self.output_activation(
            self.decoder(pop_act).view(-1, self.act_dim))
        return raw_act


class PseudoSpikeRect(torch.autograd.Function):
    """ Pseudo-gradient function for spike - Derivative of Rect Function """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(NEURON_VTH).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        spike_pseudo_grad = (abs(input - NEURON_VTH) <
                             SPIKE_PSEUDO_GRAD_WINDOW)
        return grad_input * spike_pseudo_grad.float()


class varyR_update(nn.Module):
    def __init__(self, n_neuron, init_np=0.1 , init_cdecay = 0.5 , init_vdecay = 0.75):
        super(varyR_update, self).__init__()
        self.n_neuron = n_neuron
        self.init_np = init_np
        self.pseudo_spike = PseudoSpikeRect.apply
        self.thresh = 1
 
        self.n_p = nn.Parameter(init_np * torch.ones(1, n_neuron))
        self.c_decay = nn.Parameter(init_cdecay * torch.ones(1,n_neuron))
        self.v_decay = nn.Parameter(init_vdecay * torch.ones(1,n_neuron))


    def forward(self, batch_size, ops, in_spike, current, mem, spike):

        np = self.n_p.expand([batch_size,-1]) 
        c_decay = self.c_decay.expand([batch_size,-1])
        v_decay = self.v_decay.expand([batch_size,-1])

        current = current * c_decay + (1 - np * mem / self.thresh) * V0_param * ops(in_spike)
        mem = v_decay * mem * (1. - spike) + current
        spike = self.pseudo_spike(mem)
    
        return current , mem , spike


class SpikeMLP(nn.Module):
    """ Spike MLP with Input and Output population neurons """

    def __init__(self, in_pop_dim, out_pop_dim, hidden_sizes, spike_ts, device):
        """
        :param in_pop_dim: input population dimension
        :param out_pop_dim: output population dimension
        :param hidden_sizes: list of hidden layer sizes
        :param spike_ts: spike timesteps
        :param device: device
        """
        super().__init__()
        self.in_pop_dim = in_pop_dim
        self.out_pop_dim = out_pop_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_num = len(hidden_sizes)
        self.spike_ts = spike_ts
        self.device = device
        self.pseudo_spike = PseudoSpikeRect.apply
        # Define Layers (Hidden Layers + Output Population)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(in_pop_dim, hidden_sizes[0])])
        if self.hidden_num > 1:
            for layer in range(1, self.hidden_num):
                self.hidden_layers.extend(
                    [nn.Linear(hidden_sizes[layer-1], hidden_sizes[layer])])
        self.out_pop_layer = nn.Linear(hidden_sizes[-1], out_pop_dim)

        # vary R update
        self.mem_update_hidden = varyR_update(self.hidden_sizes[0])
        self.mem_update_out = varyR_update(self.out_pop_dim)

    def forward(self, input_spikes, batch_size):
        """
        :param input_spikes: input population spikes
        :param batch_size: batch size
        :return: sum_spike
        """
         # Define LIF Neuron states: current , mem , spike
        hidden_states = []
        for layer in range(self.hidden_num):
            hidden_states.append([torch.zeros(batch_size, self.hidden_sizes[layer], device=self.device)
                                  for _ in range(3)])
        out_pop_states = [torch.zeros(batch_size, self.out_pop_dim, device=self.device)
                          for _ in range(3)]
        sum_spike = torch.zeros(
            batch_size, self.out_pop_dim, device=self.device)

        # Start Spike Timestep Iteration
        for step in range(self.spike_ts):
            in_pop_spike_t = input_spikes[:, :, step]

            # iteration of hidden_states
            hidden_states[0][0], hidden_states[0][1], hidden_states[0][2] = self.mem_update_hidden(batch_size,
                self.hidden_layers[0], in_pop_spike_t,
                hidden_states[0][0], hidden_states[0][1], hidden_states[0][2]
            )
            if self.hidden_num > 1:
                for layer in range(1, self.hidden_num):
                    hidden_states[layer][0], hidden_states[layer][1], hidden_states[layer][2] = self.mem_update_hidden(batch_size,
                        self.hidden_layers[layer], hidden_states[layer-1][2],
                        hidden_states[layer][0], hidden_states[layer][1], hidden_states[layer][2]
                    )

            # iteration of out_pop_states
            out_pop_states[0], out_pop_states[1], out_pop_states[2] = self.mem_update_out(batch_size,
                self.out_pop_layer, hidden_states[-1][2],
                out_pop_states[0], out_pop_states[1], out_pop_states[2]
            )

            # sum spike
            sum_spike += out_pop_states[2]
        sum_spike = sum_spike / self.spike_ts
        return sum_spike


class PopSpikeActor(nn.Module):
    """ Population Coding Spike Actor with Fix Encoder """

    def __init__(self, obs_dim, act_dim, en_pop_dim, de_pop_dim, hidden_sizes,
                 mean_range, std, spike_ts, act_limit, device, use_poisson):
        """
        :param obs_dim: observation dimension
        :param act_dim: action dimension
        :param en_pop_dim: encoder population dimension
        :param de_pop_dim: decoder population dimension
        :param hidden_sizes: list of hidden layer sizes
        :param mean_range: mean range for encoder
        :param std: std for encoder
        :param spike_ts: spike timesteps
        :param act_limit: action limit
        :param device: device
        :param use_poisson: if true use Poisson spikes for encoder
        """
        super().__init__()
        self.act_limit = act_limit
        if use_poisson:
            self.encoder = PopSpikeEncoderPoissonSpike(
                obs_dim, en_pop_dim, spike_ts, mean_range, std, device)
        else:
            self.encoder = PopSpikeEncoderRegularSpike(
                obs_dim, en_pop_dim, spike_ts, mean_range, std, device)
        self.snn = SpikeMLP(obs_dim*en_pop_dim, act_dim *
                            de_pop_dim, hidden_sizes, spike_ts, device)
        self.decoder = PopSpikeDecoder(act_dim, de_pop_dim)


    def forward(self, obs, batch_size):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: action scale with action limit
        """
        input_spikes = self.encoder(obs, batch_size)
        out_pop_activity = self.snn(input_spikes, batch_size)
        action = self.act_limit * self.decoder(out_pop_activity)
        return action
