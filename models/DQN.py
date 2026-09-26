import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
from collections import deque

from model.cells import GRUCell
from util import create_random_adjacency, normalize_adj





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TaskGating(nn.Module):
    def __init__(self, task_dim=2, num_synapse_types=7):
        super().__init__()
        self.fc = nn.Linear(task_dim, num_synapse_types)

    def forward(self, task_code):

        return torch.sigmoid(self.fc(task_code))

class CNNUnit(nn.Module):

    def __init__(self, in_channels=3, out_channels=32):
        super(CNNUnit, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.relu(self.conv(x)))


class ReceptiveFieldSynapse(nn.Module):

    def __init__(self, in_channels, out_channels=None, kernel_size=3):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
            bias=True,
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.conv(x))


class TopologyAwareSynapse(nn.Module):

    def __init__(self, in_channels, out_channels=None, adj_matrix=None):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if adj_matrix is None:
            self.register_buffer('adj_matrix', None)
        else:
            adj = torch.as_tensor(adj_matrix, dtype=torch.float32)
            self.register_buffer('adj_matrix', adj)

    def _normalize_adj(self, adj):
        adj = adj.to(dtype=torch.float32)
        eye = torch.eye(adj.size(0), dtype=adj.dtype, device=adj.device)
        adj = adj + eye
        deg = adj.sum(dim=-1).clamp_min(1e-6)
        deg_inv_sqrt = deg.pow(-0.5)
        return deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]

    def forward(self, x):
        x = self.proj(x)
        node_num = x.size(2)
        if self.adj_matrix is None or self.adj_matrix.size(0) != node_num:
            adj = torch.eye(node_num, dtype=x.dtype, device=x.device)
        else:
            adj = self._normalize_adj(self.adj_matrix.to(x.device))
        return torch.einsum('ij,bcjt->bcit', adj, x)


class DynamicIntegrationSynapse(nn.Module):

    def __init__(self, in_channels, out_channels=None, kernel_size=3):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.filter_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
        )
        self.gate_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
        )

    def forward(self, x):
        return torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))


class DelaySynapse(nn.Module):

    def __init__(self, max_delay=5):
        super().__init__()
        self.max_delay = max_delay
        self.delay_logits = nn.Parameter(torch.zeros(max_delay + 1))

    def forward(self, x):
        weights = torch.softmax(self.delay_logits, dim=0)
        shifted = []
        for delay in range(self.max_delay + 1):
            if delay == 0:
                shifted.append(x)
            else:
                shifted.append(torch.cat([x[..., :1].repeat(1, 1, 1, delay), x[..., :-delay]], dim=-1))
        return torch.stack(shifted, dim=0).mul(weights.view(-1, 1, 1, 1, 1)).sum(dim=0)


class DivergentProjectionSynapse(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.proj(x))


class IdentityProjectionSynapse(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.in_channels = channels
        self.out_channels = channels
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        return self.proj(x)


class ConvergentProjectionSynapse(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.proj(x))


class SkipProjectionSynapse(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.in_channels = channels
        self.out_channels = channels
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)


class AttentionProjectionSynapse(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.in_channels = channels
        self.out_channels = channels
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.scale = channels ** 0.5

    def forward(self, x):
        b, c, n, t = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(b, n * t, c)
        q = self.query(tokens)
        k = self.key(tokens)
        v = self.value(tokens)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / self.scale, dim=-1)
        out = torch.matmul(attn, v)
        return out.reshape(b, n, t, c).permute(0, 3, 1, 2)


class StochasticModulationSynapse(nn.Module):

    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        keep_prob = 1.0 - self.p
        mask = (torch.rand_like(x) < keep_prob).float()
        return x * mask / max(keep_prob, 1e-6)


class LeakyIntegrationSynapse(nn.Module):

    def __init__(self, alpha=0.9):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

    def forward(self, x):
        alpha = torch.sigmoid(self.alpha)
        out = []
        prev = torch.zeros_like(x[..., 0])
        for step in range(x.size(-1)):
            prev = alpha * prev + x[..., step]
            out.append(prev.unsqueeze(-1))
        return torch.cat(out, dim=-1)



MLPUnit = DivergentProjectionSynapse
ResnetUnit = SkipProjectionSynapse
SelfAttentionUnit = AttentionProjectionSynapse
SynapticDelayUnit = DelaySynapse
LeakyIntegrationUnit = LeakyIntegrationSynapse
SynapticNormalizationUnit = IdentityProjectionSynapse
StochasticSynapseUnit = StochasticModulationSynapse


class LinearAdapter(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class Conv1x1Adapter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)





class DynamicNetwork(nn.Module):

    SYNAPSE_POOL = {
        0: 'STOP',
        1: 'ReceptiveFieldSynapse',
        2: 'TopologyAwareSynapse',
        3: 'DynamicIntegrationSynapse',
        4: 'DelaySynapse',
        5: 'DivergentProjectionSynapse',
        6: 'IdentityProjectionSynapse',
        7: 'ConvergentProjectionSynapse',
        8: 'SkipProjectionSynapse',
        9: 'AttentionProjectionSynapse',
        10: 'StochasticModulationSynapse',
        11: 'LeakyIntegrationSynapse',
    }
    STOP_ACTION = 0

    def __init__(self, arguments, out_dim, task_index=0, horizon=12, trained_list=None):
        self.do_print = True
        super(DynamicNetwork, self).__init__()
        self.arguments = arguments
        self.hist_len = arguments.get('hist_len')
        self.pred_len = arguments.get('pred_len')
        self.horizon = horizon
        self.out_dim = out_dim
        self.hid_dim = 16
        self.layers = nn.ModuleList()
        self.output_head = nn.ModuleList()
        self.layer_num_list = []
        self.trained_list = trained_list if trained_list is not None else {}
        self.layer_cache_keys = []
        self.output_head_cache_keys = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.shape_trace = []
        self.task_index = task_index
        self.current_output_channels = int(arguments.get('input_channels', 3))
        self.current_feature_dim = arguments.get('in_dim')
        self.adj_matrix = arguments.get('adj_matrix')
        self._prepared = False

    @staticmethod
    def _clone_state_dict(module):
        return {name: param.detach().cpu().clone() for name, param in module.state_dict().items()}

    @staticmethod
    def _module_device(unit):
        first_param = next(unit.parameters(), None)
        return first_param.device if first_param is not None else device

    def _load_cached_params(self, unit, cache_key):
        if cache_key not in self.trained_list:
            self.cache_misses += 1
            return False
        target_device = self._module_device(unit)
        cached_state = {name: param.to(target_device) for name, param in self.trained_list[cache_key].items()}
        unit.load_state_dict(cached_state)
        self.cache_hits += 1
        return True

    def _append_layer_with_cache(self, unit, cache_key):
        self._load_cached_params(unit, cache_key)
        self.layers.append(unit)
        self.layer_cache_keys.append(cache_key)
        self._prepared = False

    def _make_linear_adapter(self, input_dim, output_dim, scope):
        cache_key = ('adapter', scope, 'LinearAdapter', input_dim, output_dim)
        unit = LinearAdapter(input_dim=input_dim, output_dim=output_dim)
        self._load_cached_params(unit, cache_key)
        return unit, cache_key

    def _make_conv_adapter(self, in_channels, out_channels, scope):
        cache_key = ('adapter', scope, 'Conv1x1Adapter', in_channels, out_channels)
        unit = Conv1x1Adapter(in_channels=in_channels, out_channels=out_channels)
        self._load_cached_params(unit, cache_key)
        return unit, cache_key

    def save_modules_to_cache(self):
        for unit, cache_key in zip(self.layers, self.layer_cache_keys):
            self.trained_list[cache_key] = self._clone_state_dict(unit)
        for unit, cache_key in zip(self.output_head, self.output_head_cache_keys):
            self.trained_list[cache_key] = self._clone_state_dict(unit)

    def cache_summary(self):
        return {'hits': self.cache_hits, 'misses': self.cache_misses, 'cached_modules': len(self.trained_list)}

    def shape_summary(self):
        return self.shape_trace

    def _expected_input(self, unit):
        if hasattr(unit, 'in_channels'):
            return {'channel': int(unit.in_channels)}
        if isinstance(unit, Conv1x1Adapter):
            return {'channel': unit.conv.in_channels}
        if isinstance(unit, LinearAdapter):
            return {'last_dim': unit.fc.in_features}
        return {}

    def _align_before_unit(self, x, unit, new_layers, new_keys, scope):
        expected = self._expected_input(unit)
        if 'channel' in expected and x.size(1) != expected['channel']:
            adapter, key = self._make_conv_adapter(x.size(1), expected['channel'], scope)
            adapter = adapter.to(x.device)
            before = tuple(x.shape)
            x = adapter(x)
            new_layers.append(adapter)
            new_keys.append(key)
            self.shape_trace.append({'layer': scope, 'type': 'Conv1x1Adapter', 'before': before, 'after': tuple(x.shape)})
        if 'last_dim' in expected and x.size(-1) != expected['last_dim']:
            adapter, key = self._make_linear_adapter(x.size(-1), expected['last_dim'], scope)
            adapter = adapter.to(x.device)
            before = tuple(x.shape)
            x = adapter(x)
            new_layers.append(adapter)
            new_keys.append(key)
            self.shape_trace.append({'layer': scope, 'type': 'LinearAdapter', 'before': before, 'after': tuple(x.shape)})
        return x

    def _rebuild_layers_with_alignment(self, x):
        new_layers = nn.ModuleList()
        new_keys = []
        old_layers = list(self.layers)
        old_keys = list(self.layer_cache_keys)
        self.shape_trace = []
        for idx, (unit, cache_key) in enumerate(zip(old_layers, old_keys)):
            action = self.layer_num_list[idx] if idx < len(self.layer_num_list) else 'adapter'
            scope = 'layer_%d_action_%s' % (idx, action)
            x = self._align_before_unit(x, unit, new_layers, new_keys, scope)
            unit = unit.to(x.device)
            before = tuple(x.shape)
            x = unit(x)
            new_layers.append(unit)
            new_keys.append(cache_key)
            self.shape_trace.append({'layer': scope, 'type': unit.__class__.__name__, 'before': before, 'after': tuple(x.shape)})
        self.layers = new_layers
        self.layer_cache_keys = new_keys
        return x

    def _build_output_head(self, x):
        head = nn.ModuleList()
        head_keys = []
        if x.size(-1) != self.horizon:
            unit, key = self._make_linear_adapter(x.size(-1), self.horizon, 'output_time')
            unit = unit.to(x.device)
            before = tuple(x.shape)
            x = unit(x)
            head.append(unit)
            head_keys.append(key)
            self.shape_trace.append({'layer': 'output_time', 'type': 'LinearAdapter', 'before': before, 'after': tuple(x.shape)})
        if x.size(1) != self.out_dim:
            unit, key = self._make_conv_adapter(x.size(1), self.out_dim, 'output_channel')
            unit = unit.to(x.device)
            before = tuple(x.shape)
            x = unit(x)
            head.append(unit)
            head_keys.append(key)
            self.shape_trace.append({'layer': 'output_channel', 'type': 'Conv1x1Adapter', 'before': before, 'after': tuple(x.shape)})
        self.output_head = head
        self.output_head_cache_keys = head_keys
        return x

    def prepare_for_training(self, input_shape, target_device=None):
        target_device = target_device or device
        self.to(target_device)
        dummy = torch.zeros(input_shape, dtype=torch.float32, device=target_device)
        x = dummy.transpose(1, 3)
        with torch.no_grad():
            x = self._rebuild_layers_with_alignment(x)
            x = self._build_output_head(x)
            out = x.transpose(1, 3)
        expected = (input_shape[0], self.horizon, input_shape[2], self.out_dim)
        if tuple(out.shape) != expected:
            raise RuntimeError('dummy forward shape mismatch: got %s, expected %s' % (tuple(out.shape), expected))
        if not torch.isfinite(out).all():
            raise RuntimeError('dummy forward produced non-finite values')
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if param_count == 0:
            raise RuntimeError('dynamic network has no trainable parameters after alignment')
        self._prepared = True
        return out.shape

    def add_layer(self, layer_type):
        self.output_head = nn.ModuleList()
        self.output_head_cache_keys = []
        self._prepared = False
        in_channels = int(self.current_output_channels)

        if layer_type == self.STOP_ACTION:
            raise ValueError('Action 0 is STOP and should be handled outside add_layer.')
        elif layer_type == 1:
            unit = ReceptiveFieldSynapse(in_channels=in_channels, out_channels=in_channels, kernel_size=3)
            cache_key = ('action_1', 'ReceptiveFieldSynapse', in_channels, in_channels, 3)
        elif layer_type == 2:
            unit = TopologyAwareSynapse(in_channels=in_channels, out_channels=in_channels, adj_matrix=self.adj_matrix)
            adj_shape = None if self.adj_matrix is None else tuple(np.asarray(self.adj_matrix).shape)
            cache_key = ('action_2', 'TopologyAwareSynapse', in_channels, in_channels, adj_shape)
        elif layer_type == 3:
            unit = DynamicIntegrationSynapse(in_channels=in_channels, out_channels=in_channels, kernel_size=3)
            cache_key = ('action_3', 'DynamicIntegrationSynapse', in_channels, in_channels, 3)
        elif layer_type == 4:
            unit = DelaySynapse(max_delay=5)
            cache_key = ('action_4', 'DelaySynapse', 5)
        elif layer_type == 5:
            out_channels = max(in_channels * 2, 1)
            unit = DivergentProjectionSynapse(in_channels=in_channels, out_channels=out_channels)
            cache_key = ('action_5', 'DivergentProjectionSynapse', in_channels, out_channels)
            self.current_output_channels = out_channels
        elif layer_type == 6:
            unit = IdentityProjectionSynapse(channels=in_channels)
            cache_key = ('action_6', 'IdentityProjectionSynapse', in_channels)
        elif layer_type == 7:
            out_channels = max(in_channels // 2, 1)
            unit = ConvergentProjectionSynapse(in_channels=in_channels, out_channels=out_channels)
            cache_key = ('action_7', 'ConvergentProjectionSynapse', in_channels, out_channels)
            self.current_output_channels = out_channels
        elif layer_type == 8:
            unit = SkipProjectionSynapse(channels=in_channels)
            cache_key = ('action_8', 'SkipProjectionSynapse', in_channels)
        elif layer_type == 9:
            unit = AttentionProjectionSynapse(channels=in_channels)
            cache_key = ('action_9', 'AttentionProjectionSynapse', in_channels)
        elif layer_type == 10:
            unit = StochasticModulationSynapse(p=0.1)
            cache_key = ('action_10', 'StochasticModulationSynapse', 0.1)
        elif layer_type == 11:
            unit = LeakyIntegrationSynapse(alpha=0.9)
            cache_key = ('action_11', 'LeakyIntegrationSynapse', 0.9)
        else:
            raise ValueError('Unknown MSyN synapse action %s. Valid actions are 1-11; action 0 is STOP.' % layer_type)

        self._append_layer_with_cache(unit, cache_key)
        self.layer_num_list.append(layer_type)

    def forward(self, input, label=None):
        if not self._prepared:
            self.prepare_for_training(tuple(input.shape), input.device)
        x = input.transpose(1, 3)
        for layer in self.layers:
            x = layer(x)
        for layer in self.output_head:
            x = layer(x)
        return x.transpose(1, 3)






class ControllerNet(nn.Module):

    def __init__(self, max_layers=5, action_dim=3, task_index=0, task_dim=2, state_extra_dim=0):
        super(ControllerNet, self).__init__()
        self.state_dim = max_layers + state_extra_dim
        self.fc1 = nn.Linear(self.state_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, action_dim)


        self.task_code = torch.zeros(task_dim, device=device)
        if task_index is not None and task_index < task_dim:
            self.task_code[task_index] = 1
        self.task_gating = TaskGating(task_dim=task_dim, num_synapse_types=action_dim).to(device)

    def set_task_code(self, task_code):
        task_code = torch.as_tensor(task_code, dtype=torch.float32, device=device).flatten()
        if task_code.numel() != self.task_code.numel():
            raise ValueError(f"task code dimension mismatch: got {task_code.numel()}, expected {self.task_code.numel()}")
        self.task_code = task_code
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        gates = self.task_gating(self.task_code)
        gated_q_value = gates * x

        return gated_q_value


class DQNController:
    def __init__(self, action_dim=3, max_layers=5, epsilon=0.5, epsilon_min=0.2, task_index=0, task_dim=2, state_extra_dim=0):
        self.state_dim = max_layers + state_extra_dim
        self.q_net = ControllerNet(max_layers=max_layers, action_dim=action_dim, task_index=task_index, task_dim=task_dim, state_extra_dim=state_extra_dim).to(device)
        self.target_net = ControllerNet(max_layers=max_layers, action_dim=action_dim, task_index=task_index, task_dim=task_dim, state_extra_dim=state_extra_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.memory = deque(maxlen=5000)
        self.batch_size = 64
        self.gamma = 0.9
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = 0.995
        self.action_dim = action_dim

    def set_task_code(self, task_code):
        """Update task conditioning while preserving the shared controller state."""
        self.q_net.set_task_code(task_code)
        self.target_net.set_task_code(task_code)

    def choose_action(self, state):

        if random.random() < self.epsilon:
            print('random choose')
            return random.randint(0, self.action_dim - 1)

        state = torch.tensor(np.array([state]), dtype=torch.float32).to(device)
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"state dim mismatch: got {state.shape[-1]}, expected {self.state_dim}")
        q_values = self.q_net(state)
        print('choose action q value shape:', q_values.shape)



        return q_values.argmax().item()


    def store(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    def train_step(self, success=False):
        if len(self.memory) < self.batch_size:
            return









        self.optimizer.zero_grad()
        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.tensor(state, dtype=torch.float32).to(device)
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"state dim mismatch: got {state.shape[-1]}, expected {self.state_dim}")
        action = torch.tensor(action, dtype=torch.long).view(-1, 1).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).view(-1, 1).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
        if next_state.shape[-1] != self.state_dim:
            raise ValueError(f"next_state dim mismatch: got {next_state.shape[-1]}, expected {self.state_dim}")
        done = torch.tensor(done, dtype=torch.float32).view(-1, 1).to(device)
        print('state shape:', state.shape, 'action shape:', action.shape, 'reward shape:', reward.shape, 'next state shape:', next_state.shape, 'done shape:', done.shape)



        q_values = self.q_net(state).gather(1, action)

        q_next = self.target_net(next_state).max(1)[0].view(-1, 1)
        target = reward + self.gamma * (1 - done) * q_next
        print('q value shape:', q_values.shape, 'target shape:', target.shape)
        loss = F.mse_loss(q_values, target)

        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min and success:
            self.epsilon *= self.epsilon_decay


def train_dynamic_network(model, train_loader, device, epochs, task_index, scaler, loss_fn, lrate, scheduler, clip_grad_value, log_dir, logger, seed, save_model):

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    losses = []
    num_epochs = epochs

    return np.mean(losses)
