import os
import argparse
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))
import traceback
import torch

from util import config
from src.base.engine import BaseEngine
from src.base.engine_DQN import BaseEngine_DQN
from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset_new, load_adj_from_numpy, get_dataset_info
from src.utils.metrics import masked_mae, cross_entropy, masked_mse
from src.utils.logging import get_logger
from src.utils.graph_algo import normalize_adj_mx, calculate_cheb_poly
from fastdtw import fastdtw
from models.DQN import DQNController, DynamicNetwork
import random
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

def get_config(task_index=0):
    parser = get_public_config()
    parser.add_argument('--tpd', type=int, default=96, help='time per day')
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--thres', type=float, default=0.6)
    parser.add_argument('--lrate', type=float, default=2e-3)
    parser.add_argument('--wdecay', type=float, default=0)
    parser.add_argument('--clip_grad_value', type=float, default=0)

    parser.add_argument('--order', type=int, default=3)
    parser.add_argument('--nb_block', type=int, default=2)
    parser.add_argument('--nb_chev_filter', type=int, default=64)
    parser.add_argument('--nb_time_filter', type=int, default=64)
    parser.add_argument('--time_stride', type=int, default=1)

    parser.add_argument('--embed_dim', type=int, default=10)
    parser.add_argument('--rnn_unit', type=int, default=64)
    parser.add_argument('--num_layer', type=int, default=2)
    parser.add_argument('--cheb_k', type=int, default=2)

    parser.add_argument('--adj_type', type=str, default='doubletransition')
    parser.add_argument('--adp_adj', type=int, default=1)
    parser.add_argument('--init_dim', type=int, default=32)
    parser.add_argument('--skip_dim', type=int, default=256)
    parser.add_argument('--end_dim', type=int, default=512)

    parser.add_argument('--Kt', type=int, default=3)
    parser.add_argument('--Ks', type=int, default=3)
    parser.add_argument('--block_num', type=int, default=2)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)

    parser.add_argument('--blocks', type=int, default=2)
    parser.add_argument('--mlp_expand', type=int, default=2)
    parser.add_argument('--hid_dim', type=int, default=32)
    args = parser.parse_args()

    log_dir = './experiments/{}/{}/'.format(args.model_name, args.dataset)
    logger = get_logger(log_dir, __name__, 'record_s{}_taskindex_{}.log'.format(args.seed, task_index))
    logger.info(args)

    return args, log_dir, logger

max_layers = 10
epsilon = config['train']['epsilon']
epsilon_min = config['train']['epsilon_min']

action_dim = 12
eta = config['train']['eta']
episodes = config['train']['episodes']
lambda_complexity = config['train']['lambda_complexity']

task_index = 0
task_dim = config['train']['task_dim']
TASK_CODE_DIM = task_dim + 4

class_num = 3
assert task_dim == 2

def build_task_code(task_index, X, adj_matrix):
    x = np.asarray(X, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected X with shape [B,T,N,F], got {x.shape}")
    adj = np.asarray(adj_matrix, dtype=np.float32)
    if adj.shape[0] != x.shape[2] or adj.shape[1] != x.shape[2]:
        raise ValueError(f"Adjacency/X node mismatch: {adj.shape} vs {x.shape}")

    a = adj + np.eye(adj.shape[0], dtype=np.float32)
    degree = a.sum(axis=1, keepdims=True)
    a_norm = a / np.maximum(degree, 1e-6)
    propagated = np.einsum("ij,btjf->btif", a_norm, x)
    spatial = np.array([
        np.mean(np.abs(propagated)),
        np.std(propagated),
    ], dtype=np.float32)
    spectrum = np.abs(np.fft.rfft(x, axis=1))
    frequencies = np.fft.rfftfreq(x.shape[1], d=1.0)
    spectral_strength = spectrum.mean(axis=(0, 2, 3))
    if spectral_strength.shape[0] > 1:
        dominant_idx = 1 + int(np.argmax(spectral_strength[1:]))
    else:
        dominant_idx = 0
    omega_max = float(frequencies[dominant_idx])
    f_max = float(spectral_strength[dominant_idx])
    temporal = np.array([omega_max, f_max], dtype=np.float32)

    one_hot = np.zeros(task_dim, dtype=np.float32)
    one_hot[int(task_index)] = 1.0
    return np.concatenate([one_hot, spatial, temporal])


def get_mean_std_min_max(data_list):
    data = np.asarray(data_list)
    return data.mean(), data.std(), data.min(), data.max()


def get_average_gradient_property(model):
    grad_terms = []
    for param in model.parameters():
        if param.grad is not None:
            grad_terms.append(torch.sum(param.grad.detach() ** 2))
    if not grad_terms:
        return 0.0
    return torch.stack(grad_terms).mean().item()


def compute_network_energy(model, loss, sigma_out):
    depth = max(len(model.layer_num_list), 0)
    avg_grad = get_average_gradient_property(model)
    safe_sigma_out = max(float(abs(sigma_out)), 1e-6)
    safe_loss = max(float(loss), 0.0)
    depth_factor = float(eta) ** depth
    grad_factor = float(lambda_complexity) ** avg_grad
    energy = (depth_factor * grad_factor) / (1.0 + safe_loss / safe_sigma_out)
    return energy, depth, avg_grad


def get_incremental_reward(model, loss, sigma_out, previous_energy):
    current_energy, depth, avg_grad = compute_network_energy(model, loss, sigma_out)
    return current_energy - previous_energy, current_energy, depth, avg_grad


def get_exp_info():
    exp_info =  '============== Train Info ==============\n' +\
                'max_layer: %s\n' % max_layers +\
                'epsilon: %s\n' % epsilon +\
                'epsilon_min: %s\n' % epsilon_min +\
                'lambda_complexity: %s\n' % lambda_complexity +\
                'episodes: %s\n' % episodes +\
                'task_dim :%s\n' % task_dim +\
                'task_index :%s\n' % task_index +\
                'class_num :%s\n' % class_num +\
    '========================================\n'
    return exp_info
def run_task(task_index, shared_controller=None):
    args, log_dir, logger = get_config(task_index=task_index)
    if task_index == 0:
        input_dim = args.input_dim
        out_dim = args.output_dim
    elif task_index == 1:
        input_dim = args.input_dim
        out_dim = class_num
    set_seed(args.seed)
    use_cuda = torch.cuda.is_available()

    device = torch.device('cuda' if use_cuda else 'cpu')



    dataset_list = []
    data_path_list = []
    adj_path_list = []
    node_num_list = []
    save_model = True
    model_name = args.model_name
    max_layers = 10
    logger.info(get_exp_info())

    if args.dataset == "NYC":
        dataset_list = ['NYCCROWDIN', 'NYCCROWDOUT', 'NYCTAXIDROP', 'NYCTAXIPICK']

    elif args.dataset == "CHI":
        dataset_list = ['CHIRISK', 'CHITAXIPICK', 'CHITAXIDROP']

    else:
        dataset_list = ['SIPFLOW', 'SIPSPEED']

    for dataset in dataset_list:
        data_path, adj_path, node_num = get_dataset_info(dataset)
        data_path_list.append(data_path)
        adj_path_list.append(adj_path)
        node_num_list.append(node_num)

    train_loaders = {}
    val_loaders = {}
    test_loaders = {}

    scalers = []

    for i in range(len(dataset_list)):
        train_loaders[i] = {}
        val_loaders[i] = {}
        test_loaders[i] = {}

    for i in range(len(dataset_list)):
        dataloader_list, scaler_list = load_dataset_new(data_path_list[i], args, logger, task_per_dir=1)

        for j in range(len(dataloader_list)):
            train_loaders[i][j]=(dataloader_list[j]['train_loader'])
            val_loaders[i][j]=(dataloader_list[j]['val_loader'])
            test_loaders[i][j]=(dataloader_list[j]['test_loader'])

        scalers.append(scaler_list)


    adj_path = adj_path_list[0]
    node_num = node_num_list[0]

    logger.info('Adj path: ' + adj_path)
    dummy_input_shape = (1, args.seq_len, node_num, args.input_dim)
    if args.model_name == 'DQN':

            arguments = {}


            arguments['hist_len'] = args.seq_len
            arguments['pred_len'] = args.seq_len
            arguments['in_dim'] = args.horizon
            arguments['input_channels'] = input_dim
            arguments['adj_matrix'] = load_adj_from_numpy(adj_path)
            stop_action = DynamicNetwork.STOP_ACTION
            module_param_cache = {}
            for k in range(5):
                rmse_list, mae_list, mape_list, loss_list, acc_list = [], [], [], [], []
                controller = shared_controller or DQNController(max_layers=max_layers, action_dim=action_dim, epsilon=epsilon, epsilon_min=epsilon_min, task_index=None, task_dim=TASK_CODE_DIM, state_extra_dim=2 + TASK_CODE_DIM)
                logger.info('DQN action mapping: %s' % DynamicNetwork.SYNAPSE_POOL)
                num_epochs = 5
                for i in range(len(train_loaders)):
                    for j in range(len(train_loaders[i])):

                        logger.info('========================circle %d task %d for %s start training...========================' % (k, j, dataset_list[i]))
                        train_loader = train_loaders[i][j]
                        val_loader = val_loaders[i][j]
                        test_loader = test_loaders[i][j]
                        x_probe, _ = next(train_loader.get_iterator())
                        task_code = build_task_code(task_index, x_probe, arguments['adj_matrix'])
                        controller.set_task_code(task_code)
                        scaler = scalers[i][j]
                        input_mean = float(np.asarray(scaler.mean).mean())
                        output_std = float(np.asarray(scaler.std).mean())
                        input_var = output_std ** 2
                        state_data_stats = np.array([input_mean, input_var], dtype=np.float32)
                        logger.info('State data stats | mean: %.6f, var: %.6f, sigma_out: %.6f' % (input_mean, input_var, output_std))

                        dataloader = {'train_loader':train_loader, 'val_loader':val_loader, 'test_loader':test_loader}

                        if task_index == 0:
                            loss_fn = masked_mse
                        elif task_index == 1:
                            loss_fn = cross_entropy


                        for ep in range(episodes):


                            reward = 0
                            success = True
                            logger.info(f"\n================ EPISODE {ep} ================")
                            catch_error = False

                            net = DynamicNetwork(arguments=arguments, out_dim=out_dim, task_index=task_index, horizon=args.horizon, trained_list=module_param_cache)
                            state = np.full(max_layers + 2 + TASK_CODE_DIM, -1, dtype=np.float32)
                            state[:max_layers] = -1
                            state[max_layers:max_layers + 2] = state_data_stats
                            state[max_layers + 2:] = task_code
                            next_state = state.copy()
                            action_num = 0
                            previous_energy = 0.0

                            done = False
                            while not done:
                                catch_error = False
                                action = controller.choose_action(state)
                                logger.info(action)

                                if action == stop_action or action_num >= max_layers-1:
                                    logger.info('结束')
                                    action = stop_action

                                    next_state = state.copy()
                                    next_state[action_num] = action
                                    done = True

                                    try:

                                        net.prepare_for_training(dummy_input_shape, device)
                                        logger.info('Shape trace: %s' % net.shape_summary())
                                        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lrate, weight_decay=args.wdecay)
                                        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
                                        engine = BaseEngine_DQN(device=device,
                                                model=net,
                                                dataloader=dataloader,
                                                scaler=scaler,
                                                sampler=None,
                                                loss_fn=loss_fn,
                                                lrate=args.lrate,
                                                optimizer=optimizer,
                                                scheduler=scheduler,
                                                clip_grad_value=args.clip_grad_value,
                                                max_epochs=num_epochs,
                                                patience=num_epochs,
                                                log_dir=log_dir,
                                                logger=logger,
                                                seed=args.seed,
                                                save_model=save_model,
                                                task_index=task_index
                                                )

                                        avg_test_mae, avg_test_rmse, avg_test_mape, loss, acc = engine.train()
                                    except Exception as e:


                                        success = False


                                        catch_error = True
                                        logger.info("网络训练失败:%s" % e)

                                        reward = -1000
                                        controller.store(state, action, reward, next_state, done)
                                        controller.train_step(success=success)
                                        break


                                    reward, current_energy, depth, avg_grad = get_incremental_reward(
                                        model=net,
                                        loss=loss,
                                        sigma_out=output_std,
                                        previous_energy=previous_energy
                                    )
                                    previous_energy = current_energy

                                    logger.info(
                                        f"Loss = {loss:.4f}, Depth = {depth}, AvgGrad = {avg_grad:.6f}, "
                                        f"Energy = {current_energy:.6f}, Reward = {reward:.6f}"
                                    )
                                    net.save_modules_to_cache()
                                    logger.info('Module cache summary: %s' % net.cache_summary())
                                    if not catch_error:


                                        logger.info('Test RMSE: %0.2f, MAE: %0.2f, MAPE: %0.2f, ACC: %0.4f' % (avg_test_rmse, avg_test_mae, avg_test_mape, acc))

                                        loss_list.append(loss)
                                        rmse_list.append(avg_test_rmse)
                                        mae_list.append(avg_test_mae)
                                        mape_list.append(avg_test_mape)



                                        acc_list.append(acc)


                                    controller.store(state, action, reward, next_state, done)
                                    controller.train_step(success=success)
                                    state = next_state.copy()
                                    action_num+=1
                                else:
                                    try:

                                        net.add_layer(action)

                                        next_state = state.copy()
                                        next_state[action_num]=action
                                        net.prepare_for_training(dummy_input_shape, device)
                                        logger.info('Shape trace: %s' % net.shape_summary())
                                        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lrate, weight_decay=args.wdecay)
                                        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
                                        engine = BaseEngine_DQN(device=device,
                                                model=net,
                                                dataloader=dataloader,
                                                scaler=scaler,
                                                sampler=None,
                                                loss_fn=loss_fn,
                                                lrate=args.lrate,
                                                optimizer=optimizer,
                                                scheduler=scheduler,
                                                clip_grad_value=args.clip_grad_value,
                                                max_epochs=num_epochs,
                                                patience=num_epochs,
                                                log_dir=log_dir,
                                                logger=logger,
                                                seed=args.seed,
                                                save_model=save_model,
                                                task_index=task_index
                                                )
                                        avg_test_mae, avg_test_rmse, avg_test_mape, loss, acc = engine.train()

                                    except Exception as e:

                                        logger.info("网络校验或训练失败:%s" % e)
                                        success = False
                                        catch_error = True
                                        done = True
                                        loss = 1000
                                        reward = -1000
                                        controller.store(state, action, reward, next_state, done)
                                        controller.train_step(success=success)
                                        break
                                    reward, current_energy, depth, avg_grad = get_incremental_reward(
                                        model=net,
                                        loss=loss,
                                        sigma_out=output_std,
                                        previous_energy=previous_energy
                                    )
                                    previous_energy = current_energy
                                    logger.info(
                                        f"Loss = {loss:.4f}, Depth = {depth}, AvgGrad = {avg_grad:.6f}, "
                                        f"Energy = {current_energy:.6f}, Reward = {reward:.6f}"
                                    )
                                    if not catch_error:
                                        net.save_modules_to_cache()
                                        logger.info('Module cache summary: %s' % net.cache_summary())


                                    controller.store(state, action, reward, next_state, done)
                                    controller.train_step(success=success)
                                    state = next_state.copy()
                                    action_num += 1



                            if ep % 5 == 0:
                                controller.target_net.load_state_dict(controller.q_net.state_dict())



                        logger.info("训练完毕")
                        logger.info('\nNo.%2d experiment results:' % k)
                        mean_std_min_max_rmse = get_mean_std_min_max(rmse_list)
                        mean_std_min_max_mae = get_mean_std_min_max(mae_list)



                        mean_std_min_max_mape = get_mean_std_min_max(mape_list)
                        mean_std_min_max_acc = get_mean_std_min_max(acc_list)
                        final_exp_str = 'RMSE       | mean: %0.4f std: %0.4f min: %0.4f max: %0.4f\n' % (mean_std_min_max_rmse) +\
                            'MAE        | mean: %0.4f std: %0.4f min: %0.4f max: %0.4f\n' % (mean_std_min_max_mae) +\
                            'MAPE        | mean: %0.4f std: %0.4f min: %0.4f max: %0.4f\n' % (mean_std_min_max_mape) +\
                            'ACC        | mean: %0.4f std: %0.4f min: %0.4f max: %0.4f\n' % (mean_std_min_max_acc)
                        logger.info(final_exp_str)
def construct_se_matrix(data_path, args):
    ptr = np.load(os.path.join(data_path, args.years, 'his.npz'))
    data = ptr['data'][..., 0]
    sample_num, node_num = data.shape

    data_mean = np.mean([data[args.tpd * i: args.tpd * (i + 1)] for i in range(sample_num // args.tpd)], axis=0)
    data_mean = data_mean.T

    dist_matrix = np.zeros((node_num, node_num))
    for i in range(node_num):
        for j in range(i, node_num):
            dist_matrix[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]

    for i in range(node_num):
        for j in range(i):
            dist_matrix[i][j] = dist_matrix[j][i]

    mean = np.mean(dist_matrix)
    std = np.std(dist_matrix)
    dist_matrix = (dist_matrix - mean) / std
    dist_matrix = np.exp(-dist_matrix ** 2 / args.sigma ** 2)
    dtw_matrix = np.zeros_like(dist_matrix)
    dtw_matrix[dist_matrix > args.thres] = 1
    return dtw_matrix


def normalize_adj_mx_single(adj_mx):
    alpha = 0.8
    D = np.array(np.sum(adj_mx, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), adj_mx),
                         diag.reshape((1, -1)))
    A_reg = alpha / 2 * (np.eye(adj_mx.shape[0]) + A_wave)
    return torch.from_numpy(A_reg.astype(np.float32))


def main():
    shared_controller = DQNController(max_layers=max_layers, action_dim=action_dim, epsilon=epsilon, epsilon_min=epsilon_min, task_index=None, task_dim=TASK_CODE_DIM, state_extra_dim=2 + TASK_CODE_DIM)
    for current_task in range(task_dim):
        run_task(current_task, shared_controller=shared_controller)


if __name__ == "__main__":
    main()
