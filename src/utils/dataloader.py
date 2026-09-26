import os
import pickle
import torch
import numpy as np
import threading
import multiprocessing as mp

class DataLoader(object):
    def __init__(self, data, idx, seq_len, horizon, bs, logger, pad_last_sample=False):
        if pad_last_sample:
            num_padding = (bs - (len(idx) % bs)) % bs
            idx_padding = np.repeat(idx[-1:], num_padding, axis=0)
            idx = np.concatenate([idx, idx_padding], axis=0)

        self.data = data
        self.idx = idx
        self.size = len(idx)
        self.bs = bs
        self.num_batch = int(self.size // self.bs)
        self.current_ind = 0
        logger.info('Sample num: ' + str(self.idx.shape[0]) + ', Batch num: ' + str(self.num_batch))

        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon


    def shuffle(self):
        perm = np.random.permutation(self.size)
        idx = self.idx[perm]
        self.idx = idx


    def write_to_shared_array(self, x, y, idx_ind, start_idx, end_idx):
        for i in range(start_idx, end_idx):
            x[i] = self.data[idx_ind[i] + self.x_offsets, :, :]
            y[i] = self.data[idx_ind[i] + self.y_offsets, :, :1]


    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = self.idx[start_ind: end_ind, ...]

                x_shape = (len(idx_ind), self.seq_len, self.data.shape[1], self.data.shape[-1])
                x_shared = mp.RawArray('f', int(np.prod(x_shape)))
                x = np.frombuffer(x_shared, dtype='f').reshape(x_shape)

                y_shape = (len(idx_ind), self.horizon, self.data.shape[1], 1)
                y_shared = mp.RawArray('f', int(np.prod(y_shape)))
                y = np.frombuffer(y_shared, dtype='f').reshape(y_shape)

                array_size = len(idx_ind)
                num_threads = len(idx_ind) // 2
                chunk_size = array_size // num_threads
                threads = []
                for i in range(num_threads):
                    start_index = i * chunk_size
                    end_index = start_index + chunk_size if i < num_threads - 1 else array_size
                    thread = threading.Thread(target=self.write_to_shared_array, args=(x, y, idx_ind, start_index, end_index))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

                yield (x, y)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    def __init__(self, mean, std):


        self.mean = mean
        self.std = std


    def transform(self, data):
        return (data - self.mean) / self.std


    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(data_path, args, logger):
    ptr = np.load(os.path.join('./data',data_path, args.years, 'his.npz'))
    logger.info('Data shape: ' + str(ptr['data'].shape))

    dataloader = {}
    for cat in ['train', 'val', 'test']:
        idx = np.load(os.path.join(data_path, args.years, 'idx_' + cat + '.npy'))
        dataloader[cat + '_loader'] = DataLoader(ptr['data'][..., :args.input_dim], idx,\
                                                 args.seq_len, args.horizon, args.bs, logger)

    scaler = StandardScaler(mean=ptr['mean'], std=ptr['std'])
    return dataloader, scaler


def load_adj_from_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj_from_numpy(numpy_file):
    return np.load(numpy_file)


def get_dataset_info(dataset):
    base_dir = os.getcwd() + '/data/'
    d = {
         'CA': [base_dir+'ca', base_dir+'ca/ca_rn_adj.npy', 8600],
         'GLA': [base_dir+'gla', base_dir+'gla/gla_rn_adj.npy', 3834],
         'GBA': [base_dir+'gba', base_dir+'gba/gba_rn_adj.npy', 2352],
         'SD': [base_dir+'sd', base_dir+'sd/sd_rn_adj.npy', 716],
         'SIPFLOW': [base_dir+'sipflow', base_dir+'sipflow/sip_adj.npy', 108],
         'SIPSPEED': [base_dir+'sipspeed', base_dir+'sipspeed/sip_adj.npy', 108],
         'NYCCROWDIN': [base_dir+'nyccrowdin', base_dir+'nyccrowdin/nyc_adj.npy', 206],
         'NYCCROWDOUT': [base_dir+'nyccrowdout', base_dir+'nyccrowdout/nyc_adj.npy', 206],
         'NYCTAXIDROP': [base_dir+'nyctaxidrop', base_dir+'nyctaxidrop/nyc_adj.npy', 206],
         'NYCTAXIPICK': [base_dir+'nyctaxipick', base_dir+'nyctaxipick/nyc_adj.npy', 206],
         'CHIRISK': [base_dir+'chirisk', base_dir+'chirisk/chi_adj.npy', 220],
         'CHITAXIDROP': [base_dir+'chitaxidrop', base_dir+'chitaxidrop/chi_adj.npy', 220],
         'CHITAXIPICK': [base_dir+'chitaxipick', base_dir+'chitaxipick/chi_adj.npy', 220],
        }
    assert dataset in d.keys()
    return d[dataset]


def load_dataset_new(data_path, args, logger, task_per_dir=4):
    ptr = np.load(os.path.join(data_path, args.years, 'his.npz'))
    logger.info('Data shape: ' + str(ptr['data'].shape))




    dataloader_list = []
    scaler_list = []


    tod_array = ptr['data'][:, 0, 1]
    data_array = ptr['data'][..., 0]

    for task_idx in range(task_per_dir):

        dataloader = {}


        task_start = task_idx / task_per_dir
        task_end = (task_idx + 1) / task_per_dir

        for cat in ['train', 'val', 'test']:
            idx = np.load(os.path.join(data_path, args.years, 'idx_' + cat + '.npy'))











            mask = (tod_array[idx] >= task_start) & (tod_array[idx] < task_end)
            new_idx = idx[mask]
            new_data = data_array[new_idx]

            new_idx = np.array(new_idx)
            dataloader[cat + '_loader'] = DataLoader(ptr['data'][..., :args.input_dim], new_idx,\
                                                    args.seq_len, args.horizon, args.bs, logger)

            if cat == 'train':
                scaler_list.append(StandardScaler(mean=new_data.mean(), std=new_data.std()))

        dataloader_list.append(dataloader)

    return dataloader_list, scaler_list
