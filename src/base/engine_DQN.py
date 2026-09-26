import os
import time
import torch
import numpy as np

from src.utils.metrics import masked_mape, masked_mae
from src.utils.metrics import masked_rmse, accuracy
from src.utils.metrics import compute_all_metrics
def remap_labels_by_threshold(label, threshold=20):
\
\

    label = label.copy()

    label[(label > 0) & (label <= threshold)] = 1
    label[label > threshold] = 2
    return label
class BaseEngine_DQN():
    def __init__(self, device, model, dataloader, scaler, sampler, loss_fn, lrate, optimizer,\
                 scheduler, clip_grad_value, max_epochs, patience, log_dir, logger, seed, save_model, task_index=0, threshold=20):
        super().__init__()
        self._device = device
        self.model = model
        self.model.to(self._device)

        self._dataloader = dataloader
        self._scaler = scaler

        self._loss_fn = loss_fn
        self._lrate = lrate
        self._optimizer = optimizer
        self._lr_scheduler = scheduler
        self._clip_grad_value = clip_grad_value

        self._max_epochs = max_epochs
        self._patience = patience
        self._iter_cnt = 0
        self._save_path = log_dir
        self._logger = logger
        self._seed = seed


        self._save_model = save_model
        self._task_index = task_index
        self._threshold = threshold
    def _to_device(self, tensors):
        if isinstance(tensors, list):
            return [tensor.to(self._device) for tensor in tensors]
        else:
            return tensors.to(self._device)


    def _to_numpy(self, tensors):
        if isinstance(tensors, list):
            return [tensor.detach().cpu().numpy() for tensor in tensors]
        else:
            return tensors.detach().cpu().numpy()


    def _to_tensor(self, nparray):
        if isinstance(nparray, list):
            return [torch.tensor(array, dtype=torch.float32) for array in nparray]
        else:
            return torch.tensor(nparray, dtype=torch.float32)


    def _inverse_transform(self, tensors):
        def inv(tensor):
            return self._scaler.inverse_transform(tensor)

        if isinstance(tensors, list):
            return [inv(tensor) for tensor in tensors]
        else:
            return inv(tensors)


    def save_model(self, save_path):

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = 'final_model_s{}.pt'.format(self._seed)
        if self._save_model:
            torch.save(self.model.state_dict(), os.path.join(save_path, filename))


    def load_model(self, save_path):
        filename = 'final_model_s{}.pt'.format(self._seed)
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename)))


    def train_batch(self):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        self._dataloader['train_loader'].shuffle()
        for X, label in self._dataloader['train_loader'].get_iterator():
            self._optimizer.zero_grad()

            if self._task_index == 1:
                label = remap_labels_by_threshold(label=label, threshold=self._threshold)


            X, label = self._to_device(self._to_tensor([X, label]))
            pred = self.model(self._scaler.transform(X), label)


            pred = self._inverse_transform(pred)


            mask_value = torch.tensor(0)







            loss = self._loss_fn(pred, label, mask_value)

            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()

            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)

            self._iter_cnt += 1
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)


    def train(self):
        acc = 0
        self._logger.info('Start training!')

        wait = 0
        min_loss = np.inf
        for epoch in range(self._max_epochs):
            t1 = time.time()
            mtrain_loss, mtrain_mape, mtrain_rmse = self.train_batch()
            t2 = time.time()

            v1 = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse, mvalid_acc = self.evaluate('val')
            v2 = time.time()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            message = 'Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Valid ACC: {:.4f}, Train Time: {:.4f}s/epoch, Valid Time: {:.4f}s, LR: {:.4e}'
            self._logger.info(message.format(epoch + 1, mtrain_loss, mtrain_rmse, mtrain_mape,\
                                             mvalid_loss, mvalid_rmse, mvalid_mape, mvalid_acc,\
                                             (t2 - t1), (v2 - v1), cur_lr))

            if mvalid_loss < min_loss:
                self.save_model(self._save_path)
                self._logger.info('Val loss decrease from {:.4f} to {:.4f}'.format(min_loss, mvalid_loss))
                min_loss = mvalid_loss
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info('Early stop at epoch {}, loss = {:.6f}'.format(epoch + 1, min_loss))
                    break

        avg_test_mae, avg_test_rmse, avg_test_mape, avg_test_acc = self.evaluate('test')

        return avg_test_mae, avg_test_rmse, avg_test_mape, mtrain_loss, avg_test_acc


    def evaluate(self, mode):


        self.model.eval()

        preds = []
        labels = []
        with torch.no_grad():
            for X, label in self._dataloader[mode + '_loader'].get_iterator():


                if self._task_index == 1:
                    label = remap_labels_by_threshold(label=label, threshold=self._threshold)

                X, label = self._to_device(self._to_tensor([X, label]))
                pred = self.model(self._scaler.transform(X), label)

                pred = self._inverse_transform(pred)
                if self._task_index == 1:
                    pred = pred.argmax(dim=-1, keepdim=True).float()


                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)



        mask_value = torch.tensor(0)



        if mode == 'val':
            acc = 0
            mae = masked_mae(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            if self._task_index == 1:
                acc = accuracy(preds, labels, mask_value)
            return mae, mape, rmse, acc

        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []
            test_acc = []
            print('Check mask value', mask_value)
            for i in range(self.model.horizon):
                res = compute_all_metrics(preds[:,i,:], labels[:,i,:], mask_value, self._task_index)
                log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test ACC: {:.4f}'
                self._logger.info(log.format(i + 1, res[0], res[2], res[1], res[3]))
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])
                test_acc.append(res[3])
            log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test ACC: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape), np.mean(test_acc)))

        return np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape), np.mean(test_acc)
