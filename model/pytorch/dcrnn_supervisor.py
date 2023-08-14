import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from lib import utils
from model.pytorch.dcrnn_model import DCRNNModel
from model.pytorch.loss import masked_mae_loss
from model.pytorch.loss import DistillKL
from model.pytorch.loss import L2_loss
import math
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from lib.metrics import masked_rmse_np, masked_mape_np, masked_mae_np

class DCRNNSupervisor:
    def __init__(self, adj_mx, **kwargs):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')

        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)

        # logging.
        self._log_dir = self._get_log_dir(kwargs)
        self._writer = SummaryWriter('runs/' + self._log_dir)

        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        # data set
        self._data = utils.load_dataset(**self._data_kwargs)
        self.standard_scaler = self._data['scaler']

        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(
            self._model_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(self._model_kwargs.get('horizon', 1))  # for the decoder

        # setup two model:model_t and model_s
        self.model_num=2
        dcrnn_model_s = DCRNNModel(adj_mx, self._logger, **self._model_kwargs)
        self.model0 = dcrnn_model_s.cuda() if torch.cuda.is_available() else dcrnn_model_s
        self._logger.info("Model_0 created")
        dcrnn_model_t = DCRNNModel(adj_mx, self._logger, **self._model_kwargs)
        self.model1 = dcrnn_model_t.cuda() if torch.cuda.is_available() else dcrnn_model_t
        self._logger.info("Model_1 created")
        self.alpha=0.9
        self._epoch_num = self._train_kwargs.get('epoch', 0)
        if self._epoch_num > 0:
            print(self._epoch_num)
            self.load_model()

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'dcrnn_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, epoch):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        config = dict(self._kwargs)
        config['model_state_dict'] = self.model0.state_dict()
        config['epoch'] = epoch
        torch.save(config, 'models/epo%d.tar' % epoch)
        self._logger.info("Saved model at {}".format(epoch))
        return 'models/epo%d.tar' % epoch

    def load_model(self):
        self._setup_graph()
        assert os.path.exists('models/epo%d.tar' % self._epoch_num), 'Weights at epoch %d not found' % self._epoch_num
        checkpoint = torch.load('models/epo%d.tar' % self._epoch_num, map_location='cpu')
        self.model0.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(self._epoch_num))

    def _setup_graph(self):
        with torch.no_grad():
            self.model0 = self.model0.eval()
            self.model1 = self.model1.eval()
            val_iterator = self._data['val_loader'].get_iterator()

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output = self.model0(x)
                break

    def train(self, **kwargs):

        kwargs.update(self._train_kwargs)
        return self._train(**kwargs)

    def evaluate(self, dataset='val', batches_seen=0):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.model0 = self.model0.eval()
            self.model1 = self.model1.eval()

            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
            losses0 = []
            losses1 = []
            losses = []
            y_truths = []
            y_preds = []
            y_preds1 = []

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)

                output_0,encoder_outputs_0= self.model0(x)
                output_1,encoder_outputs_1= self.model1(x)
                ce_loss0 = self._compute_loss(y, output_0)
                ce_loss1 = self._compute_loss(y, output_1)

                losses.append(ce_loss0.item())
                y_truths.append(y.cpu())
                y_preds.append(output_0.cpu())
            mean_loss = np.mean(losses)
            self._writer.add_scalar('{} loss'.format(dataset), mean_loss, batches_seen)

            y_preds = np.concatenate(y_preds, axis=1)
            y_truths = np.concatenate(y_truths, axis=1)  # concatenate on batch dimension
            y_truths_scaled = []
            y_preds_scaled = []

            for t in range(y_preds.shape[0]):
                y_truth = self.standard_scaler.inverse_transform(y_truths[t])
                y_pred = self.standard_scaler.inverse_transform(y_preds[t])
                mae = masked_mae_np(y_pred, y_truth, null_val=0)
                mape = masked_mape_np(y_pred, y_truth, null_val=0)
                rmse = masked_rmse_np(y_pred, y_truth, null_val=0)
                self._logger.info(
                    "Horizon {:02d}, MAE: {:.2f}, MAPE: {:.4f}, RMSE: {:.2f}".format(
                    t + 1, mae, mape, rmse
                    )
                )
                y_truths_scaled.append(y_truth)
                y_preds_scaled.append(y_pred)

            """self._logger.info("predict1")
            for t in range(y_preds1.shape[0]):
                y_truth = self.standard_scaler.inverse_transform(y_truths[t])
                y_pred1 = self.standard_scaler.inverse_transform(y_preds1[t])
                mae = masked_mae_np(y_pred1, y_truth, null_val=0)
                mape = masked_mape_np(y_pred1, y_truth, null_val=0)
                rmse = masked_rmse_np(y_pred1, y_truth, null_val=0)
                self._logger.info(
                    "Horizon {:02d}, MAE: {:.2f}, MAPE: {:.4f}, RMSE: {:.2f}".format(
                    t + 1, mae, mape, rmse
                    )
                )
                y_truths_scaled.append(y_truth)
                y_preds1_scaled.append(y_pred1)

            self._logger.info("predict mean")
            for t in range(y_preds_mean.shape[0]):
                y_truth = self.standard_scaler.inverse_transform(y_truths[t])
                y_pred_mean = self.standard_scaler.inverse_transform(y_preds_mean[t])
                mae = masked_mae_np(y_pred_mean, y_truth, null_val=0)
                mape = masked_mape_np(y_pred_mean, y_truth, null_val=0)
                rmse = masked_rmse_np(y_pred_mean, y_truth, null_val=0)
                self._logger.info(
                    "Horizon {:02d}, MAE: {:.2f}, MAPE: {:.4f}, RMSE: {:.2f}".format(
                        t + 1, mae, mape, rmse
                    )
                )
                y_truths_scaled.append(y_truth)
                y_preds_mean_scaled.append(y_pred_mean)

            self._logger.info("predict max")
            for t in range(y_preds_max.shape[0]):
                y_truth = self.standard_scaler.inverse_transform(y_truths[t])
                y_pred_max = self.standard_scaler.inverse_transform(y_preds_max[t])
                mae = masked_mae_np(y_pred_max, y_truth, null_val=0)
                mape = masked_mape_np(y_pred_max, y_truth, null_val=0)
                rmse = masked_rmse_np(y_pred_max, y_truth, null_val=0)
                self._logger.info(
                    "Horizon {:02d}, MAE: {:.2f}, MAPE: {:.4f}, RMSE: {:.2f}".format(
                        t + 1, mae, mape, rmse
                    )
                )
                y_truths_scaled.append(y_truth)
                y_preds_max_scaled.append(y_pred_max)"""

            return mean_loss, {'prediction': y_preds_scaled, 'truth': y_truths_scaled}


    def _train(self, base_lr,
               steps, patience=50, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=1,
               test_every_n_epochs=5, epsilon=1e-8, **kwargs):
        # steps is used in learning rate - will see if need to use it?
        min_val_loss = float('inf')
        wait = 0
        optimizer_0 = torch.optim.Adam(self.model0.parameters(), lr=base_lr, eps=epsilon)
        lr_scheduler_0 = torch.optim.lr_scheduler.MultiStepLR(optimizer_0, milestones=steps,
                                                            gamma=lr_decay_ratio)
        optimizer_1 = torch.optim.Adam(self.model1.parameters(), lr=base_lr, eps=epsilon)

        lr_scheduler_1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_1, milestones=steps,
                                                            gamma=lr_decay_ratio)
        self._logger.info('Start training ...')

        # this will fail if model is loaded with a changed batch_size
        num_batches = self._data['train_loader'].num_batch
        self._logger.info("num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num

        for epoch_num in range(self._epoch_num, epochs):
            self.model0= self.model0.train()
            self.model1 = self.model1.train()
            train_iterator = self._data['train_loader'].get_iterator()
            losses = []
            losses0 = []
            losses1 = []
            start_time = time.time()

            for _, (x, y) in enumerate(train_iterator):
                optimizer_0.zero_grad()
                optimizer_1.zero_grad()
                x, y = self._prepare_data(x, y)
                output_0,encoder_outputs_0= self.model0(x, y, batches_seen)
                output_1,encoder_outputs_1= self.model1(x, y, batches_seen)
                if batches_seen == 0:
                    # this is a workaround to accommodate dynamically registered parameters in DCGRUCell
                    optimizer_0 = torch.optim.Adam(self.model0.parameters(), lr=base_lr, eps=epsilon)
                    optimizer_1 = torch.optim.Adam(self.model1.parameters(), lr=base_lr, eps=epsilon)
                ce_loss0 = self._compute_loss(y, output_0)
                ce_loss1 = self._compute_loss(y, output_1)

                self_loss0=DistillKL(output_0.detach(),encoder_outputs_1.detach())
                self_loss1=DistillKL(output_1.detach(),encoder_outputs_0.detach())
                self._logger.debug(ce_loss0.item())
                self._logger.debug(ce_loss1.item())

                losses0.append(ce_loss0.item())
                losses1.append(ce_loss1.item())

                batches_seen += 1
                KD_loss0 = DistillKL(output_0, output_1.detach())
                KD_loss1 = DistillKL(output_1, output_0.detach())
                loss0 = ce_loss0 +(1-self.alpha)*self_loss0+self.alpha*KD_loss0
                loss1 = ce_loss1 +(1-self.alpha)*self_loss1+self.alpha*KD_loss1
                loss0.backward(retain_graph=True)
                loss1.backward()
                
                # gradient clipping - this does it in place
                torch.nn.utils.clip_grad_norm_(self.model0.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.model1.parameters(), self.max_grad_norm)
                optimizer_0.step()
                optimizer_1.step()

            self._logger.info("epoch complete")
            lr_scheduler_0.step()
            lr_scheduler_1.step()
            self._logger.info("evaluating now!")

            end_time = time.time()

            losses.append(losses0)
            self._writer.add_scalar('training loss',
                                    np.mean(losses),
                                    batches_seen)

            """self._writer.add_scalar('training loss',
                                    np.mean(losses[1]),
                                    batches_seen)"""


            if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1:
                test_loss, _ = self.evaluate(dataset='test', batches_seen=batches_seen)
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, test_mae: {:.4f},  lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(losses), test_loss, lr_scheduler_0.get_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message)
    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(device), y.to(device)

    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        """
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param y: shape (horizon, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        batch_size = x.size(1)
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        y = y[..., :self.output_dim].view(self.horizon, batch_size,
                                          self.num_nodes * self.output_dim)
        return x, y

    def _compute_loss(self, y_true, y_predicted):
        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        return masked_mae_loss(y_predicted, y_true)
