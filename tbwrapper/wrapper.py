
from tensorboardX import SummaryWriter
import torch
import logging
import os
import json

# DEBUG > INFO > WARNING > ERROR > CRITICAL    
# logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

class wrapper():

    def __init__(self, path, name = None):
        self.path = path

        self.writer = SummaryWriter(log_dir=os.path.join(path, 'log/'))

        if name is not None:
            self.logger = logging.getLogger(name)
        else:
            self.logger = logging.getLogger(path)
        logFormatter = logging.Formatter("%(asctime)s : %(message)s")
        fileHandler = logging.FileHandler(os.path.join(path, 'log.txt'))
        fileHandler.setFormatter(logFormatter)
        self.logger.addHandler(fileHandler)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        self.logger.addHandler(consoleHandler)

    def save_configue(self, config):
        with open(os.path.join(path, 'config.json'), 'w') as fout:
            json.dump(vars(config), fout)

    def set_level(self, level = 'debug'):
        level_dict = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING, 'error': logging.ERROR, 'critical': logging.CRITICAL}
        self.logger.setLevel(level_dict[level])

    def get_logger(self):
        return self.logger

    def add_loss_vs_batch(self, kv_dict, batch_index, print=True):
        for k, v in kv_dict.items():
            self.writer.add_scalar('loss_tracking/' + k, v, batch_index)
            if print:
                self.logger.info("%s : %s", k, v)

    def add_model_parameter_stats(self, model, batch_index, save=False):
        if save:
            self.param_updates = {name: param.detach().cpu().clone() for name, param in model.named_parameters()}
            self.param_updates_batchindex = batch_index

        for name, param in model.named_parameters():

            self.writer.add_scalar("parameter_value_mean/" + name, param.data.mean(),batch_index)
            self.writer.add_scalar("parameter_value_std/" + name, param.data.std(), batch_index)

            if param.requires_grad and param.grad is not None:
                    if param.grad.is_sparse:
                        grad_data = param.grad.data._values()
                    else:
                        grad_data = param.grad.data
                    self.writer.add_scalar("parameter_gradient_mean/" + name, grad_data.mean(), batch_index)
                    self.writer.add_scalar("parameter_gradient_norm/" + name, torch.norm(grad_data), batch_index)
                    self.writer.add_scalar("parameter_gradient_std/" + name, grad_data.std(), batch_index)

    def add_model_update_stats(self, model, batch_index):

        assert(self.param_updates_batchindex == batch_index)

        for name, param in model.named_parameters():
            if param.requires_grad:
                param_updates[name].sub_(param.detach().cpu())
                update_norm = torch.norm(param_updates[name].view(-1, ))
                param_norm = torch.norm(param.view(-1, )).cpu()
                self.writer.add_scalar("parameter_gradient_update/" + name, update_norm / (param_norm + 1e-7), batch_index)

    def add_model_parameter_histograms(self, model, batch_index):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._tensorboard.add_train_histogram("parameter_histogram/" + name, param, batch_index)