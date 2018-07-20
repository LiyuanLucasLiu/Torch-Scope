
from tensorboardX import SummaryWriter
import torch

class wrapper():

    def __init__(self, path):
        self.writer = SummaryWriter(log_dir=path)

    def add_loss_vs_batch(self, kv_dict, batch_index):
        for k, v in kv_dict.items():
            writer.add_scalar('loss_tracking/' + k, v, batch_index)

    def add_model_parameter_stats(self, model, batch_index, save=False):
        if save:
            self.param_updates = {name: param.detach().cpu().clone() for name, param in model.named_parameters()}
            self.param_updates_batchindex = batch_index

        for name, param in model.named_parameters():

            self.writer.add_scalar("parameter_value_mean/" + name, param.data.mean(),batch_index)
            self.writer.add_scalar("parameter_value_std/" + name, param.data.std(), batch_index)

            if param.requires_grad and param.grad is not None:
                    if is_sparse(param.grad):
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