"""
.. module:: wrapper
    :synopsis: wrapper
.. moduleauthor:: Liyuan Liu
"""

from tensorboardX import SummaryWriter
import torch
import logging
import os
import sys
import json

import git
import subprocess

import random
import numpy

class wrapper():
    """
    
    Toolkit wrapper for experiments based on pytorch. 

    This package has three features:

    1. Tracking environments, dependency, implementations and checkpoints;
    2. Logger wrapper with two handlers;
    3. tensorboard wrapper

    Parameters
    ----------
    path : ``str``, required.
        Output path for logger, checkpoint, ...
    name : ``str``, optional, (default=path).
        Name for the experiment,
    seed: ``int``, optional.
        The random seed (would be random generated if not provided).
    enable_git_track: ``bool``, optional
        If True, track the implementation with git (would automatically commit tracked files).
    checkpoints_to_keep : ``int``, optional, (default=1).
        Number of checkpoints.
    """
    def __init__(self, 
                path: str, 
                name: str = None,
                seed: int = None,
                enable_git_track: bool = False,
                checkpoints_to_keep: int = 1):

        self.path = path
        self.checkpoints_to_keep = checkpoints_to_keep
        self.counter = 0

        self.writer = SummaryWriter(log_dir=os.path.join(path, 'log/'))

        if name is not None:
            self.name = name
            self.logger = logging.getLogger(name)
        else:
            self.name = path
            self.logger = logging.getLogger(path)

        logFormatter = logging.Formatter("%(asctime)s : %(message)s")
        fileHandler = logging.FileHandler(os.path.join(path, 'log.txt'))
        fileHandler.setFormatter(logFormatter)
        self.logger.addHandler(fileHandler)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        self.logger.addHandler(consoleHandler)

        if seed is None:
            seed = random.randint(1, 10000)
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        environments = {
            "PATH": path,
            "RANDOM SEED": seed,
            "SYS ENVIRONMENT:" vars(os.environ),
            "COMMAND": sys.argv, 
            "INSTALLED PACKAGES": subprocess.check_output(["pip", "freeze"], universal_newlines=True).strip()
        }

        if enable_git_track:
            self.logger.info("setting up git tracker")
            repo = git.Repo('.', search_parent_directories=True)
            self.logger.debug("git root path: %s", repo.git.rev_parse("--show-toplevel"))
            self.logger.debug("git branch: %s", repo.active_branch.name)

            if repo.is_dirty():
                repo.git.add(u=True)
                repo.git.commit(m='experiment checkpoint for: {}'.format(self.name))

            self.logger.debug("git commit: %s", repo.head.commit.hexsha)
            
            environments['GIT HEAD COMMIT'] = repo.head.commit.hexsha

        with open(os.path.join(self.path, 'environ.json'), 'w') as fout:
            json.dump(environments, fout)
            
    def save_configue(self, config):
        """
        Save config dict to the ``config.json`` under the path.

        Parameters
        ----------
        config: required.
            config file (supporting dict, Namespace, ...)
        """
        if type(config) is not dict:
            config = vars(config)

        with open(os.path.join(self.path, 'config.json'), 'w') as fout:
            json.dump(config, fout)

    def save_checkpoint(self, 
                        model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer = None,
                        is_best: bool=False):
        """
        Save checkpoint under the path.

        Parameters
        ----------
        model: ``torch.nn.Module``, required.
            The model to be saved
        optimizer: ``torch.optim.Optimizer``, optional.
            The optimizer to be saved (if provided)
        is_best: bool, optional, (default=False).
            If set false, would only be saved as ``checkpoint_#counter.th``; otherwise, would also be saved as ``best.th``
        """
        s_dict = {'model': model.state_dict()}
        if optimizer is not None:
            s_dict['optimizer'] = optimizer.state_dict()

        if is_best:
            torch.save(s_dict, os.path.join(self.path, 'best.th'))

        torch.save(s_dict, os.path.join(self.path, 'checkpoint_{}.th'.format(self.counter)))
        self.counter += 1
        if self.counter > self.checkpoints_to_keep:
            os.remove(os.path.join(self.path, 'checkpoint_{}.th'.format(self.counter - self.checkpoints_to_keep - 1)))

    @staticmethod
    def restore_latest_checkpoint(folder_path):
        """
        Restore the latest checkpoint.

        Parameters
        ----------
        folder_path: ``str``, required.
            Path to the folder contains checkpoints

        Returns
        -------
        A ``dict`` contains 'model' and 'optimizer' (if saved).
        """
        checkpoint_list = [cp for cp in os.listdir(folder_path) if 'checkpoint_' in cp]

        if len(checkpoint_list) == 0:
            return None

        latest_counter = max([int(filter(str.isdigit, cp)) for cp in checkpoint_list])

        return wrapper.restore_checkpoint(os.path.join(folder_path, 'checkpoint_{}.th'.format(latest_counter)))

    @staticmethod
    def restore_best_checkpoint(folder_path):
        """
        Restore the best checkpoint.

        Parameters
        ----------
        folder_path: ``str``, required.
            Path to the folder contains checkpoints

        Returns
        -------
        A ``dict`` contains 'model' and 'optimizer' (if saved).
        """
        return wrapper.restore_checkpoint(os.path.join(folder_path, 'best.th'))

    @staticmethod
    def restore_checkpoint(file_path):
        """
        Restore checkpoint.

        Parameters
        ----------
        folder_path: ``str``, required.
            Path to the checkpoint file

        Returns
        -------
        A ``dict`` contains 'model' and 'optimizer' (if saved).
        """
        return torch.load(file_path, map_location=lambda storage, loc: storage)

    def set_level(self, level = 'debug'):
        """
        set the level of logging.

        Parameters
        ----------
        level: ``str``, required.
            Setting level to one of ``debug``, ``info``, ``warning``, ``error``, ``critical``

        """
        level_dict = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING, 'error': logging.ERROR, 'critical': logging.CRITICAL}
        self.logger.setLevel(level_dict[level])

    def get_logger(self):
        """
        return the logger.

        """
        return self.logger

    def get_writer(self):
        """
        return the tensorboard writer.

        """
        return self.writer

    def add_loss_vs_batch(self, 
                        kv_dict: dict, 
                        batch_index: int, 
                        add_log: bool = True):
        """
        add loss to the ``loss_tracking`` section in the tensorboard.

        Parameters
        ----------
        kv_dict: ``dict``, required.
            Dictionary contains the key-value pair of losses (or metrics)
        batch_index: ``int``, required.
            Index of the added loss.
        add_log: ``bool``, optional, (default = True).
            whether to plot the information in the log.
        """
        for k, v in kv_dict.items():
            self.writer.add_scalar('loss_tracking/' + k, v, batch_index)
            if add_log:
                self.logger.info("%s : %s", k, v)

    def add_model_parameter_stats(self, 
                                    model: torch.nn.Module, 
                                    batch_index: int,
                                    save: bool = False):
        """
        add parameter stats to the ``parameter_*`` sections in the tensorboard.

        Parameters
        ----------
        model: ``torch.nn.Module``, required.
            The model to be tracked.
        batch_index: ``int``, required.
            Index of the model parameters stats.
        save: ``bool``, optional, (default = False).
            whether to save the model parameters (for the method ``add_model_update_stats``).
        """
        if save:
            self.param_updates = {name: param.clone().detach().cpu() for name, param in model.named_parameters()}
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

    def add_model_update_stats(self, 
                                model: torch.nn.Module, 
                                batch_index: int):
        """
        add parameter update stats to the ``parameter_gradient_update`` sections in the tensorboard.

        Parameters
        ----------
        model: ``torch.nn.Module``, required.
            The model to be tracked.
        batch_index: ``int``, required.
            Index of the model parameters updates.
        """

        assert(self.param_updates_batchindex == batch_index)

        for name, param in model.named_parameters():
            if param.requires_grad:
                param_updates[name].sub_(param.clone().detach().cpu())
                update_norm = torch.norm(param_updates[name].view(-1, ))
                param_norm = torch.norm(param.view(-1, )).cpu()
                self.writer.add_scalar("parameter_gradient_update/" + name, update_norm / (param_norm + 1e-7), batch_index)

    def add_model_parameter_histograms(self, 
                                        model: torch.nn.Module, 
                                        batch_index: int):
        """
        add parameter histogram in the tensorboard.

        Parameters
        ----------
        model: ``torch.nn.Module``, required.
            The model to be tracked.
        batch_index: ``int``, required.
            Index of the model parameters updates.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_train_histogram("parameter_histogram/" + name, param.clone().detach().cpu().data.numpy(), batch_index)