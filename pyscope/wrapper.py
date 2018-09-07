"""
.. module:: wrapper
    :synopsis: wrapper
.. moduleauthor:: Liyuan Liu
"""
import os
import git
import sys
import json
import numpy
import torch
import shutil
import random
import logging
import subprocess
from typing import Dict
from tensorboardX import SummaryWriter
        
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
        # stream logger
        if name is not None:
            self.name = name
            self.logger = logging.getLogger(name)
        else:
            self.name = path
            self.logger = logging.getLogger(path)

        logFormatter = logging.Formatter("%(asctime)s : %(message)s", "%Y-%m-%d %H:%M:%S")
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        self.logger.addHandler(consoleHandler)
        self.logger.setLevel(logging.INFO)

        # check path
        if os.path.exists(path):
            self.logger.critical("Checkpoint Folder Already Exists: {}".format(path))
            self.logger.critical("Input 'yes' to confirm deleting this folder; or 'no' to exit.")
            delete_folder = False
            while not delete_folder:
                action = input("yes for delete or no for exit:").lower()
                if 'yes' == action:
                    shutil.rmtree(path)
                    delete_folder = True
                elif 'no' == action:
                    sys.exit()
                else:
                    self.logger.critical("Only 'yes' or 'no' are acceptable.")

        # file logger
        self.path = path
        self.checkpoints_to_keep = checkpoints_to_keep
        self.counter = 0

        self.writer = SummaryWriter(log_dir=os.path.join(path, 'log/'))
        fileHandler = logging.FileHandler(os.path.join(path, 'log.txt'))
        fileHandler.setFormatter(logFormatter)
        self.logger.addHandler(fileHandler)

        if seed is None:
            seed = random.randint(1, 10000)
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.logger.info("Saving up system environemnt and python packages")
        environments = {
            "PATH": path,
            "RANDOM SEED": seed,
            "SYS ENVIRONMENT": {k.decode('utf-8'): v.decode('utf-8') for k, v in os.environ._data.items()},
            "COMMAND": sys.argv, 
            "INSTALLED PACKAGES": subprocess.check_output(["pip", "freeze"], universal_newlines=True).strip()
        }

        if enable_git_track:
            self.logger.info("Setting up git tracker")
            repo = git.Repo('.', search_parent_directories=True)
            self.logger.debug("Git root path: %s", repo.git.rev_parse("--show-toplevel"))
            self.logger.debug("Git branch: %s", repo.active_branch.name)

            if repo.is_dirty():
                repo.git.add(u=True)
                repo.git.commit(m='experiment checkpoint for: {}'.format(self.name))

            self.logger.debug("Git commit: %s", repo.head.commit.hexsha)
            
            environments['GIT HEAD COMMIT'] = repo.head.commit.hexsha

        with open(os.path.join(self.path, 'environ.json'), 'w') as fout:
            json.dump(environments, fout)

    def confirm_an_empty_path(self, path):
        if os.path.exists(path):
            self.logger.critical("Checkpoint Folder Already Exists: {}".format(path))
            self.logger.critical("Input 'yes' to confirm deleting this folder; or 'no' to exit.")
            while True:
                action = input("yes for delete or no for exit:").lower()
                if 'yes' == action:
                    shutil.rmtree(path)
                    return True
                elif 'no' == action:
                    return False
                else:
                    self.logger.critical("Only 'yes' or 'no' are acceptable.")
        return True

    def nvidia_memory_map(self, add_log = True):
        """
        Get the current GPU memory usage.
        Based on https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4

        Parameters
        ----------
        add_log: ``bool``, optional, (default = True).
            Whether to add the information in the log.

        Returns
        -------
        Memory_map: ``Dict[int, str]``
            Keys are device ids as integers.
            Values are memory usage as integers in MB.
        """
        if "PCI_BUS_ID" != os.environ["CUDA_DEVICE_ORDER"]:
            self.logger.warning("It's recommended to set ``CUDA_DEVICE_ORDER`` \
                        to be ``PCI_BUS_ID`` by ``export CUDA_DEVICE_ORDER=PCI_BUS_ID``; \
                        otherwise, it's not guaranteed that the gpu index from \
                        pytorch to be consistent the ``nvidia-smi`` results. ")

        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                    '--format=csv,noheader'], encoding='utf-8')
        gpu_memory = result.strip().split('\n')
        gpu_memory_map = {x: y.split(',') for x, y in zip(range(len(gpu_memory)), gpu_memory)}

        if add_log:
            self.logger.info("GPU memory usages:")
            self.logger.info("GPU ID: Mem\t Utils")
            for k, v in gpu_memory_map.items():
                self.logger.info("GPU  {}: {}\t {}".format(k, v[0], v[1]))

        return gpu_memory_map

    def get_bytes(self, size, suffix = ''):
        """
        Convert other memory size to bytes

        Parameters
        ----------
        size: ``str``, required.
            The numeric part of the memory size.
        suffix: ``str``, optional (default='').
            The unit of the memory size.
        """
        size = float(size)

        if not suffix or suffix.isspace():
            return size
        
        size = int(size)
        suffix = suffix.lower()
        if suffix == 'kb' or suffix == 'kib':
            return size << 10
        elif suffix == 'mb' or suffix == 'mib':
            return size << 20
        elif suffix == 'gb' or suffix == 'gib':
            return size << 30

        self.logger.error("Suffix uncognized: {}".format(suffix))
        return False

    def auto_device(self, metrics='memory'):
        """
        Automatically choose the gpu (would return the gpu index with minimal used gpu memory).

        Parameters
        __________
        metrics: ``str``, optional, (default='memory').
            metric for gpu selection, supporting ``memory`` (used memory) and ``utils``.
        """
        assert (metrics == 'memory' or metrics == 'utils')

        if torch.cuda.is_available():
            memory_list = self.nvidia_memory_map()
            minimal_usage = float('inf')
            gpu_index = -1
            for k, v in memory_list.items():
                if 'memory' == metrics:
                    v = v[0].split()
                    v = self.get_bytes(v[0], v[1])
                else:
                    v = float(v[1].replace('%', ''))

                if v < minimal_usage:
                    minimal_usage = v
                    gpu_index = k
            self.logger.info("Recommended GPU Index: {}".format(gpu_index))
            return gpu_index
        else:
            return -1
            
    def save_configue(self, config, name='config.json'):
        """
        Save config dict to the ``config.json`` under the path.

        Parameters
        ----------
        config: ``dict``, required.
            Config file (supporting dict, Namespace, ...)
        name: ``str``, optional, (default = "config.json").
            Name for the configuration name.
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
        checkpoint: ``dict``.
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
        checkpoint: ``dict``.
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
        checkpoint: ``dict``.
            A ``dict`` contains 'model' and 'optimizer' (if saved).
        """
        return torch.load(file_path, map_location=lambda storage, loc: storage)

    def debug(self, *args, **kargs):
        """
        Add debug to logger
        """
        self.logger.debug(*args, **kargs)

    def info(self, *args, **kargs):
        """
        Add info to logger
        """
        self.logger.info(*args, **kargs)
    
    def warning(self, *args, **kargs):
        """
        Add warning to logger
        """
        self.logger.warning(*args, **kargs)
    
    def error(self, *args, **kargs):
        """
        Add error to logger
        """
        self.logger.error(*args, **kargs)
    
    def critical(self, *args, **kargs):
        """
        Add critical to logger
        """
        self.logger.critical(*args, **kargs)

    def set_level(self, level = 'debug'):
        """
        Set the level of logging.

        Parameters
        ----------
        level: ``str``, required.
            Setting level to one of ``debug``, ``info``, ``warning``, ``error``, ``critical``

        """
        level_dict = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING, 'error': logging.ERROR, 'critical': logging.CRITICAL}
        self.logger.setLevel(level_dict[level])

    def get_logger(self):
        """
        Return the logger.
        """
        return self.logger

    def get_writer(self):
        """
        Return the tensorboard writer.
        """
        return self.writer

    def close(self):
        """
        Close the tensorboard writer and the logger.
        """
        self.writer.close()
        self.logger.close()

    def add_loss_vs_batch(self, 
                        kv_dict: dict, 
                        batch_index: int, 
                        add_log: bool = True,
                        add_writer: bool = True):
        """
        Add loss to the ``loss_tracking`` section in the tensorboard.

        Parameters
        ----------
        kv_dict: ``dict``, required.
            Dictionary contains the key-value pair of losses (or metrics)
        batch_index: ``int``, required.
            Index of the added loss.
        add_log: ``bool``, optional, (default = True).
            Whether to print the information in the log.
        """
        for k, v in kv_dict.items():
            if add_writer:
                self.writer.add_scalar('loss_tracking/' + k, v, batch_index)
            if add_log:
                self.logger.info("%s : %s", k, v)

    def add_model_parameter_stats(self, 
                                    model: torch.nn.Module, 
                                    batch_index: int,
                                    save: bool = False):
        """
        Add parameter stats to the ``parameter_*`` sections in the tensorboard.

        Parameters
        ----------
        model: ``torch.nn.Module``, required.
            The model to be tracked.
        batch_index: ``int``, required.
            Index of the model parameters stats.
        save: ``bool``, optional, (default = False).
            Whether to save the model parameters (for the method ``add_model_update_stats``).
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
        Add parameter update stats to the ``parameter_gradient_update`` sections in the tensorboard.

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
        Add parameter histogram in the tensorboard.

        Parameters
        ----------
        model: ``torch.nn.Module``, required.
            The model to be tracked.
        batch_index: ``int``, required.
            Index of the model parameters updates.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram("parameter_histogram/" + name, param.clone().detach().cpu().data.numpy(), batch_index)