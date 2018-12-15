"""
.. module:: wrapper
    :synopsis: wrapper
.. moduleauthor:: Liyuan Liu
"""
import os
import git
import sys
import json
import time
import numpy
import torch
import shutil
import random
import logging
import subprocess
from typing import Dict
from tensorboardX import SummaryWriter

from torch_scope.sheet_writer import sheet_writer
from torch_scope.file_manager import cached_url

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"

COLORS = {
    'WARNING': WHITE,
    'INFO': GREEN,
    'DEBUG': BLUE,
    'CRITICAL': YELLOW,
    'ERROR': RED
}

def formatter_message(message, use_color = True):
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message

class ColoredFormatter(logging.Formatter):

    def __init__(self, msg, use_color = True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        msg = record.msg
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            msg_color = COLOR_SEQ % (30 + COLORS[levelname]) + msg + RESET_SEQ
            record.msg = msg_color
        return logging.Formatter.format(self, record)

logger = logging.getLogger(__name__)

consoleHandler = logging.StreamHandler()
FORMAT = "[$BOLD%(asctime)s$RESET] %(message)s"
COLOR_FORMAT = formatter_message(FORMAT, True)
color_formatter = ColoredFormatter(COLOR_FORMAT)
consoleHandler.setFormatter(color_formatter)
logging.getLogger().addHandler(consoleHandler)
logging.getLogger().setLevel(logging.INFO)

class basic_wrapper(object):
    """
    Light toolkit wrapper for experiments based on pytorch. 

    This class features all-static methods and supports:

    1. Checkpoint loading;
    2. Auto device selection.
    """
    @staticmethod
    def restore_configue(path, name='config.json'):
        """
        Restore the config dict.

        Parameters
        ----------
        path: ``str``, required.
            The path toward the folder.
        name: ``str``, optional, (default = "config.json").
            Name for the configuration name.
        """
        with open(os.path.join(path, name), 'r') as fin:
            config = json.load(fin)

        return config

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

        return basic_wrapper.restore_checkpoint(os.path.join(folder_path, 'checkpoint_{}.th'.format(latest_counter)))

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
        return basic_wrapper.restore_checkpoint(os.path.join(folder_path, 'best.th'))

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
        return torch.load(cached_url(file_path), map_location=lambda storage, loc: storage)

    @staticmethod
    def nvidia_memory_map(use_logger = True, gpu_index = None):
        """
        Get the current GPU memory usage.
        Based on https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4

        Parameters
        ----------
        use_logger: ``bool``, optional, (default = True).
            Whether to add the information in the log.
        gpu_index: ``int``, optional, (default = None).
            The index of the GPU for loggin. 

        Returns
        -------
        Memory_map: ``Dict[int, str]``
            Keys are device ids as integers.
            Values are memory usage as integers in MB.
        """
        if "CUDA_DEVICE_ORDER" not in os.environ or "PCI_BUS_ID" != os.environ["CUDA_DEVICE_ORDER"]:

            warn_info = "It's recommended to set ``CUDA_DEVICE_ORDER``" + \
                        "to be ``PCI_BUS_ID`` by ``export CUDA_DEVICE_ORDER=PCI_BUS_ID``;" + \
                        "otherwise, it's not guaranteed that the gpu index from" + \
                        "pytorch to be consistent the ``nvidia-smi`` results."

            logger.warning(warn_info)

        # result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
        #             '--format=csv,noheader'], encoding='utf-8')
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                    '--format=csv,noheader'])
        gpu_memory = result.decode().strip().split('\n')
        gpu_memory_map = {x: y.split(',') for x, y in zip(range(len(gpu_memory)), gpu_memory)}

        if use_logger:
            logger.info("GPU memory usages:")
            if not gpu_index:
                logger.info("GPU ID: Mem\t Utils")
                for k, v in gpu_memory_map.items():
                    logger.info("GPU  {}: {}\t {}".format(k, v[0], v[1]))
            else:
                logger.info("GPU {}: {} (Used Memory)\t {} (Utils)".format(gpu_index, gpu_memory_map[gpu_index][0], gpu_memory_map[gpu_index][1]))

        return gpu_memory_map

    @staticmethod
    def get_bytes(size, suffix = ''):
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

        logger.error("Suffix uncognized: {}".format(suffix))

        return -1

    @staticmethod
    def auto_device(metrics='memory', use_logger = True, required_minimal = None, wait_time = 20):
        """
        Automatically choose the gpu (would return the gpu index with minimal used gpu memory by default).

        Parameters
        __________
        metrics: ``str``, optional, (default='memory').
            metric for gpu selection, supporting ``memory`` (used memory) and ``utils``.
        use_logger: ``bool``, optional, (default = True).
            Whether to add the information in the log.
        required_minimal: ``str``, optional, (default = None).
            Required resources for device selection (using the same criterior w. metrics, e.g., ``3000 Mb`` or ``30%``).
        wait_time: ``int``, optional, (default = 20).
            Interval in secs between two gpu usage retrival. 
        """
        assert (metrics == 'memory' or metrics == 'utils')

        if torch.cuda.is_available():
            memory_list = basic_wrapper.nvidia_memory_map()
            minimal_usage = float('inf')
            gpu_index = -1
            if required_minimal is None:
                required_minimal = -1
            else:
                if 'memory' == metrics:
                    required_minimal.split()
                    required_minimal = basic_wrapper.get_bytes(required_minimal[0], required_minimal[1])
                else:
                    required_minimal = float(required_minimal.replace('%', ''))

            while minimal_usage < required_minimal or minimal_usage == float('inf'):
                for k, v in memory_list.items():
                    if 'memory' == metrics:
                        v = v[0].split()
                        v = basic_wrapper.get_bytes(v[0], v[1])
                    else:
                        v = float(v[1].replace('%', ''))

                    if v < minimal_usage:
                        minimal_usage = v
                        gpu_index = k

                if minimal_usage < required_minimal:

                    logger.info("Not satisfying the required resource: {}".format(required_minimal))

                    time.sleep(wait_time)

            if use_logger:
                logger.info("Recommended GPU Index: {}".format(gpu_index))

            return gpu_index

        else:
            if use_logger:
                logger.info("CPU would be used.")
            return -1

class wrapper(basic_wrapper):
    """
    
    Toolkit wrapper for experiments based on pytorch. 

    This class has three features:

    1. Tracking environments, dependency, implementations and checkpoints;
    2. Logger wrapper with two handlers;
    3. Tensorboard wrapper;
    4. Auto device selection;

    Parameters
    ----------
    path : ``str``, required.
        Output path for logger, checkpoint, ...
        If set to ``None``, we would not create any file-writers.
    name : ``str``, optional, (default=path).
        Name for the experiment,
    seed: ``int``, optional.
        The random seed (would be random generated if not provided).
    enable_git_track: ``bool``, optional
        If True, track the implementation with git (would automatically commit tracked files).
    sheet_track_name: ``str``, optional, (default=None).
        The name of the google sheet for tracking metric.
    credential_path: ``str``, optional, (default=None).
        The path towards the credential file for tracking with google sheet.
    checkpoints_to_keep : ``int``, optional, (default=1).
        Number of checkpoints.
    """
    def __init__(self, 
                path: str, 
                name: str = None,
                seed: int = None,
                enable_git_track: bool = False,
                sheet_track_name: str = None,
                credential_path: str = None,
                checkpoints_to_keep: int = 1):
    
        if name is not None:
            self.name = name
        elif path is not None:
            self.name = path
        else:
            self.name = "Logger"

        self.path = path

        if path is not None:

            self.checkpoints_to_keep = checkpoints_to_keep
            self.counter = 0
            # check path
            if os.path.exists(path):
                logger.critical("Checkpoint Folder Already Exists: {}".format(path))
                logger.critical("Input 'yes' to confirm deleting this folder; or 'no' to exit.")
                delete_folder = False
                while not delete_folder:
                    action = input("yes for delete or no for exit: ").lower()
                    if 'yes' == action:
                        shutil.rmtree(path)
                        delete_folder = True
                    elif 'no' == action:
                        sys.exit()
                    else:
                        logger.critical("Only 'yes' or 'no' are acceptable.")

            self.writer = SummaryWriter(log_dir=os.path.join(path, 'log/'))
            fileHandler = logging.FileHandler(os.path.join(path, 'log.txt'))
            logFormatter = logging.Formatter("[%(asctime)s]: %(message)s", "%Y-%m-%d %H:%M:%S")
            fileHandler.setFormatter(logFormatter)
            # logging.getLogger().addHandler(fileHandler)
            logger.addHandler(fileHandler)

            if seed is None:
                seed = random.randint(1, 10000)
            random.seed(seed)
            numpy.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            logger.info("Saving system environemnt and python packages")
            environments = {
                "PATH": path,
                "RANDOM SEED": seed,
                "SYS ENVIRONMENT": {k.decode('utf-8'): v.decode('utf-8') for k, v in os.environ._data.items()},
                "COMMAND": sys.argv, 
                "INSTALLED PACKAGES": subprocess.check_output(["pip", "freeze"], universal_newlines=True).strip()
            }

            if enable_git_track:
                logger.info("Setting up git tracker")
                repo = git.Repo('.', search_parent_directories=True)
                logger.debug("Git root path: %s", repo.git.rev_parse("--show-toplevel"))
                logger.debug("Git branch: %s", repo.active_branch.name)

                if repo.is_dirty():
                    repo.git.add(u=True)
                    repo.git.commit(m='experiment checkpoint for: {}'.format(self.name))

                logger.debug("Git commit: %s", repo.head.commit.hexsha)
                
                environments['GIT HEAD COMMIT'] = repo.head.commit.hexsha

            if sheet_track_name:
                root_path, folder_name = os.path.split(path + '/')
                root_path, folder_name = os.path.split(root_path)
                self.sw = sheet_writer(sheet_track_name, root_path, folder_name, credential_path)
            else:
                self.sw = None

            with open(os.path.join(self.path, 'environ.json'), 'w') as fout:
                json.dump(environments, fout)
        else:
            self.writer = None
            self.sw = None

    def restore_configue(self, name='config.json'):
        """
        Restore the config dict.

        Parameters
        ----------
        name: ``str``, optional, (default = "config.json").
            Name for the configuration name.
        """
        assert(self.path is not None)
        with open(os.path.join(self.path, name), 'r') as fin:
            config = json.load(fin)

        return config

    def restore_latest_checkpoint(self, folder_path = None):
        """
        Restore the latest checkpoint.

        Parameters
        ----------
        folder_path: ``str``, optional, (default = None).
            Path to the folder contains checkpoints

        Returns
        -------
        checkpoint: ``dict``.
            A ``dict`` contains 'model' and 'optimizer' (if saved).
        """
        if not folder_path:
            assert(self.path is not None)
            folder_path = self.path
        return basic_wrapper.restore_latest_checkpoint(folder_path)

    def restore_best_checkpoint(self, folder_path = None):
        """
        Restore the best checkpoint.

        Parameters
        ----------
        folder_path: ``str``, optional, (default = None).
            Path to the folder contains checkpoints

        Returns
        -------
        checkpoint: ``dict``.
            A ``dict`` contains 'model' and 'optimizer' (if saved).
        """
        if not folder_path:
            assert(self.path is not None)
            folder_path = self.path
        return basic_wrapper.restore_best_checkpoint(folder_path)

    def nvidia_memory_map(self, use_logger = True, gpu_index = None):
        """
        Get the current GPU memory usage.
        Based on https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4

        Parameters
        ----------
        use_logger: ``bool``, optional, (default = True).
            Whether to add the information in the log.
        gpu_index: ``int``, optional, (default = None).
            The index of the GPU for loggin. 

        Returns
        -------
        Memory_map: ``Dict[int, str]``
            Keys are device ids as integers.
            Values are memory usage as integers in MB.
        """
        return basic_wrapper.nvidia_memory_map(use_logger = use_logger, gpu_index = gpu_index)

    def get_bytes(size, suffix = ''):
        """
        Convert other memory size to bytes

        Parameters
        ----------
        size: ``str``, required.
            The numeric part of the memory size.
        suffix: ``str``, optional (default='').
            The unit of the memory size.
        """
        return basic_wrapper.get_bytes(size, suffix = suffix)

    def auto_device(self, metrics='memory', use_logger = True, required_minimal = None, wait_time = 20):
        """
        Automatically choose the gpu (would return the gpu index with minimal used gpu memory by default).

        Parameters
        __________
        metrics: ``str``, optional, (default='memory').
            metric for gpu selection, supporting ``memory`` (used memory) and ``utils``.
        use_logger: ``bool``, optional, (default = True).
            Whether to add the information in the log.
        required_minimal: ``str``, optional, (default = None).
            Required resources for device selection (using the same criterior w. metrics, e.g., ``3000 Mb`` or ``30%``).
        wait_time: ``int``, optional, (default = 20).
            Interval in secs between two gpu usage retrival. 
        """

        return basic_wrapper.auto_device(metrics = metrics, use_logger = use_logger, required_minimal = required_minimal, wait_time = wait_time)

    def confirm_an_empty_path(self, path):
        """
        Check whether a folder is an empty folder (not-exist).

        Parameters
        __________
        path: ``str``, required.
            Path to the target folder.
        """
        if os.path.exists(path):
            logger.critical("Checkpoint Folder Already Exists: {}".format(path))
            logger.critical("Input 'yes' to confirm deleting this folder; or 'no' to exit.")
            while True:
                action = input("yes for delete, ignore for ignore, or no for exit:").lower()
                if 'yes' == action:
                    shutil.rmtree(path)
                    return True
                elif 'ignore' == action:
                    return True
                elif 'no' == action:
                    return False
                else:
                    logger.critical("Only 'yes', 'ignore' and 'no' are acceptable.")
        return True

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
        assert (self.path is not None)
        if type(config) is not dict:
            config = vars(config)

        with open(os.path.join(self.path, name), 'w') as fout:
            json.dump(config, fout)

    def save_checkpoint(self, 
                        model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer = None,
                        is_best: bool=False,
                        s_dict: dict=None):
        """
        Save checkpoint under the path.

        Parameters
        ----------
        model: ``torch.nn.Module``, required.
            The model to be saved
        optimizer: ``torch.optim.Optimizer``, optional.
            The optimizer to be saved (if provided)
        is_best: bool, optional, (default=False).
            If set false, would only be saved as ``checkpoint_#counter.th``; otherwise, would also be saved as ``best.th``.
        s_dict: dict, optional, (default=None).
            Other necessay information for checkpoint tracking.
        """
        assert (self.path is not None)
        if not s_dict:
            s_dict = dict()
        s_dict['model'] = model.state_dict()

        if optimizer is not None:
            s_dict['optimizer'] = optimizer.state_dict()

        if is_best:
            torch.save(s_dict, os.path.join(self.path, 'best.th'))

        torch.save(s_dict, os.path.join(self.path, 'checkpoint_{}.th'.format(self.counter)))
        self.counter += 1
        if self.counter > self.checkpoints_to_keep:
            file_path = os.path.join(self.path, 'checkpoint_{}.th'.format(self.counter - self.checkpoints_to_keep - 1))
            if os.path.exists(file_path):
                os.remove(file_path)

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

    def add_description(self, description):
        """
        Add description for the experiment to the spreadsheet.

        Parameters
        ----------
        description: ``str``, required.
            Description for the experiment.
        """
        logger.info("Adding description: {}".format(description))
        if self.sw:
            msg = self.sw.add_description(description)
            if msg:
                logger.error(msg)
        else:
            logger.warning("No spreadsheet writer is availabel for adding description")

    def add_loss_vs_batch(self, 
                        kv_dict: dict, 
                        batch_index: int, 
                        use_logger: bool = True,
                        use_writer: bool = True,
                        use_sheet_tracker: bool = True):
        """
        Add loss to the ``loss_tracking`` section in the tensorboard.

        Parameters
        ----------
        kv_dict: ``dict``, required.
            Dictionary contains the key-value pair of losses (or metrics).
        batch_index: ``int``, required.
            Index of the added loss.
        use_logger: ``bool``, optional, (default = True).
            Whether to print the information in the log.
        use_sheet_tracker: ``bool``, optional, (default = True).
            Whether to use the sheet writer (when available).
        """
        for k, v in kv_dict.items():
            if use_writer:
                self.writer.add_scalar('loss_tracking/' + k, v, batch_index)
            if use_logger:
                logger.info("%s : %s", k, v)
            if use_sheet_tracker and self.sw:
                msg = self.sw.add_metric(k, v)
                if msg:
                    logger.error(msg)

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
        assert(self.writer is not None)
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
        assert(self.writer is not None)
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
        assert(self.writer is not None)
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram("parameter_histogram/" + name, param.clone().detach().cpu().data.numpy(), batch_index)