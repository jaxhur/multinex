import datetime
import logging
import time
from collections import deque

from .dist_util import get_dist_info, master_only

initialized_logger = {}


class MessageLogger():
    """Message logger for printing.

    Args:
        opt (dict): Config. It contains the following keys:
            name (str): Exp name.
            logger (dict): Contains ``print_freq`` and an optional
                ``eta_window`` measured in logging intervals.
            train (dict): Contains 'total_iter' (int) for total iters.
            use_tb_logger (bool): Use tensorboard logger.
        start_iter (int): Start iter. Default: 1.
        tb_logger (obj:`tb_logger`): Tensorboard logger. Default： None.
    """

    def __init__(self, opt, start_iter=1, tb_logger=None):
        self.exp_name = opt['name']
        self.interval = opt['logger']['print_freq']
        self.start_iter = start_iter
        self.max_iters = opt['train']['total_iter']
        self.use_tb_logger = opt['logger']['use_tb_logger']
        self.tb_logger = tb_logger
        self.start_time = time.perf_counter()
        self.last_log_time = self.start_time
        self.last_log_iter = start_iter

        # Keep enough log intervals to cover one validation cycle by default.
        # This smooths CUDA warm-up while retaining validation/checkpoint costs.
        val_freq = opt.get('val', {}).get('val_freq', self.interval)
        default_eta_window = max(
            1, int((val_freq + self.interval - 1) // self.interval))
        eta_window = int(opt['logger'].get('eta_window', default_eta_window))
        self.time_window = deque(maxlen=max(1, eta_window))
        self.logger = get_root_logger()

    @master_only
    def __call__(self, log_vars):
        """Format logging message.

        Args:
            log_vars (dict): It contains the following keys:
                epoch (int): Epoch number.
                iter (int): Current iter.
                lrs (list): List for learning rates.

                time (float): Iter time.
                data_time (float): Data time for each iter.
        """
        # epoch, iter, learning rates
        epoch = log_vars.pop('epoch')
        current_iter = log_vars.pop('iter')
        lrs = log_vars.pop('lrs')

        message = (
            f'[{self.exp_name[:5]}..][epoch:{epoch:3d}, ' f'iter:{current_iter:8,d}, lr:(')
        for v in lrs:
            message += f'{v:.3e},'
        message += ')] '

        # time and estimated time
        if 'time' in log_vars.keys():
            iter_time = log_vars.pop('time')
            data_time = log_vars.pop('data_time')

            current_time = time.perf_counter()
            elapsed_sec = current_time - self.start_time
            interval_iters = current_iter - self.last_log_iter
            if interval_iters > 0:
                interval_sec = current_time - self.last_log_time
                self.time_window.append((interval_sec, interval_iters))
                self.last_log_time = current_time
                self.last_log_iter = current_iter

            # A rolling wall-clock average drops one-time startup overhead but
            # still includes periodic logging, validation and checkpoint work.
            if self.time_window:
                window_sec = sum(sample[0] for sample in self.time_window)
                window_iters = sum(sample[1] for sample in self.time_window)
                time_sec_avg = window_sec / window_iters
            else:
                time_sec_avg = iter_time
            eta_sec = time_sec_avg * max(0, self.max_iters - current_iter)
            elapsed_str = str(
                datetime.timedelta(seconds=int(elapsed_sec)))
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            message += f'[elapsed: {elapsed_str}, eta: {eta_str}, '
            message += f'time (data): {iter_time:.3f} ({data_time:.3f})] '

        # other items, especially losses
        for k, v in log_vars.items():
            message += f'{k}: {v:.4e} '
            # tensorboard logger
            if self.use_tb_logger and 'debug' not in self.exp_name:
                if k.startswith('l_'):
                    self.tb_logger.add_scalar(f'losses/{k}', v, current_iter)
                else:
                    self.tb_logger.add_scalar(k, v, current_iter)
        self.logger.info(message)


@master_only
def init_tb_logger(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger


@master_only
def init_wandb_logger(opt):
    """We now only use wandb to sync tensorboard log."""
    import wandb
    logger = logging.getLogger('basicsr')

    project = opt['logger']['wandb']['project']
    resume_id = opt['logger']['wandb'].get('resume_id')
    if resume_id:
        wandb_id = resume_id
        resume = 'allow'
        logger.warning(f'Resume wandb logger with id={wandb_id}.')
    else:
        wandb_id = wandb.util.generate_id()
        resume = 'never'

    wandb.init(id=wandb_id, resume=resume,
               name=opt['name'], config=opt, project=project, sync_tensorboard=True)

    logger.info(f'Use wandb logger with id={wandb_id}; project={project}.')


def get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=None):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str): root logger name. Default: 'basicsr'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger_name in initialized_logger:
        return logger
    if logger_name != 'metric':
        format_str = '%(asctime)s %(levelname)s: %(message)s'
    else:
        format_str = ''
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        logger.setLevel(log_level)
        # add file handler
        if logger_name != 'metric':
            file_handler = logging.FileHandler(log_file, 'w')
        else:
            file_handler = logging.FileHandler(log_file, 'a')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    initialized_logger[logger_name] = True
    return logger


def get_env_info():
    """Get environment information.

    Currently, only log the software version.
    """
    import torch
    import torchvision

    from basicsr.version import __version__
    msg = r"""
                ____                _       _____  ____
               / __ ) ____ _ _____ (_)_____/ ___/ / __ \
              / __  |/ __ `// ___// // ___/\__ \ / /_/ /
             / /_/ // /_/ /(__  )/ // /__ ___/ // _, _/
            /_____/ \__,_//____//_/ \___//____//_/ |_|
     ______                   __   __                 __      __
    / ____/____   ____   ____/ /  / /   __  __ _____ / /__   / /
   / / __ / __ \ / __ \ / __  /  / /   / / / // ___// //_/  / /
  / /_/ // /_/ // /_/ // /_/ /  / /___/ /_/ // /__ / /<    /_/
  \____/ \____/ \____/ \____/  /_____/\____/ \___//_/|_|  (_)
    """
    msg += ('\nVersion Information: '
            f'\n\tBasicSR: {__version__}'
            f'\n\tPyTorch: {torch.__version__}'
            f'\n\tTorchVision: {torchvision.__version__}')
    return msg
