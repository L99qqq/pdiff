import pdb

import hydra.utils
import torch
import pytorch_lightning as pl
import sys
import os
import datetime
from core.tasks.classification import CFTask
from core.system import *
import torch
import torch.distributed as dist
from core.tasks import tasks
import time

from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

def set_seed(seed):
    pl.seed_everything(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_device(device_config):
    # set the global cuda device
    torch.backends.cudnn.enabled = True
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_config.cuda_visible_devices)
    torch.cuda.set_device(device_config.cuda)
    torch.set_float32_matmul_precision('medium')
    # warnings.filterwarnings("always")


def set_processtitle(cfg):
    # set process title
    import setproctitle
    setproctitle.setproctitle(cfg.process_title)

def init_experiment(cfg, **kwargs):
    cfg = cfg

    print("config:")
    for k, v in cfg.items():
        print(k, v)
    print("=" * 20)

    print("kwargs:")
    for k, v in kwargs.items():
        print(k, v)
    print("=" * 20)

    # set seed
    set_seed(cfg.seed)
    # set device
    set_device(cfg.device)

    # set process title
    set_processtitle(cfg)

from pytorch_lightning.callbacks import Callback

class SaveModuleCallback(Callback):
    def __init__(self,save_every_n_epoch,output_dir):
        super().__init__()
        self.save_every_n_epoch=save_every_n_epoch
        self.output_dir=output_dir
        self.save_path=os.path.join(output_dir,"saved_checkpoints")
        os.makedirs(self.save_path, exist_ok=True)
        
    def on_train_epoch_end(self, trainer, pl_module):
        if  trainer.current_epoch >0 and  trainer.current_epoch%self.save_every_n_epoch==0:
            trainer.save_checkpoint(os.path.join(self.save_path,"epoch:{}.ckpt".format(trainer.current_epoch)))
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        print("my on_save_checkpoint current_epoch:{}---------------------------------- ".format(trainer.current_epoch))
        # 收集父类模型的参数并保存到检查点字典中
        ddpm_state_dict = pl_module.model.model.state_dict()
        checkpoint['ddpm_state_dict'] = ddpm_state_dict
    
    def on_load_checkpoint(self,trainer, pl_module, checkpoint):
        print("my on_load_checkpoint current_epoch:{}----------------------------------".format(trainer.current_epoch))
        for key,_ in checkpoint.items():
            print(f"key name: {key}")
        pl_module.model.model.load_state_dict(checkpoint['ddpm_state_dict'])



class TestModuleCallback(Callback):

    def __init__(self, split_epoch, test_every_n_epoch, datamodule):
        super().__init__()
        self.split_epoch=split_epoch
        self.test_every_n_epoch=test_every_n_epoch
        self.datamodule = datamodule
        self.test_dataloader = self.datamodule.test_dataloader()

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.split_epoch:
            self.test_dataloader = self.datamodule.test_dataloader()
        if trainer.current_epoch % self.test_every_n_epoch == 0:
            batch = next(iter(self.test_dataloader))
            trainer.model.test_first_batch(batch, 0)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == self.split_epoch - 1:
            batch = next(iter(self.test_dataloader))
            trainer.model.test_first_batch(batch, 0)


def train_generation(cfg):
    init_experiment(cfg)
    system_cls = systems[cfg.system.name]   # ae_ddpm
    system = system_cls(cfg)
    datamodule = system.get_task().get_param_data() # PData

    # running
    trainer: Trainer = hydra.utils.instantiate(cfg.system.train.trainer)
    testcallback=TestModuleCallback(
        cfg.system.train.split_epoch,
        cfg.system.train.test_every_n_epoch,
        datamodule
    )
    savecallback=SaveModuleCallback(
        save_every_n_epoch=cfg.system.save_every_n_epoch,output_dir=cfg.output_dir
    )
    trainer.callbacks.append(testcallback)
    trainer.callbacks.append(savecallback)
    trainer.fit(system, datamodule=datamodule, ckpt_path=cfg.load_system_checkpoint)
    trainer.test(system, datamodule=datamodule)
    return {}

def test_generation(cfg):
    init_experiment(cfg)
    system_cls = systems[cfg.system.name]
    system = system_cls(cfg)
    datamodule = system.get_task().get_param_data()
    # running
    trainer: Trainer = hydra.utils.instantiate(cfg.system.train.trainer)
    savecallback=SaveModuleCallback(
        save_every_n_epoch=cfg.system.save_every_n_epoch,output_dir=cfg.output_dir
    )
    trainer.callbacks.append(savecallback)
    trainer.test(system, datamodule=datamodule, ckpt_path=cfg.load_system_checkpoint)

    return {}

def train_task_for_data(cfg, **kwargs):
    init_experiment(cfg, **kwargs)
    task_cls = tasks[cfg.task.name]
    task = task_cls(cfg, **kwargs)

    task_result = task.train_for_data()
    return task_result
