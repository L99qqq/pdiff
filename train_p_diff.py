import hydra
from omegaconf import DictConfig
from core.runner.runner import *


@hydra.main(config_path="configs", config_name="base", version_base='1.2')
def training_for_data(config: DictConfig):
    if config.mode == 'train':
        result = train_generation(config)
    elif config.mode == 'test':
        result = test_generation(config)

if __name__ == "__main__":
    training_for_data()


# import hydra
# from omegaconf import DictConfig
# import torch
# import torch.multiprocessing as mp
# from functools import partial
# from core.runner.runner import *


# @hydra.main(config_path="configs", config_name="base", version_base="1.2")
# def training_for_data(config: DictConfig):
#     if config.mode == "train":
#         if config.distributed:
#             mp.spawn(
#                 train_generation_distributed, nprocs=config.num_gpus, args=(config,)
#             )
#         else:
#             train_generation(config)
#     elif config.mode == "test":
#         test_generation(config)


# def train_generation_distributed(rank, config):
#     # 设置分布式环境
#     torch.cuda.set_device(rank)
#     dist.init_process_group(backend="nccl", init_method="env://")

#     # 分布式训练
#     train_generation(config)


# if __name__ == "__main__":
#     training_for_data()
