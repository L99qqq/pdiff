import pdb

import hydra.utils
import pytorch_lightning as pl
import torch
from typing import Any
import numpy as np
import torch.nn as nn

from .base import BaseSystem
from core.utils.ddpm import *
from core.utils.utils import *
from core.module.prelayer.latent_transformer import Param2Latent
from .ddpm import DDPM

# from torch.utils.tensorboard import SummaryWriter


class AE_DDPM(DDPM):
    def __init__(self, config, **kwargs):
        ae_model = hydra.utils.instantiate(
            config.system.ae_model
        )  # core.module.modules.encoder.big
        input_dim = config.system.ae_model.in_dim
        input_noise = torch.randn((1, input_dim))
        latent_dim = ae_model.encode(input_noise).shape
        config.system.model.arch.model.in_dim = latent_dim[-1] * latent_dim[-2]
        super(AE_DDPM, self).__init__(config)
        self.save_hyperparameters()
        self.split_epoch = self.train_cfg.split_epoch
        self.loss_func = nn.MSELoss()
        self.ae_model = ae_model
        
        
        if config.system.ae_model_cond2 is not None:
            ae_model_cond2 = hydra.utils.instantiate(
                config.system.ae_model_cond2
            )
            checkpoint = torch.load(config.system.ae_model_cond2_ckpt_path)['state_dict']
            checkpoint_ae_model={}
            for name,param in checkpoint.items():
                if name.startswith('ae_model'):
                    checkpoint_ae_model[".".join(name.split(".")[1:])]=param
            ae_model_cond2.load_state_dict(checkpoint_ae_model)
            ae_model_cond2.eval()
            self.ae_model_cond2=ae_model_cond2
        else:
            self.ae_model_cond2=ae_model
            
            
        self.check_seen_val_every_n_epoch = (
            config.system.train.check_seen_val_every_n_epoch
        )
        self.val_batch_size = config.system.datasets.val_batch_size
        # self.test_batch_size = config.system.datasets.test_batch_size
        torch.cuda.memory_summary(device=None, abbreviated=False)
        self.train_dataset_num=len(config.system.datasets.train_dataset)
        self.test_dataset_num=len(config.system.datasets.test_dataset)

    def ae_forward(self, batch, **kwargs):
        batch=batch[2]
        output = self.ae_model(batch)
        loss = self.loss_func(batch, output, **kwargs)
        # self.log('epoch', self.current_epoch)
        self.log(
            "ae_loss",
            loss.cpu().detach().mean().item(),
            prog_bar=True,
            logger=True,
        )
        return loss

    def training_step(self, batch, batch_idx, **kwargs):
        # if (
        #     self.current_epoch % self.check_seen_val_every_n_epoch == 0
        #     and batch_idx == 0
        # ):
        #     # 定时在seen train数据集上验证AE模型的效果
        #     dict = self.validation_on_seen_data(batch, batch_idx)
        #     for key, item in dict.items():
        #         self.log(key, item, logger=True)
        ddpm_optimizer, ae_optimizer = self.optimizers()
        if self.current_epoch < self.split_epoch:
            loss = self.ae_forward(batch, **kwargs)
            # self.writer.add_scalar('train_loss/ae_train_loss',loss,global_step=self.current_epoch)
            ae_optimizer.zero_grad()
            self.manual_backward(loss)
            ae_optimizer.step()
        else:
            loss = self.forward(batch, batch_idx, **kwargs)  # DDPM
            # self.writer.add_scalar('train_loss/ddpm_train_loss',loss,global_step=self.current_epoch)
            ddpm_optimizer.zero_grad()
            self.manual_backward(loss)
            ddpm_optimizer.step()

        if hasattr(self, "lr_scheduler"):
            self.lr_scheduler.step()
        return {"loss": loss}

    def pre_process(self, batch):
        latent = self.ae_model.encode(batch)
        self.latent_shape = latent.shape[-2:]
        return latent

    def post_process(self, outputs):
        # pdb.set_trace()
        outputs = outputs.reshape(-1, *self.latent_shape)
        return self.ae_model.decode(outputs)
        
    def pre_process_cond2(self, batch):
        latent = self.ae_model_cond2.encode(batch)
        self.latent_shape_cond2 = latent.shape[-2:]
        return latent

    def post_process_cond2(self, outputs):
        # pdb.set_trace()
        outputs = outputs.reshape(-1, *self.latent_shape_cond2)
        return self.ae_model_cond2.decode(outputs)

    def validation_step(self, batch, batch_idx, **kwargs: Any):
        if self.current_epoch < self.split_epoch:
            """
            AE reconstruction parameters
            """
            class_start = batch[0].cpu().tolist()
            class_end = batch[1].cpu().tolist()
            print("\n---------------------------------")
            print("Val the AE model on unseen train dataset:",end='')
            for i in range(len(class_start)):
                print(" {}-{}".format(class_start[i],class_end[i]),end=',')
            print()
            batch = batch[2]
            good_param = batch[: self.val_batch_size]
            input_accs = []
            for i, param in enumerate(good_param):
                acc, test_loss, output_list = self.task_func(param, class_start[i],class_end[i])
                input_accs.append(acc)
            print("\ninput model accuracy:{}".format(input_accs))
            input_maxs = []
            if len(input_accs)>=self.train_dataset_num:
                for i in range(0, len(input_accs), len(input_accs)//self.train_dataset_num):
                    input_maxs.append(max(input_accs[i:i+len(input_accs)//self.train_dataset_num]))
            else:
                for i in range(0, len(input_accs), 1):
                    input_maxs.append(max(input_accs[i:i+1]))
            print("input model best accuracy:", input_maxs)

            ae_rec_accs = []
            print("input shape:",good_param.shape)
            latent = self.ae_model.encode(good_param)
            print("latent shape:{}".format(latent.shape))
            ae_params = self.ae_model.decode(latent)
            mse_loss_on_val_dataset=self.loss_func(good_param, ae_params, **kwargs).to('cpu').item()
            print("ae params shape:{}".format(ae_params.shape))
            ae_params = ae_params.cpu()
            for i, param in enumerate(ae_params):
                param = param.to(batch.device)
                acc, test_loss, output_list = self.task_func(param, class_start[i],class_end[i])
                ae_rec_accs.append(acc)

            best_ae = max(ae_rec_accs)
            print(f"AE reconstruction models accuracy:{ae_rec_accs}")
            rec_maxs = []
            if len(ae_rec_accs)>=self.train_dataset_num:
                for i in range(0, len(ae_rec_accs), len(ae_rec_accs)//self.train_dataset_num):
                    rec_maxs.append(max(ae_rec_accs[i:i+len(ae_rec_accs)//self.train_dataset_num]))
            else:
                for i in range(0, len(ae_rec_accs), 1):
                    rec_maxs.append(max(ae_rec_accs[i:i+1]))
            print("AE reconstruction models best accuracy:", rec_maxs)
            precents=[rec_maxs[i]/input_maxs[i]*100 for i in range(len(rec_maxs))]
            mean_precents=np.mean(precents)
            self.log("mean_ae_rec_precents_on_val_dataset",mean_precents)
            print("AE mean reconstruction precent on val dataset:",mean_precents,"% (",precents,")")
            print("AE mse loss between rec and target on val dataset:",mse_loss_on_val_dataset)
            
            print("---------------------------------")
            self.log("best_ae_rec_acc_on_unseen_train_dataset", best_ae)
            self.log("mean_ae_rec_acc_on_unseen_train_dataset", np.mean(ae_rec_accs))
            self.log("mean_g_acc_on_unseen_train_dataset", -1.0)
            self.log("ae_mse_loss_between_rec_and_target_on_val_dataset:",mse_loss_on_val_dataset)
        else:
            dict = super(AE_DDPM, self).validation_step(batch, batch_idx, **kwargs)
            self.log("best_ae_rec_acc_on_unseen_train_dataset", -1.0)
            return dict

    # def validation_on_seen_data(self, batch, batch_idx):
    #     if self.current_epoch < self.split_epoch:
    #         """
    #         AE reconstruction parameters
    #         """
    #         original_name = batch[0]
    #         print("\n---------------------------------")
    #         print(
    #             "Val the AE model on seen train dataset:",
    #             batch[0][: self.val_batch_size],
    #         )
    #         # batch = batch[1].clone().detach().requires_grad_(False)
    #         batch = batch[1]
    #         good_param = batch[: self.val_batch_size]
    #         # print("\noriginal_name:", original_name)
    #         # print("good_param:", good_param)
    #         input_accs = []
    #         for i, param in enumerate(good_param):
    #             acc, test_loss, output_list = self.task_func(param, original_name[i])
    #             input_accs.append(acc)
    #         # print("original_name:",original_name)
    #         print("\ninput model accuracy:{}".format(input_accs))
    #         print("input model best accuracy:", max(input_accs))
    #         # print("input model best accuracy: {},{},{},{}".format(max(input_accs[:len(input_accs)//4]),max(input_accs[len(input_accs)//4:len(input_accs)//4*2]),max(input_accs[len(input_accs)//4*2:len(input_accs)//4*3]),max(input_accs[len(input_accs)//4*3:])))

    #         ae_rec_accs = []
    #         latent = self.ae_model.encode(good_param)
    #         print("latent shape:{}".format(latent.shape))
    #         ae_params = self.ae_model.decode(latent)
    #         print("ae params shape:{}".format(ae_params.shape))
    #         ae_params = ae_params.cpu()
    #         for i, param in enumerate(ae_params):
    #             param = param.to(batch.device)
    #             acc, test_loss, output_list = self.task_func(param, original_name[i])
    #             ae_rec_accs.append(acc)

    #         best_ae = max(ae_rec_accs)
    #         print(f"AE reconstruction models accuracy:{ae_rec_accs}")
    #         print(f"AE reconstruction models best accuracy:{best_ae}")
    #         # print('AE reconstruction models best accuracy:{},{},{},{}'.format(max(ae_rec_accs[:len(ae_rec_accs)//4]),max(ae_rec_accs[len(ae_rec_accs)//4:len(ae_rec_accs)//4*2]),max(ae_rec_accs[len(ae_rec_accs)//4*2:len(ae_rec_accs)//4*3]),max(ae_rec_accs[len(ae_rec_accs)//4*3:])))
    #         print("---------------------------------")
    #         # self.log("best_ae_rec_acc_on_seen_train_dataset", best_ae)
    #         # self.log("best_g_acc_on_unseen_train_dataset", -1.0)
    #         return {"best_ae_rec_acc_on_seen_train_dataset": best_ae}
    #     else:
    #         dict = super(AE_DDPM, self).validation_on_seen_data(batch, batch_idx)
    #         # self.log("best_ae_rec_acc_on_unseen_train_dataset", -1.0)
    #         return dict

    def test_first_batch(self, batch, batch_idx, **kwargs: Any):
        if self.current_epoch < self.split_epoch:
            """
            AE reconstruction parameters
            """
            class_start = batch[0].cpu().tolist()
            class_end = batch[1].cpu().tolist()     
            batch = batch[2].cuda()
            print("\n---------------------------------")
            print("Test the AE model on", end='')
            for i in range(len(class_start)):
                print(" {}-{}".format(class_start[i],class_end[i]),end=',')
            print()
            good_param = batch[: self.test_batch_size]
            input_accs = []
            for i, param in enumerate(good_param):
                acc, test_loss, output_list = self.task_func(param, class_start[i],class_end[i])
                input_accs.append(acc)
            print("\ninput model accuracy:{}".format(input_accs))
            # print("input model best accuracy:", max(input_accs))
            input_maxs = []
            if len(input_accs)>=self.test_dataset_num:
                for i in range(0, len(input_accs), len(input_accs)//self.test_dataset_num):
                    input_maxs.append(max(input_accs[i:i+len(input_accs)//self.test_dataset_num]))
            else:
                for i in range(0, len(input_accs), 1):
                    input_maxs.append(max(input_accs[i:i+1]))
            print("input model best accuracy:", input_maxs)

            ae_rec_accs = []
            print("input shape:",good_param.shape)
            latent = self.ae_model.encode(good_param)
            print("latent shape:{}".format(latent.shape))
            ae_params = self.ae_model.decode(latent)
            mse_loss_on_test_dataset=self.loss_func(good_param, ae_params, **kwargs).to('cpu').item()
            print("ae params shape:{}".format(ae_params.shape))
            ae_params = ae_params.cpu()
            for i, param in enumerate(ae_params):
                param = param.to(batch.device)
                acc, test_loss, output_list = self.task_func(param, class_start[i],class_end[i])
                ae_rec_accs.append(acc)

            best_ae = max(ae_rec_accs)
            print(f"AE reconstruction models accuracy:{ae_rec_accs}")
            # print(f"AE reconstruction models best accuracy:{best_ae}")
            rec_maxs = []
            if len(ae_rec_accs)>=self.test_dataset_num:
                for i in range(0, len(ae_rec_accs), len(ae_rec_accs)//self.test_dataset_num):
                    rec_maxs.append(max(ae_rec_accs[i:i+len(ae_rec_accs)//self.test_dataset_num]))
            else:
                for i in range(0, len(ae_rec_accs), 1):
                    rec_maxs.append(max(ae_rec_accs[i:i+1]))
            print("AE reconstruction models best accuracy:", rec_maxs)
            precents=[rec_maxs[i]/input_maxs[i]*100 for i in range(len(rec_maxs))]
            mean_precents=np.mean(precents)
            self.log("mean_ae_rec_precents_on_test_dataset",mean_precents)
            print("AE mean reconstruction precent on test dataset:",mean_precents,"% (",precents,")")
            print("AE mse loss between rec and target on test dataset",mse_loss_on_test_dataset)
            
            print("---------------------------------")
            self.log("best_ae_rec_acc_on_test_dataset", best_ae)
            self.log("mean_ae_rec_acc_on_test_dataset", np.mean(ae_rec_accs))
            self.log("ae_mse_loss_between_rec_and_target_on_test_dataset:",mse_loss_on_test_dataset)
            return {
                "best_g_acc_on_test_dataset": -1.0,
                "mean_g_acc_on_test_dataset": -1.0,
                "med_g_acc_on_test_dataset": -1.0,
            }
            
        else:
            dict = super(AE_DDPM, self).test_first_batch(batch, batch_idx, **kwargs)
            return dict

    def configure_optimizers(self, **kwargs):
        ae_parmas = self.ae_model.parameters()
        ddpm_params = self.model.parameters()

        self.ddpm_optimizer = hydra.utils.instantiate(
            self.train_cfg.optimizer, ddpm_params
        )
        self.ae_optimizer = hydra.utils.instantiate(self.train_cfg.optimizer, ae_parmas)

        if "lr_scheduler" in self.train_cfg and self.train_cfg.lr_scheduler is not None:
            self.lr_scheduler = hydra.utils.instantiate(self.train_cfg.lr_scheduler)

        return self.ddpm_optimizer, self.ae_optimizer
