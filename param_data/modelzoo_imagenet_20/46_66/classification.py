import pdb

import hydra.utils

from .base_task import BaseTask
from core.data.vision_dataset import VisionData
from core.data.parameters import PData
from core.utils.utils import *
import torch.nn as nn
import datetime
from core.utils import *
import glob
import omegaconf
import json

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from models.ConvNet3 import count_parameters

import pytorch_lightning as pl
import re


class CFTask(BaseTask):
    def __init__(self, config, **kwargs):
        super(CFTask, self).__init__(config, **kwargs)
        # ----------只有task_training需要-------------
        self.train_loader = self.task_data.train_dataloader()
        self.eval_loader = self.task_data.val_dataloader()
        self.test_loader = self.task_data.test_dataloader()
        
        # ----------只有train_p_diff需要---------------
        self.sub_task_dataloader = {}
        # print(self.task_data)
        # print(config)
        # print(self.cfg)
        for key in config.system.datasets.train_dataset.keys():
            class_start, class_end = re.findall(r"(\d+)_(\d+)$", key)[-1]
            class_start, class_end=int(class_start), int(class_end)
            self.sub_task_dataloader[(class_start, class_end)] = VisionData(
                self.cfg.task.data, key
            ).test_dataloader()
        for key in config.system.datasets.test_dataset.keys():
            class_start, class_end = re.findall(r"(\d+)_(\d+)$", key)[-1]
            class_start, class_end=int(class_start), int(class_end)
            # print("classification key:",key)
            self.sub_task_dataloader[(class_start, class_end)] = VisionData(
                self.cfg.task.data, key
            ).test_dataloader()

    def init_task_data(self):
        # print(self.cfg)
        # 这里李雅萍改了，但是无法运行
        # return VisionData(self.cfg.task.data)
        return VisionData(self.cfg.task.data)

    # override the abstract method in base_task.py
    def set_param_data(self):
        param_data = PData(self.cfg)
        # self.model = param_data.get_model()
        # self.train_layer = param_data.get_train_layer()
        return param_data

    def test_g_model(self, input, class_start, class_end):
        net = self.param_data.get_model(class_start,class_end)
        train_layer = self.param_data.get_train_layer(class_start,class_end)

        param = input
        target_num = 0
        for name, module in net.named_parameters():
            if name in train_layer:
                target_num += torch.numel(module)
        params_num = torch.squeeze(param).shape[0] 
        # print("params_num",params_num)
        # assert (target_num == params_num)
        param = torch.squeeze(param)
        if self.cfg.task.channels=='all':
            model = partial_reverse_tomodel(param, net, train_layer).to(param.device)
        else:
            model = partial_reverse_tomodel_for_part_conv(param, net, train_layer,self.cfg.task.channels).to(param.device)

        model.eval()
        
        # print("after model.eval")
        test_loss = 0
        correct = 0
        total = 0
        output_list = []

        test_loader = self.sub_task_dataloader[(class_start, class_end)]

        with torch.no_grad():
            for data, target in test_loader:
                # print("CFTask in testloader")
                data, target = data.cuda(), target.cuda()
                output = model(data)
                _,indices=torch.max(output, dim=1)
                target = target.to(torch.int64)       
                test_loss += F.cross_entropy(
                    output, target, size_average=False
                ).item()  # sum up batch loss

                total += data.shape[0]
                pred = torch.max(output, 1)[1]
                # print("CFTask model pred:", pred)
                output_list += pred.cpu().numpy().tolist()
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= total
        acc = 100.0 * correct / total
        del model
        return acc, test_loss, output_list

    # 没有用到val_g_model
    # def val_g_model(self, input):
    #     # net = self.model
    #     # train_layer = self.train_layer
    #     net = self.param_data.get_model(original_name)
    #     train_layer = self.param_data.get_train_layer(original_name)

    #     param = input
    #     target_num = 0
    #     for name, module in net.named_parameters():
    #         if name in train_layer:
    #             target_num += torch.numel(module)
    #     params_num = torch.squeeze(param).shape[0]  # + 30720
    #     assert target_num == params_num
    #     param = torch.squeeze(param)
    #     model = partial_reverse_tomodel(param, net, train_layer).to(param.device)

    #     model.eval()
    #     test_loss = 0
    #     correct = 0
    #     total = 0

    #     output_list = []

    #     with torch.no_grad():
    #         for data, target in self.train_loader:
    #             data, target = data.cuda(), target.cuda()
    #             output = model(data)
    #             target = target.to(torch.int64)
    #             test_loss += F.cross_entropy(
    #                 output, target, size_average=False
    #             ).item()  # sum up batch loss

    #             total += data.shape[0]
    #             pred = torch.max(output, 1)[1]
    #             output_list += pred.cpu().numpy().tolist()
    #             correct += pred.eq(target.view_as(pred)).sum().item()

    #     test_loss /= total
    #     acc = 100.0 * correct / total
    #     del model
    #     return acc, test_loss, output_list

    # override the abstract method in base_task.py, you obtain the model data for generation
    
    def train_for_data(self):
        net = self.build_model()

        optimizer = self.build_optimizer(net)
        criterion = nn.CrossEntropyLoss()
        scheduler = hydra.utils.instantiate(self.cfg.task.lr_scheduler, optimizer)
        # all_epoch = self.cfg.all_epoch
        # start_save_epoch=self.cfg.start_save_epoch
        random_seed_num = self.cfg.task.random_seed_num
        each_seed_save_num = self.cfg.task.each_seed_save_num
        channels= self.cfg.task.channels

        best_acc = 0
        train_loader = self.train_loader
        eval_loader = self.eval_loader
        train_layer = self.cfg.task.train_layer

        if train_layer == "all":
            train_layer = [name for name, module in net.named_parameters()]
        
        # resnet18冻住不需要训练的层
        fix_partial_model(train_layer, net)
        
        # for name, weights in net.named_parameters():
        #     if name in ["fc.weight","fc.bias", "layer4.1.conv2.weight", "layer4.1.bn2.weight","layer4.1.bn2.bias"]:
        #         print(name, weights)

        data_path = getattr(self.cfg.task, "save_root", "param_data")

        # tmp_path = os.path.join(
        #     data_path,
        #     "tmp_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
        # )
        # tmp_path = os.path.join(data_path, 'tmp')
        tmp_path = os.path.join(
            data_path,
            "tmp_{}".format(self.cfg.output_dir.split("/")[-1]),
        )

        # save_name = self.cfg.task.param.data_root.split("/")[-2]
        save_name = self.cfg.task.param.data_root.rsplit('/',1)[0].split('/',1)[1]
        print(save_name)

        final_path = os.path.join(
            data_path,
            save_name,
        )

        os.makedirs(tmp_path, exist_ok=True)
        os.makedirs(final_path, exist_ok=True)

        save_model_accs = []
        parameters = []

        tensorboard_callback = TensorBoard(log_dir=os.path.join(final_path, "logs"))

        for i in range(0, random_seed_num):
            pl.seed_everything(i)
            print("start training the ", i, "th param")
            net = self.build_model()
            optimizer = self.build_optimizer(net)
            criterion = nn.CrossEntropyLoss()
            scheduler = hydra.utils.instantiate(self.cfg.task.lr_scheduler, optimizer)
            net = net.cuda()
            parameters = []
            start_save_flag = False
            save_num = 0
            current_epoch = 0
            accs = []

            with tf.summary.create_file_writer(
                os.path.join(final_path, "logs/seed_{}/epochs".format(i))
            ).as_default():
                while save_num < each_seed_save_num:
                    # 每轮训练前都要冻住
                    fix_partial_model(train_layer, net)
                    
                    # print('test_acc before epoch{}: {}'.format(current_epoch + 1,self.test(net, criterion, eval_loader)))
                    train_loss, train_acc = self.train(
                        net, criterion, optimizer, train_loader, current_epoch,set_grad=start_save_flag,train_list=train_layer
                    )
                    test_acc = self.test(net, criterion, eval_loader)   # 这里用了eval_loader，可能有问题
                    best_acc = max(test_acc, best_acc)
                    tf.summary.scalar("train_loss", train_loss, step=current_epoch)
                    tf.summary.scalar("train_acc", train_acc, step=current_epoch)
                    tf.summary.scalar("test_acc", test_acc, step=current_epoch)
                    accs.append(test_acc)
                    current_epoch += 1

                    # original: 在最近的20个epoch里，test acc的最大值和最小值之差小于1
                    if (
                        current_epoch > 20
                        and abs(max(accs[-20:-1]) - min(accs[-20:-1])) < 1
                        and start_save_flag == False
                    ) or current_epoch > 150:
                    
                    # for name, weights in net.named_parameters():
                    #     if name in ["fc.weight","fc.bias", "layer4.1.conv2.weight", "layer4.1.bn2.weight","layer4.1.bn2.bias"]:
                    #         print(name, weights)
                    
                    # if current_epoch > 150:
                        start_save_flag = True
                        print(abs(max(accs[-20:-1]) - min(accs[-20:-1])))
                        with tf.summary.create_file_writer(
                            os.path.join(
                                final_path, "logs/seed_{}/start_save_epoch".format(i)
                            )
                        ).as_default():
                            tf.summary.scalar("train_loss", train_loss, step=current_epoch)
                            tf.summary.scalar(
                                "train_acc", train_acc, step=current_epoch
                            )
                            tf.summary.scalar("test_acc", test_acc, step=current_epoch)

                    if start_save_flag:
                        if save_num == 0:
                            print("start saving the model")
                            torch.save(net, os.path.join(tmp_path, "whole_model.pth"))
                            # 除了train_layer以外的层不更新
                            fix_partial_model(train_layer, net)

                        print("use {} channels".format(channels))
                        if channels=='all':
                            parameters.append(state_part(train_layer, net))
                        else:
                            parameters.append(state_part_for_part_conv(train_layer, net,channels))
                        
                        save_model_accs.append(test_acc)
                        if len(parameters) == 10 or save_num == each_seed_save_num - 1:
                            torch.save(
                                parameters,
                                os.path.join(
                                    tmp_path,
                                    "p_data_{}.pt".format(
                                        datetime.datetime.now().strftime(
                                            "%Y-%m-%d_%H-%M-%S"
                                        )
                                    ),
                                ),
                            )
                            print("temp path add", len(parameters), " params")
                            parameters = []

                        save_num += 1

                    scheduler.step()

        print("training over")
        pl.seed_everything(self.cfg.seed)

        pdata = []
        for file in glob.glob(os.path.join(tmp_path, "p_data_*.pt")):
            buffers = torch.load(file)
            for buffer in buffers:
                param = []
                for key in buffer.keys():
                    if key in train_layer:
                        param.append(buffer[key].data.reshape(-1))
                param = torch.cat(param, 0)
                pdata.append(param)

        print("saving ", len(pdata), " number of params")
        batch = torch.stack(pdata)
        mean = torch.mean(batch, dim=0)
        std = torch.std(batch, dim=0)

        # check the memory of p_data
        useage_gb = get_storage_usage(tmp_path)
        print(f"path {tmp_path} storage usage: {useage_gb:.2f} GB")

        state_dic = {
            "pdata": batch.cpu().detach(),
            "mean": mean.cpu(),
            "std": std.cpu(),
            "model": torch.load(os.path.join(tmp_path, "whole_model.pth")),
            "train_layer": train_layer,
            "performance": save_model_accs,
            "cfg": config_to_dict(self.cfg),
        }

        torch.save(state_dic, os.path.join(final_path, "data.pt"))
        json_state = {"cfg": config_to_dict(self.cfg), "performance": save_model_accs}
        json.dump(json_state, open(os.path.join(final_path, "config.json"), "w"), indent=4, ensure_ascii=False)

        # copy the code file(the file) in state_save_dir
        shutil.copy(
            os.path.abspath(__file__),
            os.path.join(final_path, os.path.basename(__file__)),
        )

        # delete the tmp_path
        shutil.rmtree(tmp_path)
        print("data process over")
        return {"save_path": final_path}

    def train(self, net, criterion, optimizer, trainloader, epoch, set_grad=False,train_list=None):
        print("\nEpoch: %d" % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # if set_grad:
            #     for name, weights in net.named_parameters():
            #         if name in train_list and weights.grad is not None:
            #             weights.grad[self.cfg.task.channels:]=torch.zeros_like(weights.grad[self.cfg.task.channels:])
            
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        progress_bar(
            len(trainloader),
            len(trainloader),
            "Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)"
            % (train_loss / (len(trainloader) + 1), 100.0 * correct / total, correct, total),
        )

        # 获取 trainloader 中的批次数。
        num_batches = len(trainloader)

        return train_loss / num_batches, 100.0 * correct / total

    def test(self, net, criterion, testloader):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(testloader),
                "Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )
            return 100.0 * correct / total

