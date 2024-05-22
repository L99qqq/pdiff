import abc
import hydra
import torch

class BaseTask():
    def __init__(self, config, **kwargs):
        self.cfg = config
        self.task_data = self.init_task_data()

    def init_task_data(self):
        pass

    def build_model(self):
        # get from param_data or init from cfg
        model=hydra.utils.instantiate(self.cfg.task.model)
        if self.cfg.task.model_ckpt_path is not None:
            model = torch.load(self.cfg.task.model_ckpt_path)['model']
            # model.load_state_dict(checkpoint)
            for name, weights in model.named_parameters():
                if name not in self.cfg.task.train_layer:
                    weights.requires_grad = False
        return model

    def build_optimizer(self, net):
        # get from param_data or init from cfg
        return hydra.utils.instantiate(self.cfg.task.optimizer, net.parameters())

    def get_task_data(self):
        return self.task_data

    def get_param_data(self):
        return self.param_data

    @property
    def param_data(self):
        return self.set_param_data()

    @abc.abstractmethod
    def set_param_data(self):
        raise NotImplementedError

    @abc.abstractmethod
    def test_g_model(self, **kwargs):
        # test generation model
        pass

    @abc.abstractmethod
    def train_for_data(self):
        raise NotImplementedError