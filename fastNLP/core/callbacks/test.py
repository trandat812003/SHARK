__all__ = [
    "Callback",

    "GradientClipCallback",
    "WarmupCallback",
]

import os
import sys
from copy import deepcopy

import torch

# from .utils import _save_model

try:
    from tensorboardX import SummaryWriter
    
    tensorboardX_flag = True
except:
    tensorboardX_flag = False

try:
    import fitlog
except:
    pass


class Callback(object):
    r"""
    Callback是fastNLP中被设计用于增强 :class:`~fastNLP.Trainer` 的类。
    如果Callback被传递给了 Trainer , 则 Trainer 会在对应的阶段调用Callback的函数，
    具体调用时机可以通过 :mod:`trainer 模块<fastNLP.core.trainer>` 查看。
    这是Callback的基类，所有的callback必须继承自这个类

    """
    
    def __init__(self):
        super(Callback, self).__init__()
        self._trainer = None  # 在Trainer内部被重新赋值
        self._disabled = False

    def __repr__(self):
        return self.__class__.__name__

    @property
    def trainer(self):
        r"""
        该属性可以通过self.trainer获取到，一般情况下不需要使用这个属性。
        """
        return self._trainer

    @property
    def grad_scaler(self):
        r"""
        float16的gradient scaler
        """
        return self._trainer.grad_scaler

    @property
    def auto_cast(self):
        r"""
        float16用的auto cast环境
        """
        return self._trainer.auto_cast
    
    @property
    def step(self):
        r"""当前运行到的step, 范围为[1, self.n_steps+1)"""
        return self._trainer.step
    
    @property
    def n_steps(self):
        r"""Trainer一共会采多少个batch。当Trainer中update_every设置为非1的值时，该值不等于update的次数"""
        return self._trainer.n_steps
    
    @property
    def batch_size(self):
        r"""train和evaluate时的batch_size为多大"""
        return self._trainer.batch_size
    
    @property
    def epoch(self):
        r"""当前运行的epoch数，范围是[1, self.n_epochs+1)"""
        return self._trainer.epoch
    
    @property
    def n_epochs(self):
        r"""一共会运行多少个epoch"""
        return self._trainer.n_epochs
    
    @property
    def optimizer(self):
        r"""初始化Trainer时传递的Optimizer"""
        return self._trainer.optimizer
    
    @property
    def model(self):
        r"""正在被Trainer训练的模型"""
        return self._trainer.model
    
    @property
    def pbar(self):
        r"""如果在Callback中需要打印内容，请使用self.pbar.write(str)。否则可能出现命令行显示效果不太好的问题。在
        on_train_begin(), on_train_end(), on_exception()中请不要使用该属性，通过print输出即可。"""
        return self._trainer.pbar
    
    @property
    def update_every(self):
        r"""Trainer中的模型多少次反向传播才进行一次梯度更新，在Trainer初始化时传入的。"""
        return self._trainer.update_every
    
    @property
    def batch_per_epoch(self):
        r"""每个epoch一共有多少个batch，只有在on_epoch_begin之后才能调用该属性。"""
        return self._trainer.batch_per_epoch

    @property
    def is_master(self):
        return self._trainer.is_master

    @property
    def disabled(self):
        return self._disabled

    def on_train_begin(self):
        r"""
        在Train过程开始之前调用。

        :return:
        """
        pass

    
    def on_epoch_begin(self):
        r"""
        在每个epoch开始之前调用一次

        :return:
        """
        pass

    
    def on_batch_begin(self, batch_x, batch_y, indices):
        r"""
        每次采集到一个batch的数据则调用一次。这里对batch_x或batch_y删除添加内容是可以影响到Trainer中内容的。所以在这一步
        可以进行一些负采样之类的操作。batch_x和batch_y中的tensor已经被放置到了模型所在的设备上。

        :param dict batch_x: DataSet中被设置为input的field的batch。
        :param dict batch_y: DataSet中被设置为target的field的batch。
        :param list(int) indices: 这次采样使用到的indices，可以通过DataSet[indices]获取出这个batch采出的Instance，在一些
            情况下可以帮助定位是哪个Sample导致了错误。仅当num_workers=0时有效。
        :return:
        """
        pass

    
    def on_loss_begin(self, batch_y, predict_y):
        r"""
        在计算loss前调用，即这里修改batch_y或predict_y的值是可以影响到loss计算的。

        :param dict batch_y: 在DataSet中被设置为target的field的batch集合。
        :param dict predict_y: 模型的forward()返回的结果。
        :return:
        """
        pass

    
    def on_backward_begin(self, loss):
        r"""
        在loss得到之后，但在反向传播之前。可能可以进行loss是否为NaN的检查。

        :param torch.Tensor loss: 计算得到的loss值
        :return:
        """
        pass

    
    def on_backward_end(self):
        r"""
        反向梯度传播已完成，但由于update_every的设置，可能并不是每一次调用都有梯度。到这一步，还没有更新参数。

        :return:
        """
        pass

    
    def on_step_end(self):
        r"""
        到这里模型的参数已经按照梯度更新。但可能受update_every影响，并不是每次都更新了。

        :return:
        """
        pass

    
    def on_batch_end(self):
        r"""
        这一步与on_step_end是紧接着的。只是为了对称性加上了这一步。

        """
        pass

    
    def on_valid_begin(self):
        r"""
        如果Trainer中设置了验证，则发生验证前会调用该函数

        :return:
        """
        pass

    
    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        r"""
        每次执行验证集的evaluation后会调用。

        :param Dict[str: Dict[str: float]] eval_result: , evaluation的结果。一个例子为{'AccuracyMetric':{'acc':1.0}}，即
            传入的dict是有两层，第一层是metric的名称，第二层是metric的具体指标。
        :param str metric_key: 初始化Trainer时传入的metric_key。
        :param torch.Optimizer optimizer: Trainer中使用的优化器。
        :param bool is_better_eval: 当前dev结果是否比之前的好。
        :return:
        """
        pass

    
    def on_epoch_end(self):
        r"""
        每个epoch结束将会调用该方法
        """
        pass

    
    def on_train_end(self):
        r"""
        训练结束，调用该方法
        """
        pass

    
    def on_exception(self, exception):
        r"""
        当训练过程出现异常，会触发该方法
        :param exception: 某种类型的Exception，比如KeyboardInterrupt等
        """
        pass


class GradientClipCallback(Callback):
    r"""
    每次backward前，将parameter的gradient clip到某个范围。
    """
    
    def __init__(self, parameters=None, clip_value=1, clip_type='norm'):
        r"""
        
        :param None,torch.Tensor,List[torch.Tensor] parameters: 一般通过model.parameters()获得。
            如果为None则默认对Trainer的model中所有参数进行clip
        :param float clip_value: 将gradient 限制到[-clip_value, clip_value]。clip_value应该为正数
        :param str clip_type: 支持'norm', 'value'
            两种::
    
                1 'norm', 将gradient的norm rescale到[-clip_value, clip_value]
            
                2 'value', 将gradient限制在[-clip_value, clip_value],
                    小于-clip_value的gradient被赋值为-clip_value;
                    大于clip_value的gradient被赋值为clip_value.
        """
        super().__init__()
        
        from torch import nn
        if clip_type == 'norm':
            self.clip_fun = nn.utils.clip_grad_norm_
        elif clip_type == 'value':
            self.clip_fun = nn.utils.clip_grad_value_
        else:
            raise ValueError("Only supports `norm` or `value` right now.")
        if parameters is not None:
            self.parameters = list(parameters)
        else:
            self.parameters = None
        self.clip_value = clip_value



class WarmupCallback(Callback):
    r"""
    learning rate按照一定的速率从0上升到设置的learning rate。
    """
    def __init__(self, warmup=0.1, schedule='constant'):
        r"""
        
        :param int,float warmup: 如果warmup为int，则在该step之前，learning rate根据schedule的策略变化; 如果warmup为float，
            如0.1, 则前10%的step是按照schedule策略调整learning rate。
        :param str schedule: 以哪种方式调整。
            linear: 前warmup的step上升到指定的learning rate(从Trainer中的optimizer处获取的), 后warmup的step下降到0；
            constant前warmup的step上升到指定learning rate，后面的step保持learning rate.
        """
        super().__init__()
        self.warmup = max(warmup, 0.)

        self.initial_lrs = []  # 存放param_group的learning rate
        if schedule == 'constant':
            self.get_lr = self._get_constant_lr
        elif schedule == 'linear':
            self.get_lr = self._get_linear_lr
        else:
            raise RuntimeError("Only support 'linear', 'constant'.")


    def _get_constant_lr(self, progress):
        if progress<self.warmup:
            return progress/self.warmup
        return 1

    def _get_linear_lr(self, progress):
        if progress<self.warmup:
            return progress/self.warmup
        return max((progress - 1.) / (self.warmup - 1.), 0.)

    def on_train_begin(self):
        self.t_steps = (len(self.trainer.train_data) // (self.batch_size*self.update_every) +
                            int(len(self.trainer.train_data) % (self.batch_size*self.update_every)!= 0)) * self.n_epochs
        if self.warmup>1:
            self.warmup = self.warmup/self.t_steps
        self.t_steps = max(2, self.t_steps)  # 不能小于2
        # 获取param_group的初始learning rate
        for group in self.optimizer.param_groups:
            self.initial_lrs.append(group['lr'])

    def on_backward_end(self):
        if self.step%self.update_every==0:
            progress = (self.step/self.update_every)/self.t_steps
            for lr, group in zip(self.initial_lrs, self.optimizer.param_groups):
                group['lr'] = lr * self.get_lr(progress)


