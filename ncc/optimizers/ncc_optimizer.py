# -*- coding: utf-8 -*-

import torch

from ncc.utils.gradient_clip import tf_clip
from ncc.utils.gradient_clip import fairseq_clip


class NccOptimizer(object):

    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args

    @classmethod
    def setup_optimizer(cls, args, params=None, **kwargs):
        return cls(args, params=params, **kwargs)

    @property
    def optimizer(self):
        """Return a torch.optim.optimizer.Optimizer instance."""
        if not hasattr(self, '_optimizer'):
            raise NotImplementedError
        if not isinstance(self._optimizer, torch.optim.Optimizer):
            raise ValueError('_optimizer must be an instance of torch.optim.Optimizer')
        return self._optimizer

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        raise NotImplementedError

    @property
    def params(self):
        """Return an iterable of the parameters held by the optimizer."""
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                yield p

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def __getstate__(self):
        return self._optimizer.__getstate__()

    def get_lr(self):
        """Return the current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def set_lr(self, lr):
        """
        Set the learning rate.
        lr_factor: different learning rates for modules
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr * param_group['lr_factor'] if param_group.get('lr_factor', None) is not None else lr

    def state_dict(self):
        """Return the optimizer's state dict."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        """Load an optimizer state dict.

        In general we should prefer the configuration of the existing optimizer
        instance (e.g., learning rate) over that found in the state_dict. This
        allows us to resume training from a checkpoint using a new set of
        optimizer args.
        """
        self.optimizer.load_state_dict(state_dict)

        if optimizer_overrides is not None and len(optimizer_overrides) > 0:
            # override learning rate, momentum, etc. with latest values
            for group in self.optimizer.param_groups:
                group.update(optimizer_overrides)

    def backward(self, loss):
        """Computes the sum of gradients of the given tensor w.r.t. graph leaves."""
        loss.backward()

    def all_reduce_grads(self, module):
        """Manually all-reduce gradients (if required)."""
        if hasattr(module, "all_reduce_grads"):
            module.all_reduce_grads()

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        for p in self.params:
            if p.grad is not None:
                if torch.is_tensor(c):
                    c = c.to(p.grad.device)
                p.grad.data.mul_(c)

    def clip_grad_norm(self, max_norm, aggregate_norm_fn=None):
        """Clips gradient norm."""
        return fairseq_clip.clip_grad_norm_(self.params, max_norm, aggregate_norm_fn)

    def tf_clip_grad_value(self, min_norm, max_norm, **kwargs):
        return tf_clip.clip_grad_value_(self.params, min_norm, max_norm)

    def tf_clip_by_norm(self, max_norm, **kwargs):
        return tf_clip.clip_by_norm_(self.params, max_norm)

    def tf_clip_by_average_norm(self, max_norm, **kwargs):
        return tf_clip.clip_by_average_norm_(self.params, max_norm)

    def tf_clip_by_global_norm(self, max_norm, **kwargs):
        return tf_clip.clip_by_global_norm_(self.params, max_norm)

    def step(self, closure=None):
        """Performs a single optimization step."""
        self.optimizer.step(closure)

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for p in self.params:
            p.grad = None
        self.optimizer.zero_grad()

    @property
    def supports_memory_efficient_fp16(self):
        if hasattr(self.optimizer, 'supports_memory_efficient_fp16'):
            return self.optimizer.supports_memory_efficient_fp16
        return False

    @property
    def supports_flat_params(self):
        """
        Whether the optimizer supports collapsing of the model
        parameters/gradients into a single contiguous Tensor.
        """
        if hasattr(self.optimizer, 'supports_flat_params'):
            return self.optimizer.supports_flat_params
        return False

    def average_params(self):
        pass
