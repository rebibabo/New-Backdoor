# -*- coding: utf-8 -*-

import contextlib
import sys
from itertools import chain

import torch

from ncc import (
    models,
    optimizers,
    LOGGER,
)
from ncc.optimizers import lr_schedulers
from ncc.utils import (
    checkpoint_utils,
    distributed_utils,
)
from ncc.utils import utils
from ncc.utils.logging import meters, metrics
from ncc.utils.nan_detector import NanDetector
from ncc.utils.path_manager import PathManager
from ncc.utils.utils import move_to_cuda


class Trainer(object):
    """Main class for data parallel training.

    This class supports synchronous distributed data parallel training,
    where multiple workers each have a full model replica and gradients
    are accumulated across workers before each update. We use
    :class:`~torch.nn.parallel.DistributedDataParallel` to handle
    communication of the gradients across workers.
    """

    def __init__(self, args, task, model, criterion):
        self.args = args
        self.task = task

        self.cuda = torch.cuda.is_available() and not args['common']['cpu']
        if self.cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # copy model and criterion to current device
        self._criterion = criterion
        self._model = model
        if args['common']['fp16']:
            assert not args['common'].get('amp', False), "Cannot use fp16 and AMP together"
            self._criterion = self._criterion.half()
            self._model = self._model.half()
        elif args['common']['bf16']:
            self._criterion = self._criterion.to(dtype=torch.bfloat16)
            self._model = self._model.to(dtype=torch.bfloat16)
        elif args['common'].get('amp', False):
            self._amp_retries = 0

        self._criterion = self._criterion.to(device=self.device)
        self._model = self._model.to(device=self.device)

        self._dummy_batch = "DUMMY"  # indicates we don't have a dummy batch at first
        self._lr_scheduler = None
        self._num_updates = 0
        self._optim_history = None
        self._optimizer = None
        self._warn_once = set()
        self._wrapped_criterion = None
        self._wrapped_model = None

        if self.cuda and args['distributed_training']['distributed_world_size'] > 1:
            self._grad_norm_buf = torch.cuda.DoubleTensor(args['distributed_training']['distributed_world_size'])
        else:
            self._grad_norm_buf = None

        # get detailed cuda environment
        if self.cuda:
            self.cuda_env = utils.CudaEnvironment()
            if self.data_parallel_world_size > 1:
                self.cuda_env_arr = distributed_utils.all_gather_list(
                    self.cuda_env, group=distributed_utils.get_global_group()
                )
            else:
                self.cuda_env_arr = [self.cuda_env]
            if self.data_parallel_rank == 0:
                utils.CudaEnvironment.pretty_print_cuda_env_list(self.cuda_env_arr)
        else:
            self.cuda_env = None
            self.cuda_env_arr = None

        metrics.log_start_time("wall", priority=790, round=0)

    def reinitialize(self):
        """Reinitialize the Trainer, typically after model params change."""
        self._lr_scheduler = None
        self._optimizer = None
        self._wrapped_criterion = None
        self._wrapped_model = None

    @property
    def data_parallel_world_size(self):
        return self.args['distributed_training']['distributed_world_size']

    @property
    def data_parallel_process_group(self):
        return None

    @property
    def data_parallel_rank(self):
        return self.args['distributed_training']['distributed_rank']

    @property
    def is_data_parallel_master(self):
        return distributed_utils.is_master(self.args)

    @property
    def criterion(self):
        if self._wrapped_criterion is None:
            if (
                utils.has_parameters(self._criterion)
                and self.args['distributed_training']['distributed_world_size'] > 1
                and not self.args['optimization']['use_bmuf']
            ):
                self._wrapped_criterion = models.DistributedNccModel(
                    self.args, self._criterion
                )
            else:
                self._wrapped_criterion = self._criterion
        return self._wrapped_criterion

    @property
    def model(self):
        if self._wrapped_model is None:
            if self.args['distributed_training']['distributed_world_size'] > 1 and \
                not self.args['optimization']['use_bmuf']:
                self._wrapped_model = models.DistributedNccModel(
                    self.args, self._model
                )
            else:
                self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._setup_optimizer()
        return self._optimizer

    @property
    def lr_scheduler(self):
        if self._lr_scheduler is None:
            self._setup_optimizer()  # this will initialize self._lr_scheduler
        return self._lr_scheduler

    def _setup_optimizer(self, params=None):
        """
        params: None, set learning rates, decay factors for parameters
        """
        if params is None:
            params = list(
                filter(
                    lambda p: p.requires_grad,
                    chain(self.model.parameters(), self.criterion.parameters()),
                )
            )

        if self.args['common']['fp16'] or self.args['common']['bf16'] or self.args['common'].get('amp', False):
            if self.cuda and torch.cuda.get_device_capability(0)[0] < 7:
                LOGGER.info(
                    "NOTE: your device does NOT support faster training with fp16 or --amp, "
                    "please switch to FP32 which is likely to be faster"
                )
            if self.args['common']['memory_efficient_fp16'] or self.args['common']['memory_efficient_bf16']:
                self._optimizer = optimizers.MemoryEfficientFP16Optimizer.setup_optimizer(
                    self.args, params
                )
            elif self.args['common'].get('amp', False):
                self._optimizer = optimizers.AMPOptimizer.setup_optimizer(self.args, params)
            else:
                self._optimizer = optimizers.FP16Optimizer.setup_optimizer(self.args, params)
        else:
            if self.cuda and torch.cuda.get_device_capability(0)[0] >= 7:
                LOGGER.info("NOTE: your device may support faster training with fp16 or --amp")
            self._optimizer = optimizers.setup_optimizer(self.args, params)

        if self.args['optimization']['use_bmuf']:
            self._optimizer = optimizers.NccBMUF(self.args, self._optimizer)

        # We should initialize the learning rate scheduler immediately after
        # building the optimizer, so that the initial learning rate is set.
        self._lr_scheduler = lr_schedulers.build_lr_scheduler(self.args, self.optimizer)
        self._lr_scheduler.step_update(0)

    @property
    def is_fsdp(self):
        return self.args['distributed_training']['ddp_backend'] == "fully_sharded"

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        if distributed_utils.is_master(self.args):  # only save one checkpoint
            extra_state["metrics"] = metrics.state_dict()
            checkpoint_utils.save_state(
                filename,
                self.args,
                self.get_model().state_dict(),
                self.get_criterion(),
                self.optimizer,
                self.lr_scheduler,
                self.get_num_updates(),
                self._optim_history,
                extra_state,
            )

    def metrics2cuda(self, metrics):
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                metrics[key] = self.metrics2cuda(value)
        elif isinstance(metrics, list) or isinstance(metrics, tuple):
            metrics = list(metrics)
            for idx, m in enumerate(metrics):
                metrics[idx] = self.metrics2cuda(m)
        elif isinstance(metrics, torch.Tensor):
            metrics = metrics.to(self.device)
        return metrics

    def load_checkpoint(
        self,
        filename,
        reset_optimizer=False,
        reset_lr_scheduler=False,
        optimizer_overrides=None,
        reset_meters=False,
    ):
        """Load all training state from a checkpoint file."""
        extra_state, self._optim_history, last_optim_state = None, [], None

        bexists = PathManager.is_file(filename)
        if bexists:
            state = checkpoint_utils.load_checkpoint_to_cpu(filename)

            # load model parameters
            try:
                self.get_model().load_state_dict(
                    state["model"], strict=True, args=self.args
                )
                if utils.has_parameters(self.get_criterion()):
                    self.get_criterion().load_state_dict(
                        state["criterion"], strict=True
                    )
            except Exception:
                raise Exception(
                    "Cannot load model parameters from checkpoint {}; "
                    "please ensure that the architectures match.".format(filename)
                )

            extra_state = state["extra_state"]
            self._optim_history = state["optimizer_history"]
            last_optim_state = state.get("last_optimizer_state", None)

        if last_optim_state is not None and not reset_optimizer:
            # rebuild optimizer after loading model, since params may have changed
            self._setup_optimizer()

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]
            assert (
                last_optim["criterion_name"] == self.get_criterion().__class__.__name__
            ), "Criterion does not match; please reset the optimizer (--reset-optimizer)."
            assert (
                last_optim["optimizer_name"] == self.optimizer.__class__.__name__
            ), "Optimizer does not match; please reset the optimizer (--reset-optimizer)."

            if not reset_lr_scheduler:
                self.lr_scheduler.load_state_dict(last_optim["lr_scheduler_state"])
            self.optimizer.load_state_dict(last_optim_state, optimizer_overrides)

            self.set_num_updates(last_optim["num_updates"])

        if extra_state is not None:
            epoch = extra_state["train_iterator"]["epoch"]
            LOGGER.info(
                "loaded checkpoint {} (epoch {} @ {} updates)".format(
                    filename, epoch, self.get_num_updates()
                )
            )

            self.lr_step(epoch)

            if "metrics" in extra_state and not reset_meters:
                # if load pre-trained checkpoint, we need to convert metrics to cuda
                self.metrics2cuda(extra_state["metrics"])
                metrics.load_state_dict(extra_state["metrics"])

                # reset TimeMeters, since their start times don't make sense anymore
                for meter in metrics.get_meters("default"):
                    if isinstance(meter, meters.TimeMeter):
                        meter.reset()
        else:
            LOGGER.info("no existing checkpoint found {}".format(filename))

        return extra_state

    def get_train_iterator(
        self,
        epoch,
        combine=True,
        load_dataset=True,
        data_selector=None,
        shard_batch_itr=True,
        **kwargs,
    ):
        """Return an EpochBatchIterator over the training set for a given epoch."""
        if load_dataset:
            LOGGER.info("loading train data for epoch {}".format(epoch))
            self.task.load_dataset(
                self.args['dataset']['train_subset'],
                epoch=epoch,
                combine=combine,
                data_selector=data_selector,
                **kwargs,
            )
        max_positions = utils.resolve_max_positions(
            self.task.max_positions(),
            self.model.max_positions(),
            self.args['dataset']['max_tokens'],
        )
        return self.task.get_batch_iterator(
            dataset=self.task.dataset(self.args['dataset']['train_subset']),
            max_tokens=self.args['dataset']['max_tokens'],
            max_sentences=self.args['dataset']['max_sentences'],
            max_positions=max_positions,
            ignore_invalid_inputs=True,
            required_batch_size_multiple=self.args['dataset']['required_batch_size_multiple'],
            seed=self.args['common']['seed'],
            num_shards=self.args['distributed_training']['distributed_world_size'] if shard_batch_itr else 1,
            shard_id=self.args['distributed_training']['distributed_rank'] if shard_batch_itr else 0,
            num_workers=self.args['dataset']['num_workers'],
            epoch=epoch,
        )

    @metrics.aggregate("train")
    def train_step(self, samples, raise_oom=False):
        """Do forward, backward and parameter update."""
        if self._dummy_batch == "DUMMY":
            self._dummy_batch = samples[0]

        self._set_seed()
        self.model.train()
        self.criterion.train()
        self.zero_grad()

        metrics.log_start_time("train_wall", priority=800, round=0)

        # forward and backward pass
        logging_outputs, sample_size, ooms = [], 0, 0
        for i, sample in enumerate(samples):
            sample, is_dummy_batch = self._prepare_sample(sample)

            def maybe_no_sync():
                """
                Whenever *samples* contains more than one mini-batch, we
                want to accumulate gradients locally and only call
                all-reduce in the last backwards pass.
                """
                if (
                    self.args['distributed_training']['distributed_world_size'] > 1
                    and hasattr(self.model, "no_sync")
                    and i < len(samples) - 1
                ):
                    return self.model.no_sync()
                else:
                    return contextlib.ExitStack()  # dummy contextmanager

            try:
                with maybe_no_sync():
                    # forward and backward
                    loss, sample_size_i, logging_output = self.task.train_step(
                        sample=sample,
                        model=self.model,
                        criterion=self.criterion,
                        optimizer=self.optimizer,
                        update_num=self.get_num_updates(),
                        ignore_grad=is_dummy_batch,
                    )
                    del loss

                logging_outputs.append(logging_output)
                sample_size += sample_size_i

                # emptying the CUDA cache after the first step can
                # reduce the chance of OOM
                if self.cuda and self.get_num_updates() == 0:
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._log_oom(e)
                    if raise_oom:
                        raise e
                    LOGGER.warning(
                        "attempting to recover from OOM in forward/backward pass"
                    )
                    ooms += 1
                    self.zero_grad()
                    if self.cuda:
                        torch.cuda.empty_cache()
                    if self.args['distributed_training']['distributed_world_size'] == 1:
                        return None
                else:
                    raise e

        if is_dummy_batch:
            if torch.is_tensor(sample_size):
                sample_size.zero_()
            else:
                sample_size *= 0.0  # multiply by 0 to preserve device

        if torch.is_tensor(sample_size):
            sample_size = sample_size.float()
        else:
            sample_size = float(sample_size)

        # gather logging outputs from all replicas
        if self._sync_stats():
            logging_outputs, (sample_size, ooms) = self._aggregate_logging_outputs(
                logging_outputs, sample_size, ooms, ignore=is_dummy_batch,
            )

        overflow = False
        try:
            with torch.autograd.profiler.record_function("reduce-grads"):
                # reduce gradients across workers
                self.optimizer.all_reduce_grads(self.model)
                if utils.has_parameters(self.criterion):
                    self.optimizer.all_reduce_grads(self.criterion)

            with torch.autograd.profiler.record_function("multiply-grads"):
                # multiply gradients by (data_parallel_size / sample_size) since
                # DDP normalizes by the number of data parallel workers for
                # improved fp16 precision.
                # Thus we get (sum_of_gradients / sample_size) at the end.
                # In case of fp16, this step also undoes loss scaling.
                # (Debugging note: Some optimizers perform this scaling on the
                # fly, so inspecting model.parameters() or optimizer.params may
                # still show the original, unscaled gradients.)
                num = (
                    self.args['distributed_training']['distributed_world_size']
                    if not self.args['optimization']['use_bmuf'] or self._sync_stats()
                    else 1
                )
                self.optimizer.multiply_grads(num / (sample_size or 1.0))

            with torch.autograd.profiler.record_function("clip-grads"):
                # clip grads
                grad_norm = self.clip_grad_norm(self.args['optimization']['clip_norm'])

            # check that grad norms are consistent across workers
            if not self.args['optimization']['use_bmuf']:
                self._check_grad_norms(grad_norm)
            if not torch.isfinite(grad_norm).all():
                if self.args['common'].get('amp', False):
                    overflow = True
                else:
                    raise FloatingPointError("gradients are Nan/Inf")

            with torch.autograd.profiler.record_function("optimizer"):
                # take an optimization step
                self.optimizer.step()
                if self.args['common'].get('amp', False) and overflow:
                    if self._amp_retries == self.args['common']['amp_batch_retries']:
                        LOGGER.info("AMP: skipping this batch.")
                        self._amp_retries = 0
                    else:
                        self._amp_retries += 1
                        return self.train_step(samples, raise_oom)  # recursion to feed in same batch

        except FloatingPointError:
            # re-run the forward and backward pass with hooks attached to print
            # out where it fails
            self.zero_grad()
            with NanDetector(self._model):
                for _, sample in enumerate(samples):
                    sample, _ = self._prepare_sample(sample)
                    self.task.train_step(
                        sample,
                        self.model,
                        self.criterion,
                        self.optimizer,
                        self.get_num_updates(),
                        ignore_grad=False
                    )
            raise
        except OverflowError as e:
            overflow = True
            LOGGER.info(
                f"NOTE: gradient overflow detected, ignoring gradient, {str(e)}"
            )
            grad_norm = torch.tensor(0.0).cuda()
            self.zero_grad()
        except RuntimeError as e:
            if "out of memory" in str(e):
                self._log_oom(e)
                LOGGER.error("OOM during optimization, irrecoverable")
            raise e

        logging_output = None
        if not overflow:
            self.set_num_updates(self.get_num_updates() + 1)

            if self.cuda and self.cuda_env is not None:
                # log minimum free memory over the iteration
                gb_used = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
                torch.cuda.reset_peak_memory_stats()
                gb_free = self.cuda_env.total_memory_in_GB - gb_used
                metrics.log_scalar(
                    "gb_free", gb_free, priority=1500, round=1, weight=0
                )

            # log stats
            logging_output = self._reduce_and_log_stats(
                logging_outputs, sample_size, grad_norm,
            )

            # clear CUDA cache to reduce memory fragmentation
            if (
                self.cuda
                and self.args['common']['empty_cache_freq'] > 0
                and (
                (self.get_num_updates() + self.args['common']['empty_cache_freq'] - 1)
                % self.args['common']['empty_cache_freq']
            )
                == 0
            ):
                torch.cuda.empty_cache()

        if self.args['common']['fp16'] or self.args['common'].get('amp', False):
            metrics.log_scalar(
                "loss_scale",
                (
                    self.optimizer.scaler.loss_scale
                    if self.args['common']['fp16']
                    else self.optimizer.scaler.get_scale()
                ),
                priority=700,
                round=4,
                weight=0,
            )

        metrics.log_stop_time("train_wall")
        return logging_output

    def clip_grad_norm(self, clip_norm):
        _clip_norm_fn = self.args['optimization'].get('clip_norm_version', 'clip_grad_norm')
        return getattr(self.optimizer, _clip_norm_fn)(clip_norm, aggregate_norm_fn=None)

    @metrics.aggregate("valid")
    def valid_step(self, sample, raise_oom=False):
        """Do forward pass in evaluation mode."""
        with torch.no_grad():
            self.model.eval()
            self.criterion.eval()

            sample, is_dummy_batch = self._prepare_sample(sample)

            try:
                _loss, sample_size, logging_output = self.task.valid_step(
                    sample, self.model, self.criterion
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._log_oom(e)
                    if not raise_oom:
                        LOGGER.warning(
                            "ran out of memory in validation step, retrying batch"
                        )
                        for p in self.model.parameters():
                            if p.grad is not None:
                                p.grad = None  # free some memory
                        if self.cuda:
                            torch.cuda.empty_cache()
                        return self.valid_step(sample, raise_oom=True)
                raise e

            logging_outputs = [logging_output]
            if is_dummy_batch:
                if torch.is_tensor(sample_size):
                    sample_size.zero_()
                else:
                    sample_size *= 0.0

        # gather logging outputs from all replicas
        if self.args['distributed_training']['distributed_world_size'] > 1:
            logging_outputs, (sample_size,) = self._aggregate_logging_outputs(
                logging_outputs, sample_size, ignore=is_dummy_batch,
            )
            if 'bleu' in logging_outputs[0]:
                logging_outputs[0]['bleu'] /= self.args['distributed_training']['distributed_world_size']
            if 'rouge_l' in logging_outputs[0]:
                logging_outputs[0]['rouge_l'] /= self.args['distributed_training']['distributed_world_size']
            if 'meteor' in logging_outputs[0]:
                logging_outputs[0]['meteor'] /= self.args['distributed_training']['distributed_world_size']
        # log validation stats
        logging_output = self._reduce_and_log_stats(logging_outputs, sample_size)
        return logging_output

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate at the end of the epoch."""
        self.lr_scheduler.step(epoch, val_loss)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update()

    def lr_step_update(self):
        """Update the learning rate after each update."""
        new_lr = self.lr_scheduler.step_update(self.get_num_updates())
        metrics.log_scalar("lr", new_lr, weight=0, priority=300, round=6)
        return new_lr

    def set_lr(self, new_lr):
        """Set the learning rate by users"""
        self.lr_scheduler.lr = new_lr
        self.lr_scheduler.optimizer.set_lr(new_lr)
        metrics.log_scalar("lr", new_lr, weight=0, priority=300, round=6)
        return self.lr_scheduler.optimizer.get_lr()

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    def get_model(self):
        """Get the (non-wrapped) model instance."""
        return self._model

    def get_criterion(self):
        """Get the (non-wrapped) criterion instance."""
        return self._criterion

    def get_meter(self, name):
        """[deprecated] Get a specific meter by name."""
        from ncc import meters

        if 'get_meter' not in self._warn_once:
            self._warn_once.add('get_meter')
            utils.deprecation_warning(
                'Trainer.get_meter is deprecated. Please use fairseq.metrics instead.'
            )

        train_meters = metrics.get_meters("train")
        if train_meters is None:
            train_meters = {}

        if name == "train_loss" and "loss" in train_meters:
            return train_meters["loss"]
        elif name == "train_nll_loss":
            # support for legacy train.py, which assumed this meter is
            # always initialized
            m = train_meters.get("nll_loss", None)
            return m or meters.AverageMeter()
        elif name == "wall":
            # support for legacy train.py, which assumed this meter is
            # always initialized
            m = metrics.get_meter("default", "wall")
            return m or meters.TimeMeter()
        elif name == "wps":
            m = metrics.get_meter("train", "wps")
            return m or meters.TimeMeter()
        elif name in {"valid_loss", "valid_nll_loss"}:
            # support for legacy train.py, which assumed these meters
            # are always initialized
            k = name[len("valid_"):]
            m = metrics.get_meter("valid", k)
            return m or meters.AverageMeter()
        elif name == "oom":
            return meters.AverageMeter()
        elif name in train_meters:
            return train_meters[name]
        return None

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self._num_updates = num_updates
        self.lr_step_update()
        metrics.log_scalar("num_updates", self._num_updates, weight=0, priority=200, round=6)

    def _fp_convert_sample(self, sample):
        def apply_half(t):
            if t.dtype is torch.float32:
                return t.to(dtype=torch.half)
            return t

        def apply_bfloat16(t):
            if t.dtype is torch.float32:
                return t.to(dtype=torch.bfloat16)
            return t

        if self.args['common']['fp16']:
            sample = utils.apply_to_sample(apply_half, sample)

        if self.args['common']['fp16']:
            sample = utils.apply_to_sample(apply_bfloat16, sample)

        return sample

    def _prepare_sample(self, sample, is_dummy=False):
        if sample == "DUMMY":
            raise Exception(
                "Trying to use an uninitialized 'dummy' batch. This usually indicates "
                "that the total number of batches is smaller than the number of "
                "participating GPUs. Try reducing the batch size or using fewer GPUs."
            )

        if sample is None or len(sample) == 0:
            assert (
                self._dummy_batch is not None and len(self._dummy_batch) > 0
            ), "Invalid dummy batch: {}".format(self._dummy_batch)
            sample, _ = self._prepare_sample(self._dummy_batch, is_dummy=True)
            return sample, True

        if self.args['common'].get('on_cpu_convert_precision', False):
            sample = self._fp_convert_sample(sample)

        if self.cuda:
            sample = move_to_cuda(sample)

        if not self.args['common'].get('on_cpu_convert_precision', False):
            sample = self._fp_convert_sample(sample)

        if self._dummy_batch == "DUMMY":
            self._dummy_batch = sample

        return sample, False

    def _set_seed(self):
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.args['common']['seed'] + self.get_num_updates()
        torch.manual_seed(seed)
        if self.cuda:
            torch.cuda.manual_seed(seed)

    def _sync_stats(self):
        # Return True if it's using multiple GPUs and DDP or multiple GPUs with
        # BMUF and it's a bmuf sync with warmup iterations completed before.
        return self.args['distributed_training']['distributed_world_size'] > 1 and (
            (not self.args['optimization']['use_bmuf'])
            or (
                self.args['optimization']['use_bmuf']
                and (self.get_num_updates() + 1) % self.args['distributed_training']['global_sync_iter'] == 0
                and (self.get_num_updates() + 1) > self.args['distributed_training']['warmup_iterations']
            )
        )

    def _log_oom(self, exc):
        msg = "OOM: Ran out of memory with exception: {}".format(exc)
        LOGGER.warning(msg)
        if torch.cuda.is_available() and hasattr(torch.cuda, "memory_summary"):
            for device_idx in range(torch.cuda.device_count()):
                LOGGER.warning(torch.cuda.memory_summary(device=device_idx))
        sys.stderr.flush()

    def _aggregate_logging_outputs(
        self,
        logging_outputs,
        *extra_stats_to_sum,
        ignore=False,
    ):
        if self.task.__class__.logging_outputs_can_be_summed(self.get_criterion()):
            return self._fast_stat_sync_sum(
                logging_outputs, *extra_stats_to_sum, ignore=ignore
            )
        else:
            return self._all_gather_list_sync(
                logging_outputs, *extra_stats_to_sum, ignore=ignore
            )

    def _all_gather_list_sync(
        self,
        logging_outputs,
        *extra_stats_to_sum,
        ignore=False,
    ):
        """
        Sync logging outputs across workers. all_gather_list_sync is
        suitable when logging outputs are complex types.
        """
        if ignore:
            logging_outputs = []
        results = list(zip(
            *distributed_utils.all_gather_list(
                [logging_outputs] + list(extra_stats_to_sum),
                max_size=getattr(self.args, 'all_gather_list_size', 16384),
            )
        ))
        logging_outputs, extra_stats_to_sum = results[0], results[1:]
        logging_outputs = list(chain.from_iterable(logging_outputs))
        extra_stats_to_sum = [sum(s) for s in extra_stats_to_sum]
        return logging_outputs, extra_stats_to_sum

    def _fast_stat_sync_sum(
        self,
        logging_outputs,
        *extra_stats_to_sum,
        ignore=False,
    ):
        """
        Sync logging outputs across workers. fast_stat_sync_sum is
        faster than all_gather_list_sync, but is only suitable when
        logging outputs are scalars and can be summed. Note that
        *logging_outputs* cannot contain any nested dicts/lists.
        """
        data = {}
        for i, stat in enumerate(extra_stats_to_sum):
            data['extra_stats_' + str(i)] = stat
        if len(logging_outputs) > 0:
            log_keys = list(logging_outputs[0].keys())
            for k in log_keys:
                if not ignore:
                    v = sum(log[k] for log in logging_outputs if k in log)
                else:
                    v = logging_outputs[0][k]
                    v = torch.zeros_like(v) if torch.is_tensor(v) else 0
                data['logging_outputs_' + k] = v
        else:
            log_keys = None

        data = distributed_utils.all_reduce_dict(
            data,
            device=self.device,
        )

        extra_stats_to_sum = [
            data['extra_stats_' + str(i)] for i in range(len(extra_stats_to_sum))
        ]
        if log_keys is not None:
            logging_outputs = [{k: data['logging_outputs_' + k] for k in log_keys}]
        else:
            logging_outputs = []
        return logging_outputs, extra_stats_to_sum

    def _check_grad_norms(self, grad_norm):
        """Check that grad norms are consistent across workers."""
        if self._grad_norm_buf is not None:
            self._grad_norm_buf.zero_()
            self._grad_norm_buf[self.args['distributed_training']['distributed_rank']] = grad_norm
            distributed_utils.all_reduce(
                self._grad_norm_buf,
                group=self.data_parallel_process_group
            )

            def is_consistent(tensor):
                max_abs_diff = torch.max(torch.abs(tensor - tensor[0]))
                return (
                    not torch.isfinite(tensor).any()
                    or (max_abs_diff / (tensor[0] + 1e-6) < 1e-6).all()
                    or
                    (self.args['common'].get('amp', False) and not torch.isfinite(tensor).all())
                )

            if not is_consistent(self._grad_norm_buf):
                pretty_detail = "\n".join(
                    "rank {:3d} = {:.8f}".format(r, n)
                    for r, n in enumerate(self._grad_norm_buf.tolist())
                )
                error_detail = "grad_norm across the workers:\n{}\n".format(pretty_detail)
                raise RuntimeError(
                    "Fatal error: gradients are inconsistent between workers. "
                    "Try --ddp-backend=no_c10d. "
                    "Or are you mixing up different generation of GPUs in training?"
                    + "\n"
                    + "-" * 80
                    + "\n{}\n".format(error_detail)
                    + "-" * 80
                )

    def _reduce_and_log_stats(self, logging_outputs, sample_size, grad_norm=None):
        if grad_norm is not None:
            metrics.log_speed("ups", 1., priority=100, round=2)
            metrics.log_scalar("gnorm", grad_norm, priority=400, round=3)
            if self.args['optimization']['clip_norm'] > 0:
                metrics.log_scalar(
                    "clip",
                    torch.where(
                        grad_norm > self.args['optimization']['clip_norm'],
                        grad_norm.new_tensor(100),
                        grad_norm.new_tensor(0),
                    ),
                    priority=500,
                    round=1,
                )

        with metrics.aggregate() as agg:
            if logging_outputs is not None:
                self.task.reduce_metrics(logging_outputs, self.get_criterion())

            # support legacy interface
            logging_output = agg.get_smoothed_values()
            logging_output["sample_size"] = sample_size
            for key_to_delete in ["ppl", "wps", "wpb", "bsz"]:
                if key_to_delete in logging_output:
                    del logging_output[key_to_delete]
            return logging_output

    def test_bleu_step(self, generator, sample, bleu_scorers, print_to_file=None):
        with torch.no_grad():
            self.model.eval()
            sample = self._prepare_sample(sample)
            if sample is not None:
                self.test_bleu(generator, sample, bleu_scorers, print_to_file)

    def test_bleu(self, generator, sample, bleu_scorers, print_to_file=None):
        if sample is None:
            return

        beam = self.args['eval']['beam']
        with torch.no_grad():
            while True:
                try:
                    hypos = generator.generate(
                        {'src_tokens': sample['net_input']['src_tokens'],
                         'src_lengths': sample['net_input']['src_lengths'], },
                        beam
                    )
                    break
                except RuntimeError as e:
                    if 'out of memory' in str(e) and beam >= 3:
                        beam = beam - 1
                        print('| WARNING: ran out of memory, reduce beam size to %d' % beam)
                    else:
                        raise e

            assert len(sample['target']) == len(hypos)
            for dataset_id, tgt, hypo in zip(
                list(sample['dataset_id']) if 'dataset_id' in sample else list(torch.LongTensor([0] * len(hypos))),
                list(sample['target']),
                hypos
            ):
                # remove BOS/EOS and keep UNK
                target_tokens = torch.IntTensor(
                    [
                        i for i in tgt.tolist()
                        if not (i in {self.task.tgt_dict.eos(), self.task.tgt_dict.pad(), self.task.tgt_dict.bos()})
                    ]
                )
                hypo_tokens = torch.IntTensor(
                    [
                        i for i in hypo[0]['tokens'].tolist()
                        if not (i in {self.task.tgt_dict.eos(), self.task.tgt_dict.pad(), self.task.tgt_dict.bos()})
                    ]
                )
                bleu_scorer_ = bleu_scorers[dataset_id.item()]
                bleu_scorer_.add(target_tokens, hypo_tokens)

                if print_to_file:
                    target_str = self.task.tgt_dict.string(tgt.tolist(), escape_unk=True)
                    hypo_str = self.task.tgt_dict.string(hypo[0]['tokens'].tolist())
                    print_to_file(dataset_id.item(), target_str, hypo_str)
