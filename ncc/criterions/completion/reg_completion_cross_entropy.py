# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from ncc.criterions import NccCriterion, register_criterion
from ncc.data.dictionary import Dictionary
from ncc.utils import utils
from ncc.utils.logging import metrics


@register_criterion('reg_completion_cross_entropy')
class RegCompletionCrossEntropyCriterion(NccCriterion):

    def __init__(self, task, sentence_avg, reg_weight):
        super().__init__(task)
        if hasattr(task, 'target_dictionary'):
            tgt_dict = task.target_dictionary
            self.padding_idx = tgt_dict.pad() if (tgt_dict is not None) and isinstance(tgt_dict, Dictionary) else -100
        self.sentence_avg = sentence_avg
        self.reg_weight = reg_weight

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        with torch.no_grad():
            model_params = model.state_dict()
            model.load_state_dict(sample['last_model_params'])
            hidden_reprs = model.extract_features(**sample['net_input']).detach()
            model.load_state_dict(model_params)
        net_output = model(**sample['net_input'])

        loss, _ = self.compute_loss(model, net_output, hidden_reprs, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, hidden_reprs, sample, reduce=True):
        # ignore pad
        idx = sample['net_input']['src_tokens'].view(-1) != self.padding_idx

        # ignore overlapping tokens
        max_len = sample['target'].size(-1)
        for i, ext_i in enumerate(sample['extends']):
            idx[i * max_len:i * max_len + ext_i] = 0

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))[idx]
        target = model.get_targets(sample, net_output).view(-1)[idx]

        # l2 norm
        regloss = F.mse_loss(
            hidden_reprs,  # [B, L, D]
            net_output[1],  # hidden_repr
            reduction='sum' if reduce else 'none',
        )

        nll_loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )
        return nll_loss + self.reg_weight * regloss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        else:
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
