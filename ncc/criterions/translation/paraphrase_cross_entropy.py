# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch.nn.functional as F

from ncc.criterions import NccCriterion, register_criterion
from ncc.data.dictionary import Dictionary
from ncc.utils import utils
from ncc.utils.logging import metrics
from ncc.criterions.common.cross_entropy import CrossEntropyCriterion


@register_criterion('paraphrase_cross_entropy')
class ParaphraseCrossEntropyCriterion(CrossEntropyCriterion):

    def __init__(self, task, sentence_avg):
        super().__init__(task, sentence_avg)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # net_output = model(**sample['net_input'])
        # loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        # sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']

        # recon_loss, cls_loss = model(**sample['net_input'])
        # loss = recon_loss / sample['ntokens'] + cls_loss / sample['nsentences']

        recon_loss = model(**sample['net_input'])
        loss = recon_loss / sample['ntokens']
        sample_size = 1

        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)

        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        # ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        # if sample_size != ntokens:
        #     metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
        #     metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        # else:
        #     metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
