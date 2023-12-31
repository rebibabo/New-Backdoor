# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch.utils.data


class EpochListening:
    """Mixin for receiving updates whenever the epoch increments."""

    def set_epoch(self, epoch):
        """Will receive the updated epoch number at the beginning of the epoch.
        """
        pass


class NccDataset(torch.utils.data.Dataset, EpochListening):
    """A dataset that provides helpers for batching."""

    # default: shuffle indices
    shuffle = True

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        raise NotImplementedError

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        raise NotImplementedError

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        raise NotImplementedError

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        # return np.arange(len(self))
        indices = np.arange(len(self))
        src_sizes = getattr(self, "src_sizes", None)
        tgt_sizes = getattr(self, "tgt_sizes", None)
        if src_sizes is not None and tgt_sizes is not None:
            # ignore the data pairs which contains invalid sentence
            valid_indices = (src_sizes > 0) & (tgt_sizes > 0)
            indices = indices[valid_indices]
        if self.shuffle:
            # sort by target length, then source length
            if tgt_sizes is not None:
                indices = indices[
                    np.argsort(tgt_sizes[indices], kind='mergesort')
                ]
            return indices[np.argsort(src_sizes[indices], kind='mergesort')]
        else:
            return indices

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False

    def attr(self, attr: str, index: int):
        return getattr(self, attr, None)

    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        raise NotImplementedError


class NccIterableDataset(torch.utils.data.IterableDataset, EpochListening):
    """For datasets that need to be read sequentially, usually because the data
    is being streamed or otherwise can't be manipulated on a single machine.
    """

    def __iter__(self):
        raise NotImplementedError
