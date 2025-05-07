"""A module implementing utilities for sequence losses."""

# pylint: disable=no-member, invalid-name, import-error

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"
# Standard imports
import enum

# Third-party imports
import numpy as np
import torch

LOG_0 = torch.tensor(np.log(np.finfo(np.float64).tiny).astype(np.float32))

def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
    """Applies sequence masking.

    This implementation is based on the following website.
    https://discuss.pytorch.org/t/pytorch-equivalent-for-tf-sequence-mask/39036/3

    """
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    return mask.type(dtype)


def to_blank_augmented_labels(
        inputs: dict, blank_index: int=0, boundary_blanks: bool=True,
        update_non_blank_token_index: bool=True) -> dict:  # yapf: disable
    """Expands the input sequence with blank labels.

    The blank symbol is inserted at the beginning and at the end of the
    sequences, as well as between labels. If boundary_blanks is False, then
    blank labels are not inserted at the beginning and the end of the sequence.

    Args:
        inputs: A dict containing the input sequence.
            SEQ_DATA: A sparse tensor containing ground truth values.
                The shape is (batch_size, sequence_length).
            SEQ_LEN: A tensor of rank one containing the length of each
                ground truth sequence. The shape is (batch_size).
        blank_index:
            An integer for the blank label in the CTC loss.
        boundary_blanks:
            A boolean flag to insert labels at the boundaries of the sequence.
        unpdate_non_blank_token_index:
            A boolean flag to update non-blank token indices.
                When the blank token index is added, we may need to update the
                indices of non-blank tokens to make a room for the blank token
                index to avoid having conflicting indices. If this issue has
                been already taken care of, then set this flag to False. In
                fine tuning the Wav2Vec2.0 huggingface model, this flag needs
                to be set False.
    Returns:
        A dictionary containing a blank augmented sequence.
    """
    assert isinstance(inputs, dict)
    assert {"SEQ_DATA", "SEQ_LEN"} <= inputs.keys()


    return output


def label_trans_allowance_table(labels, labels_len):
    """Constructs a table containing the label transition allowance flags.

    We assume that label_seq should contain "blank labels" described in the
    original CTC paper.
    The shape of the returned tensor is (batch_size, max_seq_len, max_seq_len).
    The transition rule is as follows:

    Depending on whether the transition from the i-th label to the j-th label
    in the label sequence is allowed,
      a(b, i, j) = 0,         if this transition is allowed.
      a[b, i, j] = LOG_0:     if this transition is not allowed.

    Args:
        label_seq: A dictionary containing a batch of label sequences.
            * "DATA": A tensor containing label sequences.
                The shape is (batch_size, max_seq_length). Note that the data
                should follow the blank label rule, which states that "blank"
                labels should be interleaved with real labels. In addition to
                this, blank symbols are prepended and appended to the sequence.
            * "SEQ_LEN": A tensor containing the length of each label sequence.
                The shape is (batch_size).
    Returns:
        A tensor containing flags whether transitions are allowed.
            The shape is (batch_size, max_label_seq_len, max_seq_len)
    """

    return trans_table
