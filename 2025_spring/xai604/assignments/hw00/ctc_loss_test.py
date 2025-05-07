"""Unit tests for the ctc_loss_lib module."""

# pylint: disable=import-error, no-member, no-name-in-module

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import unittest

# Third-party imports
import torch

# Custom imports
import ctc_loss_lib

# Sets the log of minius inifinty of the float 32 type.
LOG_0 = ctc_loss_lib.LOG_0


class SeqFormatConversionTest(unittest.TestCase):
    """A class for testing methods in the ctc_loss module."""

    def test_to_blank_augmented_labels_default_blank_index(self):
        blank_index = 0

        inputs = {}
        inputs["SEQ_DATA"] = torch.tensor([[1, 3, 4, 2, 5], [1, 4, 2, 5, 0]],
                                          dtype=torch.int32)
        inputs["SEQ_LEN"] = torch.tensor([5, 4], dtype=torch.int32)

        actual_output = ctc_loss_lib.to_blank_augmented_labels(
            inputs, blank_index)

        expected_output = {}
        expected_output["SEQ_DATA"] = torch.tensor(
            [[0, 2, 0, 4, 0, 5, 0, 3, 0, 6, 0],
             [0, 2, 0, 5, 0, 3, 0, 6, 0, 0, 0]],
            dtype=torch.int32)
        expected_output["SEQ_LEN"] = torch.tensor([11, 9], dtype=torch.int32)

        self.assertTrue(
            torch.equal(expected_output["SEQ_DATA"],
                        actual_output["SEQ_DATA"]))
        self.assertTrue(
            torch.equal(expected_output["SEQ_LEN"], actual_output["SEQ_LEN"]))


class SeqLossUtilTest(unittest.TestCase):

    def test_label_trans_allowance_table_ctc(self):
        """Tests the label_trans_allowance_table_ctc method.

        In this unit test, it is assumed that "0" corresponds to the blank
        label.
        """

        # yapf: disable
        labels = torch.tensor([[0, 1, 0, 2, 0, 3, 0],
                               [0, 1, 0, 2, 0, 2, 0],
                               [0, 1, 0, 1, 0, 0, 0]])
        labels_len = torch.tensor([7, 7, 5])
        # yapf: enable

        actual = ctc_loss_lib.label_trans_allowance_table(labels, labels_len)

        expected = torch.tensor(
            [[[  0.0,   0.0, LOG_0, LOG_0, LOG_0, LOG_0, LOG_0],
              [LOG_0,   0.0,   0.0,   0.0, LOG_0, LOG_0, LOG_0],
              [LOG_0, LOG_0,   0.0,   0.0, LOG_0, LOG_0, LOG_0],
              [LOG_0, LOG_0, LOG_0,   0.0,   0.0,   0.0, LOG_0],
              [LOG_0, LOG_0, LOG_0, LOG_0,   0.0,   0.0, LOG_0],
              [LOG_0, LOG_0, LOG_0, LOG_0, LOG_0,   0.0,   0.0],
              [LOG_0, LOG_0, LOG_0, LOG_0, LOG_0, LOG_0,   0.0]],
             [[  0.0,   0.0, LOG_0, LOG_0, LOG_0, LOG_0, LOG_0],
              [LOG_0,   0.0,   0.0,   0.0, LOG_0, LOG_0, LOG_0],
              [LOG_0, LOG_0,   0.0,   0.0, LOG_0, LOG_0, LOG_0],
              [LOG_0, LOG_0, LOG_0,   0.0,   0.0, LOG_0, LOG_0],
              [LOG_0, LOG_0, LOG_0, LOG_0,   0.0,   0.0, LOG_0],
              [LOG_0, LOG_0, LOG_0, LOG_0, LOG_0,   0.0,   0.0],
              [LOG_0, LOG_0, LOG_0, LOG_0, LOG_0, LOG_0,   0.0]],
             [[  0.0,  0.0,  LOG_0, LOG_0, LOG_0, LOG_0, LOG_0],
              [LOG_0,  0.0,    0.0, LOG_0, LOG_0, LOG_0, LOG_0],
              [LOG_0, LOG_0,   0.0,   0.0, LOG_0, LOG_0, LOG_0],
              [LOG_0, LOG_0, LOG_0,   0.0,  0.0,    0.0, LOG_0],
              [LOG_0, LOG_0, LOG_0, LOG_0,  0.0,    0.0, LOG_0],
              [LOG_0, LOG_0, LOG_0, LOG_0, LOG_0,   0.0,   0.0],
              [LOG_0, LOG_0, LOG_0, LOG_0, LOG_0, LOG_0,   0.0]]],
             dtype=torch.float32) # yapf: disable

        # Checks the actual output with respect to the expected output.
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
