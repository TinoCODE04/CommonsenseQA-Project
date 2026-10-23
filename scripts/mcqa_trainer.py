"""Trainer adapter for RoBERTa's per-microbatch mean classification loss."""

from transformers import Trainer


class MultipleChoiceTrainer(Trainer):
    """Normalize RoBERTa's mean cross-entropy over accumulated microbatches.

    RoBERTa accepts generic forward kwargs but does not divide its loss by
    num_items_in_batch. Explicitly override Trainer's kwargs-based inference.
    """

    loss_is_scaled_for_ga = False
