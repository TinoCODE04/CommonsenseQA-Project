"""Pin the observed Trainer/RoBERTa loss-scaling compatibility issue."""

import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from project_env import configure_cache

configure_cache()

import torch
from transformers import RobertaConfig, RobertaForMultipleChoice, TrainingArguments
from mcqa_trainer import MultipleChoiceTrainer


class GradientAccumulationTests(unittest.TestCase):
    def test_two_microbatches_match_one_combined_batch_loss_and_gradient(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        config = RobertaConfig(
            vocab_size=31, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=32,
            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
            pad_token_id=1, bos_token_id=0, eos_token_id=2,
        )
        combined_model = RobertaForMultipleChoice(config)
        accumulated_model = copy.deepcopy(combined_model)
        ids = torch.arange(80).reshape(2, 5, 8) % 27 + 4
        labels = torch.tensor([0, 3])
        combined = combined_model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
        combined.loss.backward()
        expected_loss = combined.loss.item()
        expected_gradient = combined_model.classifier.weight.grad.detach().clone()
        args = TrainingArguments(
            output_dir=str(ROOT / "tmp/ga-regression"), use_cpu=True,
            gradient_accumulation_steps=2, per_device_train_batch_size=1,
            report_to="none", disable_tqdm=True, dataloader_pin_memory=False,
        )
        trainer = MultipleChoiceTrainer(model=accumulated_model, args=args)
        trainer.current_gradient_accumulation_steps = 2
        reported_loss = 0.0
        for index in range(2):
            microbatch = {
                "input_ids": ids[index:index + 1],
                "attention_mask": torch.ones_like(ids[index:index + 1]),
                "labels": labels[index:index + 1],
            }
            reported_loss += trainer.training_step(
                accumulated_model, microbatch, num_items_in_batch=torch.tensor(2),
            ).item()
        self.assertAlmostEqual(reported_loss, expected_loss, places=6)
        torch.testing.assert_close(
            accumulated_model.classifier.weight.grad, expected_gradient,
            atol=1e-8, rtol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
