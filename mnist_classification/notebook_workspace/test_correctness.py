"""Scientific invariants: scoring, probability mixtures, BN freezing, FGSM, resume."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from engine import (load_checkpoint, perturb, predict, seed_all, train_member, training_step)
from metrics import classification_metrics
from model import MLP, evaluation_mode


class Correctness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_metrics_known_values(self):
        labels = torch.tensor([0, 9])
        result = classification_metrics(torch.full((2, 10), .1), labels)
        self.assertAlmostEqual(result["nll"], math.log(10), places=6)
        self.assertAlmostEqual(result["brier"], .09, places=7)
        self.assertEqual(result["accuracy"], .5)  # argmax tie selects zero
        perfect = nn.functional.one_hot(labels, 10).float()
        self.assertEqual(classification_metrics(perfect, labels)["brier"], 0)
        wrong = perfect.flip(0)
        self.assertAlmostEqual(classification_metrics(wrong, labels)["brier"], .2)
        self.assertEqual(classification_metrics(wrong, labels)["classification_error"], 1)

    def test_probability_mixture_is_not_logit_average(self):
        logits = torch.zeros(2, 1, 10)
        logits[0, 0, 0] = 10
        logits[1, 0, 1] = 2
        average_probabilities = logits.softmax(-1).mean(0)
        self.assertFalse(torch.allclose(average_probabilities, logits.mean(0).softmax(-1)))
        self.assertTrue(torch.allclose(average_probabilities.sum(-1), torch.ones(1)))

    def test_dropout_stochastic_bn_frozen(self):
        seed_all(3)
        model = MLP(.1)
        x = torch.rand(32, 784)
        model.train()
        model(x)
        before = {k: v.clone() for k, v in model.named_buffers()}
        a, b = predict(model, x, mc_dropout=True), predict(model, x, mc_dropout=True)
        self.assertFalse(torch.equal(a, b))
        for key, value in model.named_buffers():
            self.assertTrue(torch.equal(before[key], value), key)
        self.assertTrue(all(not m.training for m in model.modules() if isinstance(m, nn.BatchNorm1d)))
        self.assertTrue(torch.equal(predict(model, x), predict(model, x)))

    def test_fgsm_random_and_gradient_update(self):
        x = torch.full((12, 784), .5)
        epsilon = torch.linspace(0, .01, 784)
        gradient = torch.linspace(-1, 1, 784).expand_as(x)
        adv = perturb(x, epsilon, "adversarial", gradient)
        self.assertTrue(torch.allclose(adv - x, epsilon * gradient.sign(), atol=3e-8))
        rand = perturb(x, epsilon, "random")
        self.assertTrue(torch.allclose((rand - x).abs(), epsilon.expand_as(x), atol=3e-8))
        for method in ("ensemble", "random", "adversarial", "mc_dropout"):
            model = MLP(.1 if method == "mc_dropout" else 0)
            opt = torch.optim.Adam(model.parameters(), lr=.001)
            old = next(model.parameters()).detach().clone()
            result = training_step(model, opt, x, torch.arange(12) % 10, epsilon, method)
            self.assertTrue(torch.isfinite(result).all())
            self.assertFalse(torch.equal(old, next(model.parameters())))
        self.assertFalse(adv.requires_grad)

    def test_exact_resume_all_methods_and_mismatch(self):
        # Includes Adam moments, random perturbations, dropout masks and BN buffers.
        c = {"dropout": .1, "learning_rate": .001, "epochs": 2,
             "batch_size": 8, "eval_batch_size": 20}
        seed_all(100)
        base_x, base_y = torch.rand(24, 784), torch.arange(24) % 10
        with tempfile.TemporaryDirectory() as tmp:
            for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
                x, y = base_x.to(device), base_y.to(device)
                epsilon = torch.full((784,), .01, device=device)
                for method in ("ensemble", "random", "adversarial", "mc_dropout"):
                    a, b = Path(tmp) / (device + method + "_a"), Path(tmp) / (device + method + "_b")
                    p1 = train_member(c, "test", a, method, 42, 0, x, y, x, y, epsilon)
                    train_member(c, "test", b, method, 42, 0, x, y, x, y, epsilon, stop_after_epoch=1)
                    p2 = train_member(c, "test", b, method, 42, 0, x, y, x, y, epsilon)
                    s1, s2 = load_checkpoint(p1), load_checkpoint(p2)
                    for key in s1["model"]:
                        self.assertTrue(torch.equal(s1["model"][key], s2["model"][key]), f"{device}/{method}/{key}")
                    self.assertEqual(s1["history"][-1]["clean_loss"], s2["history"][-1]["clean_loss"])
                    with self.assertRaisesRegex(ValueError, "mismatch"):
                        train_member(c, "changed", b, method, 42, 0, x, y, x, y, epsilon)


if __name__ == "__main__":
    unittest.main(verbosity=2)
