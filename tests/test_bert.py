"""Tests for BERT utilities, training, and finalization."""

import json
import os
import sys
import tempfile

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.charlson_mapping import CATEGORIES
from src.bert_utils import (
    compute_metrics_fn,
    full_evaluation,
    list_checkpoints,
    load_checkpoint_model_tokenizer,
    load_run_config,
    optimize_thresholds,
    predict_probs,
    resolve_checkpoint,
    save_final,
    select_device,
)
from src.train_bert import CharlsonDataset

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_PMHS = [
    "HTN, DM2, COPD",
    "CHF, CKD stage 4",
    "breast cancer, s/p mastectomy",
    "ESRD on hemodialysis",
    "rheumatoid arthritis on methotrexate",
    "hepatitis C, cirrhosis",
    "metastatic lung cancer",
    "HIV/AIDS, Pneumocystis",
    "hemiplegia after stroke",
    "peptic ulcer disease",
    "STEMI, s/p PCI",
    "Alzheimer's disease",
    "peripheral vascular disease",
    "asthma, home oxygen",
    "SLE, lupus nephritis",
    "diabetic nephropathy, CKD 5",
    "decompensated cirrhosis with ascites",
    "colon cancer stage IV",
    "no significant PMH",
    "HTN, hyperlipidemia",
]

NUM_LABELS = len(CATEGORIES)


def _create_tiny_model(tmpdir, save_tokenizer=True):
    """Create a tiny BERT model in tmpdir. Returns (model, tokenizer)."""
    config = AutoConfig.from_pretrained(
        "bert-base-uncased",
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=64,
    )
    model = AutoModelForSequenceClassification.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    model.save_pretrained(tmpdir)
    if save_tokenizer:
        tokenizer.save_pretrained(tmpdir)
    return model, tokenizer


def _create_fake_checkpoint(run_dir, step, epoch, tokenizer=None, best_checkpoint=None):
    """Create a fake checkpoint-<step> directory with trainer_state.json."""
    ckpt_dir = os.path.join(run_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save a tiny model in checkpoint
    config = AutoConfig.from_pretrained(
        "bert-base-uncased",
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=64,
    )
    model = AutoModelForSequenceClassification.from_config(config)
    model.save_pretrained(ckpt_dir)

    if tokenizer is not None:
        tokenizer.save_pretrained(ckpt_dir)

    state = {"epoch": epoch, "global_step": step}
    if best_checkpoint:
        state["best_model_checkpoint"] = best_checkpoint
    with open(os.path.join(ckpt_dir, "trainer_state.json"), "w") as f:
        json.dump(state, f)

    return ckpt_dir


@pytest.fixture(scope="module")
def tiny_model_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_tiny_model(tmpdir)
        yield tmpdir


@pytest.fixture(scope="module")
def tiny_tokenizer(tiny_model_dir):
    return AutoTokenizer.from_pretrained(tiny_model_dir)


@pytest.fixture(scope="module")
def tiny_model(tiny_model_dir):
    return AutoModelForSequenceClassification.from_pretrained(tiny_model_dir)


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestCharlsonDataset:
    def test_length(self, tiny_tokenizer):
        labels = np.zeros((len(FAKE_PMHS), NUM_LABELS))
        ds = CharlsonDataset(FAKE_PMHS, labels, tiny_tokenizer, max_length=32)
        assert len(ds) == len(FAKE_PMHS)

    def test_item_keys(self, tiny_tokenizer):
        labels = np.zeros((len(FAKE_PMHS), NUM_LABELS))
        ds = CharlsonDataset(FAKE_PMHS, labels, tiny_tokenizer, max_length=32)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

    def test_label_shape(self, tiny_tokenizer):
        labels = np.random.randint(0, 2, (len(FAKE_PMHS), NUM_LABELS)).astype(float)
        ds = CharlsonDataset(FAKE_PMHS, labels, tiny_tokenizer, max_length=32)
        assert ds[0]["labels"].shape == (NUM_LABELS,)

    def test_input_ids_length(self, tiny_tokenizer):
        labels = np.zeros((len(FAKE_PMHS), NUM_LABELS))
        ds = CharlsonDataset(FAKE_PMHS, labels, tiny_tokenizer, max_length=32)
        assert ds[0]["input_ids"].shape == (32,)


# ---------------------------------------------------------------------------
# Model output shape
# ---------------------------------------------------------------------------

class TestModelOutput:
    def test_output_shape(self, tiny_model, tiny_tokenizer):
        encoding = tiny_tokenizer(
            FAKE_PMHS[:4], max_length=32, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            outputs = tiny_model(**encoding)
        assert outputs.logits.shape == (4, NUM_LABELS)


# ---------------------------------------------------------------------------
# Threshold optimization
# ---------------------------------------------------------------------------

class TestOptimizeThresholds:
    def test_returns_all_categories(self):
        np.random.seed(42)
        probs = np.random.rand(100, NUM_LABELS)
        labels = (probs > 0.5).astype(int)
        thresholds = optimize_thresholds(probs, labels, CATEGORIES)
        assert set(thresholds.keys()) == set(CATEGORIES)

    def test_threshold_range(self):
        np.random.seed(42)
        probs = np.random.rand(100, NUM_LABELS)
        labels = (probs > 0.5).astype(int)
        thresholds = optimize_thresholds(probs, labels, CATEGORIES)
        for t in thresholds.values():
            assert 0.05 <= t <= 0.95


# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_returns_macro_auprc(self):
        np.random.seed(42)
        logits = np.random.randn(50, NUM_LABELS)
        labels = np.random.randint(0, 2, (50, NUM_LABELS))
        result = compute_metrics_fn((logits, labels))
        assert "macro_auprc" in result
        assert 0.0 <= result["macro_auprc"] <= 1.0


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

class TestFullEvaluation:
    def test_output_rows(self):
        np.random.seed(42)
        probs = np.random.rand(100, NUM_LABELS)
        labels = (probs > 0.5).astype(int)
        thresholds = {cat: 0.5 for cat in CATEGORIES}
        df = full_evaluation(probs, labels, thresholds)
        assert len(df) == NUM_LABELS + 2

    def test_metric_columns(self):
        np.random.seed(42)
        probs = np.random.rand(100, NUM_LABELS)
        labels = (probs > 0.5).astype(int)
        thresholds = {cat: 0.5 for cat in CATEGORIES}
        df = full_evaluation(probs, labels, thresholds)
        for col in ["auroc", "auprc", "precision", "recall", "f1"]:
            assert col in df.columns


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

class TestSelectDevice:
    def test_returns_string(self):
        device = select_device()
        assert device in ("mps", "cuda", "cpu")


# ---------------------------------------------------------------------------
# Save / reload model
# ---------------------------------------------------------------------------

class TestSaveReload:
    def test_reload_produces_same_output(self, tiny_model, tiny_tokenizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_model.save_pretrained(tmpdir)
            tiny_tokenizer.save_pretrained(tmpdir)

            reloaded = AutoModelForSequenceClassification.from_pretrained(tmpdir)
            encoding = tiny_tokenizer(
                ["test text"], max_length=32, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            with torch.no_grad():
                orig_out = tiny_model(**encoding).logits.numpy()
                reload_out = reloaded(**encoding).logits.numpy()
            np.testing.assert_array_almost_equal(orig_out, reload_out)

    def test_thresholds_json_roundtrip(self):
        thresholds = {cat: round(0.3 + 0.02 * i, 2) for i, cat in enumerate(CATEGORIES)}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "thresholds.json")
            with open(path, "w") as f:
                json.dump(thresholds, f)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded == thresholds


# ---------------------------------------------------------------------------
# predict_probs
# ---------------------------------------------------------------------------

class TestPredictProbs:
    def test_output_shape(self, tiny_model, tiny_tokenizer):
        probs = predict_probs(tiny_model, tiny_tokenizer, FAKE_PMHS, max_length=32,
                              batch_size=8, device="cpu")
        assert probs.shape == (len(FAKE_PMHS), NUM_LABELS)

    def test_probs_range(self, tiny_model, tiny_tokenizer):
        probs = predict_probs(tiny_model, tiny_tokenizer, FAKE_PMHS, max_length=32,
                              batch_size=8, device="cpu")
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

class TestListCheckpoints:
    def test_finds_checkpoints(self):
        with tempfile.TemporaryDirectory() as run_dir:
            tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            _create_fake_checkpoint(run_dir, step=100, epoch=1.0, tokenizer=tok)
            _create_fake_checkpoint(run_dir, step=200, epoch=2.0, tokenizer=tok)
            ckpts = list_checkpoints(run_dir)
            assert len(ckpts) == 2
            assert ckpts[0]["step"] == 100
            assert ckpts[1]["step"] == 200

    def test_sorted_by_step(self):
        with tempfile.TemporaryDirectory() as run_dir:
            tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            _create_fake_checkpoint(run_dir, step=300, epoch=3.0, tokenizer=tok)
            _create_fake_checkpoint(run_dir, step=100, epoch=1.0, tokenizer=tok)
            ckpts = list_checkpoints(run_dir)
            assert ckpts[0]["step"] < ckpts[1]["step"]

    def test_empty(self):
        with tempfile.TemporaryDirectory() as run_dir:
            assert list_checkpoints(run_dir) == []


# ---------------------------------------------------------------------------
# Resolve checkpoint
# ---------------------------------------------------------------------------

class TestResolveCheckpoint:
    def test_last(self):
        with tempfile.TemporaryDirectory() as run_dir:
            tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            _create_fake_checkpoint(run_dir, step=100, epoch=1.0, tokenizer=tok)
            _create_fake_checkpoint(run_dir, step=200, epoch=2.0, tokenizer=tok)
            ckpt = resolve_checkpoint(run_dir, "last")
            assert ckpt["step"] == 200

    def test_best_with_state(self):
        with tempfile.TemporaryDirectory() as run_dir:
            tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            ckpt1_path = _create_fake_checkpoint(run_dir, step=100, epoch=1.0, tokenizer=tok)
            _create_fake_checkpoint(run_dir, step=200, epoch=2.0, tokenizer=tok,
                                    best_checkpoint=ckpt1_path)
            ckpt = resolve_checkpoint(run_dir, "best")
            assert ckpt["step"] == 100

    def test_best_fallback_to_last(self):
        with tempfile.TemporaryDirectory() as run_dir:
            tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            _create_fake_checkpoint(run_dir, step=100, epoch=1.0, tokenizer=tok)
            ckpt = resolve_checkpoint(run_dir, "best")
            assert ckpt["step"] == 100

    def test_explicit_name(self):
        with tempfile.TemporaryDirectory() as run_dir:
            tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            _create_fake_checkpoint(run_dir, step=100, epoch=1.0, tokenizer=tok)
            _create_fake_checkpoint(run_dir, step=200, epoch=2.0, tokenizer=tok)
            ckpt = resolve_checkpoint(run_dir, "checkpoint-100")
            assert ckpt["step"] == 100

    def test_not_found_raises(self):
        with tempfile.TemporaryDirectory() as run_dir:
            tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            _create_fake_checkpoint(run_dir, step=100, epoch=1.0, tokenizer=tok)
            with pytest.raises(FileNotFoundError):
                resolve_checkpoint(run_dir, "checkpoint-999")

    def test_no_checkpoints_raises(self):
        with tempfile.TemporaryDirectory() as run_dir:
            with pytest.raises(FileNotFoundError):
                resolve_checkpoint(run_dir, "best")


# ---------------------------------------------------------------------------
# load_run_config
# ---------------------------------------------------------------------------

class TestLoadRunConfig:
    def test_from_file(self):
        with tempfile.TemporaryDirectory() as run_dir:
            cfg = {"model": "my-model", "max_length": 128}
            with open(os.path.join(run_dir, "run_config.json"), "w") as f:
                json.dump(cfg, f)
            model, ml = load_run_config(run_dir)
            assert model == "my-model"
            assert ml == 128

    def test_no_file_uses_defaults(self):
        with tempfile.TemporaryDirectory() as run_dir:
            model, ml = load_run_config(run_dir)
            assert model == "emilyalsentzer/Bio_ClinicalBERT"
            assert ml == 256

    def test_no_file_uses_args(self):
        with tempfile.TemporaryDirectory() as run_dir:
            model, ml = load_run_config(run_dir, args_base_model="custom", args_max_length=64)
            assert model == "custom"
            assert ml == 64


# ---------------------------------------------------------------------------
# load_checkpoint_model_tokenizer
# ---------------------------------------------------------------------------

class TestLoadCheckpointModelTokenizer:
    def test_with_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_tiny_model(tmpdir, save_tokenizer=True)
            model, tokenizer = load_checkpoint_model_tokenizer(tmpdir, "bert-base-uncased", NUM_LABELS)
            assert model.config.num_labels == NUM_LABELS
            assert tokenizer is not None

    def test_without_tokenizer_falls_back(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_tiny_model(tmpdir, save_tokenizer=False)
            model, tokenizer = load_checkpoint_model_tokenizer(tmpdir, "bert-base-uncased", NUM_LABELS)
            assert model.config.num_labels == NUM_LABELS
            assert tokenizer is not None


# ---------------------------------------------------------------------------
# save_final
# ---------------------------------------------------------------------------

class TestSaveFinal:
    def test_creates_all_artifacts(self, tiny_model, tiny_tokenizer):
        thresholds = {cat: 0.5 for cat in CATEGORIES}
        with tempfile.TemporaryDirectory() as tmpdir:
            final_dir = os.path.join(tmpdir, "final")
            save_final(tiny_model, tiny_tokenizer, thresholds, final_dir,
                       checkpoint_path="/fake/path", epoch=2, step=500,
                       base_model="test-model", max_length=128)

            assert os.path.exists(os.path.join(final_dir, "thresholds.json"))
            assert os.path.exists(os.path.join(final_dir, "final_config.json"))

            with open(os.path.join(final_dir, "final_config.json")) as f:
                cfg = json.load(f)
            assert cfg["epoch"] == 2
            assert cfg["step"] == 500
            assert cfg["max_length"] == 128

            # Model can be reloaded
            reloaded = AutoModelForSequenceClassification.from_pretrained(final_dir)
            assert reloaded.config.num_labels == NUM_LABELS


# ---------------------------------------------------------------------------
# Finalization integration: single checkpoint
# ---------------------------------------------------------------------------

class TestFinalizeSingleCheckpoint:
    def test_single_checkpoint(self):
        """Finalize with only 1 checkpoint produces correct output."""
        with tempfile.TemporaryDirectory() as run_dir:
            tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            ckpt_path = _create_fake_checkpoint(run_dir, step=100, epoch=1.0, tokenizer=tok)

            model, tokenizer = load_checkpoint_model_tokenizer(
                ckpt_path, "bert-base-uncased", NUM_LABELS,
            )
            probs = predict_probs(model, tokenizer, FAKE_PMHS[:10], max_length=32,
                                  batch_size=8, device="cpu")
            assert probs.shape == (10, NUM_LABELS)

            labels = np.random.randint(0, 2, (10, NUM_LABELS))
            thresholds = optimize_thresholds(probs, labels, CATEGORIES)
            metrics = full_evaluation(probs, labels, thresholds)
            assert len(metrics) == NUM_LABELS + 2

            final_dir = os.path.join(run_dir, "final")
            save_final(model, tokenizer, thresholds, final_dir,
                       checkpoint_path=ckpt_path, epoch=1, step=100,
                       base_model="bert-base-uncased", max_length=32)
            assert os.path.exists(os.path.join(final_dir, "thresholds.json"))


# ---------------------------------------------------------------------------
# Finalization integration: 3 checkpoints, pick best
# ---------------------------------------------------------------------------

class TestFinalizeMultipleCheckpoints:
    def test_three_checkpoints_pick_best(self):
        """With 3 checkpoints, resolve_checkpoint('best') with best_model_checkpoint works."""
        with tempfile.TemporaryDirectory() as run_dir:
            tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            ckpt1 = _create_fake_checkpoint(run_dir, step=100, epoch=1.0, tokenizer=tok)
            _create_fake_checkpoint(run_dir, step=200, epoch=2.0, tokenizer=tok)
            _create_fake_checkpoint(run_dir, step=300, epoch=3.0, tokenizer=tok,
                                    best_checkpoint=ckpt1)

            ckpt = resolve_checkpoint(run_dir, "best")
            assert ckpt["step"] == 100

    def test_all_checkpoints_evaluation(self):
        """Evaluate all checkpoints and compare macro_auprc."""
        with tempfile.TemporaryDirectory() as run_dir:
            tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            _create_fake_checkpoint(run_dir, step=100, epoch=1.0, tokenizer=tok)
            _create_fake_checkpoint(run_dir, step=200, epoch=2.0, tokenizer=tok)
            _create_fake_checkpoint(run_dir, step=300, epoch=3.0, tokenizer=tok)

            ckpts = list_checkpoints(run_dir)
            assert len(ckpts) == 3

            labels = np.random.randint(0, 2, (10, NUM_LABELS))
            results = []
            for c in ckpts:
                model, tokenizer = load_checkpoint_model_tokenizer(
                    c["path"], "bert-base-uncased", NUM_LABELS,
                )
                probs = predict_probs(model, tokenizer, FAKE_PMHS[:10],
                                      max_length=32, batch_size=8, device="cpu")
                thresholds = optimize_thresholds(probs, labels, CATEGORIES)
                metrics = full_evaluation(probs, labels, thresholds)
                macro_auprc = metrics[metrics["category"] == "macro_avg"].iloc[0]["auprc"]
                results.append((c["step"], macro_auprc))

            assert len(results) == 3
            # All AUPRCs should be numeric
            for step, auprc in results:
                assert isinstance(auprc, float)
