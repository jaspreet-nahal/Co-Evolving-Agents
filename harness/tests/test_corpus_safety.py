"""Part I: real-mode benchmark execution must never silently use sample data.

These tests never touch the network: `mode="real"` is expected to fail
fast (raising RealDataUnavailableError) whenever no local data/corpus was
supplied, and the HF-fallback path is monkeypatched to fail immediately
instead of actually attempting a download.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest

from harness.benchmarks.qampari import QampariBenchmark, RealDataUnavailableError as QampariRealDataError
from harness.benchmarks.financebench import FinanceBenchBenchmark, RealDataUnavailableError as FinanceBenchRealDataError
from harness.benchmarks.browsecomp_plus import BrowseCompPlusBenchmark, RealDataUnavailableError as BrowseCompRealDataError


class TestQampariCorpusSafety:

    def test_test_mode_falls_back_to_sample_data(self, monkeypatch):
        benchmark = QampariBenchmark(mode="test")
        monkeypatch.setattr(benchmark, "_load_from_hf", lambda max_examples: (_ for _ in ()).throw(RuntimeError("no network in test")))
        examples = benchmark.load_data(max_examples=3)
        assert len(examples) > 0  # sample fallback succeeded, as intended in test mode

    def test_real_mode_rejects_missing_data_path(self, monkeypatch):
        benchmark = QampariBenchmark(mode="real")
        monkeypatch.setattr(benchmark, "_load_from_hf", lambda max_examples: (_ for _ in ()).throw(RuntimeError("no network in test")))
        with pytest.raises(QampariRealDataError):
            benchmark.load_data(max_examples=3)

    def test_real_mode_rejects_missing_corpus_path(self):
        benchmark = QampariBenchmark(mode="real")
        with pytest.raises(QampariRealDataError):
            benchmark.load_corpus()

    def test_real_mode_never_produces_sample_examples(self, monkeypatch):
        benchmark = QampariBenchmark(mode="real")
        monkeypatch.setattr(benchmark, "_load_from_hf", lambda max_examples: (_ for _ in ()).throw(RuntimeError("no network in test")))
        try:
            benchmark.load_data(max_examples=3)
        except QampariRealDataError:
            pass
        assert benchmark.examples == []  # _create_sample_data was never reached


class TestFinanceBenchCorpusSafety:

    def test_test_mode_falls_back_to_sample_data(self, monkeypatch):
        benchmark = FinanceBenchBenchmark(mode="test")
        monkeypatch.setattr(benchmark, "_load_from_hf", lambda max_examples, split: (_ for _ in ()).throw(RuntimeError("no network in test")))
        examples = benchmark.load_data(max_examples=3)
        assert len(examples) > 0

    def test_real_mode_rejects_missing_data(self, monkeypatch):
        benchmark = FinanceBenchBenchmark(mode="real")
        monkeypatch.setattr(benchmark, "_load_from_hf", lambda max_examples, split: (_ for _ in ()).throw(RuntimeError("no network in test")))
        with pytest.raises(FinanceBenchRealDataError):
            benchmark.load_data(max_examples=3)

    def test_real_mode_rejects_missing_corpus(self, monkeypatch):
        benchmark = FinanceBenchBenchmark(mode="real")
        monkeypatch.setattr(benchmark, "_load_from_hf_corpus", lambda: (_ for _ in ()).throw(RuntimeError("no network in test")))
        with pytest.raises(FinanceBenchRealDataError):
            benchmark.load_corpus()


class TestBrowseCompPlusCorpusSafety:

    def test_test_mode_falls_back_to_sample_data(self, monkeypatch):
        benchmark = BrowseCompPlusBenchmark(mode="test")
        monkeypatch.setattr(benchmark, "_load_from_hf_encrypted", lambda max_examples: (_ for _ in ()).throw(RuntimeError("no network in test")))
        examples = benchmark.load_data(max_examples=3)
        assert len(examples) > 0

    def test_real_mode_rejects_missing_data(self, monkeypatch):
        benchmark = BrowseCompPlusBenchmark(mode="real")
        monkeypatch.setattr(benchmark, "_load_from_hf_encrypted", lambda max_examples: (_ for _ in ()).throw(RuntimeError("no network in test")))
        with pytest.raises(BrowseCompRealDataError):
            benchmark.load_data(max_examples=3)

    def test_real_mode_rejects_missing_corpus(self):
        benchmark = BrowseCompPlusBenchmark(mode="real")
        with pytest.raises(BrowseCompRealDataError):
            benchmark.load_corpus()

    def test_real_mode_prebuilt_index_dir_still_raises_not_implemented(self, tmp_path):
        # A real pre-built index dir existing is a different failure mode
        # (NotImplementedError, not RealDataUnavailableError) from the
        # prior P0 pass -- confirm mode="real" doesn't change that behavior.
        index_dir = tmp_path / "prebuilt_index"
        index_dir.mkdir()
        benchmark = BrowseCompPlusBenchmark(index_dir=str(index_dir), mode="real")
        with pytest.raises(NotImplementedError):
            benchmark.load_corpus()


class TestModeValidation:

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            QampariBenchmark(mode="bogus")
        with pytest.raises(ValueError):
            FinanceBenchBenchmark(mode="bogus")
        with pytest.raises(ValueError):
            BrowseCompPlusBenchmark(mode="bogus")

    def test_factories_accept_mode(self):
        from harness.benchmarks.qampari import create_qampari_benchmark
        from harness.benchmarks.financebench import create_financebench_benchmark
        from harness.benchmarks.browsecomp_plus import create_browsecomp_plus_benchmark

        assert create_qampari_benchmark(mode="real").mode == "real"
        assert create_financebench_benchmark(mode="real").mode == "real"
        assert create_browsecomp_plus_benchmark(mode="real").mode == "real"
