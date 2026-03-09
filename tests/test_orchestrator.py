from pathlib import Path

from src.pipeline.llm_detector import LLMDetector
from src.pipeline.lookup_table import LookupTable
from src.pipeline.orchestrator import Orchestrator

FIXTURES = Path(__file__).parent / "fixtures"


class MockEngine:
    def __init__(self, responses: dict[str, list[dict[str, str]]] | None = None):
        self._responses = responses or {}

    def detect_pii(self, user_prompt: str, categories: list[str]) -> list[dict[str, str]]:
        for key, response in self._responses.items():
            if key in user_prompt:
                return response
        return []


class MockPresidioDetector:
    """No-op Presidio detector for tests that don't need real NER."""

    def pre_detect_emails(self, doc, lookup):
        pass

    def pre_detect_orgs(self, doc, lookup):
        pass

    def pre_detect_patterns(self, doc, lookup):
        pass

    def process(self, doc, lookup, on_progress=None):
        if on_progress:
            total = len(doc.lines)
            for i in range(total):
                on_progress(i + 1, total)


def test_full_pipeline_txt(tmp_path: Path):
    engine = MockEngine(responses={
        "Jane Smith": [{"original": "Jane Smith", "category": "NAME"}],
        "Bob Jones": [{"original": "Bob Jones", "category": "NAME"}],
        "Acme Corp": [{"original": "Acme Corp", "category": "ORG"}],
    })
    orchestrator = Orchestrator(
        presidio_detector=MockPresidioDetector(),
        llm_detector=LLMDetector(engine),
    )
    lookup = LookupTable()
    result = orchestrator.process_file(
        FIXTURES / "sample.txt", lookup, tmp_path, preserve_format=True
    )
    assert result.output_path.exists()
    assert result.output_path.suffix == ".txt"
    assert result.lookup_path.exists()

    output_text = result.output_path.read_text()
    # LLM should have caught names
    assert "Jane Smith" not in output_text
    assert "[NAME" in output_text

    # Lookup should have entries
    assert len(lookup) > 0


def test_full_pipeline_csv(tmp_path: Path):
    engine = MockEngine(responses={
        "Jane Smith": [{"original": "Jane Smith", "category": "NAME"}],
        "Manchester": [{"original": "Manchester", "category": "LOCATION"}],
    })
    orchestrator = Orchestrator(
        presidio_detector=MockPresidioDetector(),
        llm_detector=LLMDetector(engine),
    )
    lookup = LookupTable()
    result = orchestrator.process_file(
        FIXTURES / "sample.csv", lookup, tmp_path, preserve_format=True
    )
    assert result.output_path.suffix == ".csv"
    assert result.output_path.exists()


def test_pipeline_force_txt_output(tmp_path: Path):
    engine = MockEngine()
    orchestrator = Orchestrator(
        presidio_detector=MockPresidioDetector(),
        llm_detector=LLMDetector(engine),
    )
    lookup = LookupTable()
    result = orchestrator.process_file(
        FIXTURES / "sample.csv", lookup, tmp_path, preserve_format=False
    )
    assert result.output_path.suffix == ".txt"


def test_pipeline_progress_callbacks(tmp_path: Path):
    engine = MockEngine()
    orchestrator = Orchestrator(
        presidio_detector=MockPresidioDetector(),
        llm_detector=LLMDetector(engine),
    )
    lookup = LookupTable()
    presidio_progress: list[tuple[int, int]] = []
    llm_progress: list[tuple[int, int]] = []
    orchestrator.process_file(
        FIXTURES / "sample.txt",
        lookup,
        tmp_path,
        on_presidio_progress=lambda c, t: presidio_progress.append((c, t)),
        on_llm_progress=lambda c, t: llm_progress.append((c, t)),
    )
    assert len(presidio_progress) > 0
    assert len(llm_progress) > 0


def test_pipeline_per_file_lookup_isolation(tmp_path: Path):
    engine = MockEngine(responses={
        "Jane Smith": [{"original": "Jane Smith", "category": "NAME"}],
    })
    orchestrator = Orchestrator(
        presidio_detector=MockPresidioDetector(),
        llm_detector=LLMDetector(engine),
    )

    lookup1 = LookupTable()
    orchestrator.process_file(FIXTURES / "sample.txt", lookup1, tmp_path)

    lookup2 = LookupTable()
    orchestrator.process_file(FIXTURES / "sample.txt", lookup2, tmp_path)

    # Both should have entries independently
    assert len(lookup1) > 0
    assert len(lookup2) > 0
