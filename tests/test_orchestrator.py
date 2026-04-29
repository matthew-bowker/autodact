from pathlib import Path

from src.pipeline.lookup_table import LookupTable
from src.pipeline.orchestrator import Orchestrator

FIXTURES = Path(__file__).parent / "fixtures"


class MockDetector:
    """Stand-in for DebertaDetector.

    Registers entities into the lookup table whenever any keyword from the
    ``responses`` map appears in a line's text.  Same role the real DeBERTa
    detector plays in the orchestrator's contextual pass.
    """

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self._responses = responses or {}

    def process(
        self,
        doc,
        lookup,
        on_progress=None,
        mapped_columns=None,
    ) -> list[int]:
        total = len(doc.lines)
        for i, line in enumerate(doc.lines):
            for term, category in self._responses.items():
                if term in line.text:
                    lookup.register(
                        term, category,
                        doc.source_path.name, line.line_number,
                    )
            if on_progress:
                on_progress(i + 1, total)
        return []


class MockPresidioDetector:
    """No-op Presidio detector for tests that don't need real NER."""

    def pre_detect_emails(self, doc, lookup):
        pass

    def pre_detect_orgs(self, doc, lookup):
        pass

    def pre_detect_titled_names(self, doc, lookup):
        pass

    def pre_detect_patterns(self, doc, lookup):
        pass

    def process(self, doc, lookup, on_progress=None):
        if on_progress:
            total = len(doc.lines)
            for i in range(total):
                on_progress(i + 1, total)


def test_full_pipeline_txt(tmp_path: Path):
    detector = MockDetector(responses={
        "Jane Smith": "NAME",
        "Bob Jones": "NAME",
        "Acme Corp": "ORG",
    })
    orchestrator = Orchestrator(
        presidio_detector=MockPresidioDetector(),
        deberta_detector=detector,
    )
    lookup = LookupTable()
    result = orchestrator.process_file(
        FIXTURES / "sample.txt", lookup, tmp_path, preserve_format=True
    )
    assert result.output_path.exists()
    assert result.output_path.suffix == ".txt"
    assert result.lookup_path.exists()

    output_text = result.output_path.read_text()
    # Detector should have caught names
    assert "Jane Smith" not in output_text
    assert "[NAME" in output_text

    # Lookup should have entries
    assert len(lookup) > 0


def test_full_pipeline_csv(tmp_path: Path):
    detector = MockDetector(responses={
        "Jane Smith": "NAME",
        "Manchester": "LOCATION",
    })
    orchestrator = Orchestrator(
        presidio_detector=MockPresidioDetector(),
        deberta_detector=detector,
    )
    lookup = LookupTable()
    result = orchestrator.process_file(
        FIXTURES / "sample.csv", lookup, tmp_path, preserve_format=True
    )
    assert result.output_path.suffix == ".csv"
    assert result.output_path.exists()


def test_pipeline_force_txt_output(tmp_path: Path):
    detector = MockDetector()
    orchestrator = Orchestrator(
        presidio_detector=MockPresidioDetector(),
        deberta_detector=detector,
    )
    lookup = LookupTable()
    result = orchestrator.process_file(
        FIXTURES / "sample.csv", lookup, tmp_path, preserve_format=False
    )
    assert result.output_path.suffix == ".txt"


def test_pipeline_progress_callbacks(tmp_path: Path):
    detector = MockDetector()
    orchestrator = Orchestrator(
        presidio_detector=MockPresidioDetector(),
        deberta_detector=detector,
    )
    lookup = LookupTable()
    presidio_progress: list[tuple[int, int]] = []
    deberta_progress: list[tuple[int, int]] = []
    orchestrator.process_file(
        FIXTURES / "sample.txt",
        lookup,
        tmp_path,
        on_presidio_progress=lambda c, t: presidio_progress.append((c, t)),
        on_deberta_progress=lambda c, t: deberta_progress.append((c, t)),
    )
    assert len(presidio_progress) > 0
    assert len(deberta_progress) > 0


def test_pipeline_per_file_lookup_isolation(tmp_path: Path):
    detector = MockDetector(responses={"Jane Smith": "NAME"})
    orchestrator = Orchestrator(
        presidio_detector=MockPresidioDetector(),
        deberta_detector=detector,
    )

    lookup1 = LookupTable()
    orchestrator.process_file(FIXTURES / "sample.txt", lookup1, tmp_path)

    lookup2 = LookupTable()
    orchestrator.process_file(FIXTURES / "sample.txt", lookup2, tmp_path)

    # Both should have entries independently
    assert len(lookup1) > 0
    assert len(lookup2) > 0
