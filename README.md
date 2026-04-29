# Autodact

Privacy-first desktop application for anonymising personally identifiable information (PII) in documents. All processing happens locally on your machine — no data ever leaves your computer.

## Features

- **Drag-and-drop** CSV, XLSX, DOCX, and TXT files
- **Multi-layer detection** pipeline combining regex, NER, dictionaries, syntax analysis, and a local DeBERTa token-classification model
- **Human review step** to verify and correct detections before saving
- **Consistent lookup tables** so the same name always maps to the same tag
- **Pause and resume** long-running jobs, even after closing the app
- **Fully offline** — models run locally via PyTorch + transformers

## System Requirements

- macOS 12+ or Windows 10+
- 8 GB RAM minimum (16 GB recommended)
- ~2 GB disk space (for models and app)
- Internet connection for first-run model downloads

## Installation

### Pre-built binaries (recommended)

Download the latest release from the [Releases](../../releases) page:

- **macOS**: Download `Autodact.dmg`, open it, and drag Autodact to Applications
- **Windows**: Download `Autodact-Windows.zip`, extract it, and run `Autodact.exe`

On first launch, the app will download a DeBERTa PII detection model (~750 MB) and a spaCy language model (~50 MB). This only happens once.

> **macOS note**: If you see "unidentified developer", right-click the app and choose **Open**.

### From source (for development)

```bash
# Clone the repo
git clone https://github.com/yourusername/autodact.git
cd autodact

# Create a virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Download the spaCy language model
python -m spacy download en_core_web_lg

# Run the app
autodact
# or: python -m src.main

# Run tests
pytest
```

## How It Works

Autodact uses a multi-pass pipeline to detect PII. Results from every layer are combined additively on the original text, so nothing detected by one layer is lost by another.

### Detection Techniques

| # | Layer | Technique | What it catches |
|---|-------|-----------|-----------------|
| 1 | **Regex pre-pass** | Pattern matching with validation (Luhn, date plausibility, phone structure) | Emails, phone numbers, UK postcodes, NI numbers, SSNs, credit cards, IBANs, IPs, URLs, passport numbers, dates |
| 2 | **Corporate suffix heuristic** | Regex for company suffixes (Ltd, Inc, GmbH, University, Hospital, etc.) | Organisation names that NER often misclassifies as person names |
| 3 | **Title + next-word heuristic** | "Mr", "Dr", "Prof", etc. (with or without period) followed by 1-3 title-cased words; individual name words registered as aliases | Names preceded by a title that NER may miss (e.g. "Dr Jane Smith", "Mr O'Brien") |
| 4 | **Column detection** (CSV/XLSX) | User maps column headers to PII categories; adjacent NAME columns are combined; email-local-part name variations generated | Structured tabular PII where the column header tells you the category |
| 5 | **Column cross-reference** | Known values from mapped columns are searched in freetext/unmapped columns with case-aware matching | Names and other PII that appear outside their labelled column |
| 6 | **Entropy-based detection** | Shannon entropy calculation on alphanumeric tokens >= 8 chars; requires mixed character types (letters + digits or mixed case); rejects pure-alpha words and programming identifiers | API keys, session tokens, hashes, and random reference IDs (e.g. `sk_live_abc123XYZ`) |
| 7 | **Presidio NER** | Microsoft Presidio with spaCy `en_core_web_md` — neural named entity recognition with overlap resolution and address reclassification | Person names, locations, organisations, and other entities recognised by the NER model |
| 8 | **Syntactic proper noun detection** | spaCy dependency parsing identifies proper nouns (`PROPN`) in meaningful syntactic roles (subjects, objects, appositives) and filters against domain exclusions and a 234K common-word dictionary | Proper nouns in sentence context that NER missed (e.g. "I spoke to Johnson") |
| 9 | **Name dictionary** | 1.7 million first and last names cross-referenced against a common-words dictionary; pair detection for ambiguous words adjacent to high-confidence names | Names that NER and syntax analysis both missed (e.g. uncommon names, names from non-English cultures) |
| 10 | **Custom word lists** | User-provided word lists tied to a PII category, matched via compiled regex with word boundaries | Domain-specific terms the user always wants redacted (e.g. local place names, internal staff names) |
| 11 | **DeBERTa contextual pass** | Local mDeBERTa-v3-base token-classification model (Piranha v1) via transformers + PyTorch; runs on Metal/CUDA where available, CPU otherwise; results cached per unique line | Context-dependent PII that rule-based layers miss — names, addresses, job titles, structured IDs, and ambiguous references |
| 12 | **Fuzzy matching** (opt-in) | Damerau-Levenshtein edit distance against known lookup entries; conservative thresholds (1 for short words, 2 for longer); common-word guard | Misspellings of already-detected PII (e.g. "Johnsen" when "Johnson" is known) |
| 13 | **Phonetic matching** (opt-in) | Metaphone phonetic encoding as a fallback when edit-distance fails; groups words that sound alike; length guard within 3 chars | Sound-alike variants that are too different for edit distance (e.g. "Sean"/"Shawn", "Smith"/"Smyth", "Catherine"/"Katherine") |
| 14 | **Embedding similarity** (opt-in) | Cosine similarity on spaCy `en_core_web_md` 300-dim word vectors (already in memory); threshold >= 0.85; OOV words skipped | Semantic relatives of known PII that other matchers miss |
| 15 | **Post-validation** | Removes stop words, placeholders, pure-numeric false positives, single characters, and NAME-category role words | False positives from any layer |

### Consistency

Each detected entity gets a consistent replacement tag (e.g. `[NAME 1]`, `[ORG 3]`) tracked in a normalised lookup table. Punctuation variants (O'Connor / OConnor), hyphens (Al-Hassan / Al Hassan), and aliases all map to the same tag across the entire document.

## Configuration

Settings are accessible from the main window:

- **Output format**: preserve original format or convert to plain text
- **Model selection**: choose a DeBERTa PII model or point at a local snapshot directory
- **Device**: auto / CPU / MPS (Apple Silicon) / CUDA
- **Detection categories**: toggle names, organisations, locations, job titles
- **Custom word lists**: add your own lists of terms to always redact, each tied to a PII category
- **Fuzzy matching**: opt-in feature to catch misspellings (edit distance), sound-alikes (phonetic), and semantic relatives (embedding similarity) of detected PII
- **Context window**: number of surrounding lines used as context
- **Human review**: enable/disable the review step before saving

## License

[MIT](LICENSE)
