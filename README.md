# Autodact

Privacy-first desktop application for anonymising personally identifiable information (PII) in documents. All processing happens locally on your machine — no data ever leaves your computer.

## Features

- **Drag-and-drop** CSV, XLSX, DOCX, and TXT files
- **Three-layer detection**: regex patterns, spaCy NER (via Microsoft Presidio), and a local LLM
- **Human review step** to verify and correct detections before saving
- **Consistent lookup tables** so the same name always maps to the same tag
- **Pause and resume** long-running jobs, even after closing the app
- **Fully offline** — models run locally via llama.cpp

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

On first launch, the app will download a PII detection model (~540 MB) and a language model (~560 MB). This only happens once.

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

Autodact uses a multi-pass pipeline to detect PII:

1. **Column detection** — identifies name/email/phone columns in structured data
2. **Regex pre-pass** — catches emails, phone numbers, postcodes, NI numbers, and other structured patterns
3. **NER pass** — Microsoft Presidio with spaCy finds names, organisations, locations, and dates
4. **LLM pass** — a local Distil-PII model (SmolLM2 135M or Llama 3.2 1B) catches context-dependent PII the other layers miss

Each detected entity gets a consistent replacement tag (e.g. `[NAME 1]`, `[ORG 3]`) tracked in a lookup table, so the same person always maps to the same tag across the entire document.

## Configuration

Settings are accessible from the main window:

- **Output format**: preserve original format or convert to plain text
- **Model selection**: Fast (135M params) or Standard (1B params)
- **Detection categories**: toggle names, organisations, locations, job titles
- **Context window**: number of surrounding lines sent to the LLM for context
- **Human review**: enable/disable the review step before saving

## License

[MIT](LICENSE)
