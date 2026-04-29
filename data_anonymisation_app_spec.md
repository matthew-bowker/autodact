# Data Anonymisation App — Product & Technical Specification

## 1. Overview

A lightweight, self-contained desktop application that anonymises personally identifiable information (PII) in documents using a combination of regex pattern matching and a local DeBERTa-v3 token-classification model. Designed to run entirely on consumer hardware with no cloud dependencies. The app produces anonymised output files alongside a researcher lookup table (CSV) mapping original PII terms to their anonymised replacements.

---

## 2. Goals & Constraints

- **Fully offline** — no data leaves the user's machine.
- **Consumer hardware** — must run comfortably on a machine with 8–16 GB RAM. The encoder is a small token-classification model (~750 MB; DeBERTa-v3-base, ~86M parameters) that runs on CPU, Apple MPS, or CUDA where available.
- **Cross-platform** — Windows, macOS, Linux. Distributed as a self-contained downloadable package (not via app stores).
- **Simple UX** — drag-and-drop files, configure minimal settings, get anonymised output.

---

## 3. Technology Stack

| Layer | Technology |
|---|---|
| UI framework | PyQt6 / PySide6 |
| Encoder inference | transformers + PyTorch |
| Document parsing | openpyxl (xlsx), python-docx (docx), built-in csv/txt |
| Document output | Same libraries for format-preserving output; csv/txt via built-in |
| Regex engine | Python `re` standard library |
| Packaging | PyInstaller or Nuitka → standalone .app / .exe / AppImage |

---

## 4. Supported Formats

### 4.1 Input

| Format | Notes |
|---|---|
| `.txt` | Plain text, read line by line |
| `.csv` | Parsed row by row; each cell treated as a text segment |
| `.xlsx` | Parsed sheet by sheet, row by row, cell by cell |
| `.docx` | Parsed paragraph by paragraph |

### 4.2 Output

User selects output format per session:

- **Preserve original format** — anonymised `.xlsx` → `.xlsx`, `.docx` → `.docx`, etc.
- **Plain text / CSV** — all output written as `.txt` or `.csv` regardless of input.

### 4.3 Lookup Table

Always output as `.csv` with columns:

```
original_term, anonymised_term, pii_category, source_file, first_seen_line
```

Example:

```
Jane Smith,[NAME 1],NAME,interview_notes.docx,3
Acme Corp,[ORG 1],ORG,interview_notes.docx,7
jane.smith@acme.com,[EMAIL 1],EMAIL,interview_notes.docx,12
```

---

## 5. PII Detection — Hybrid Approach

### 5.1 Layer 1: Regex (Structured PII)

Run first on each line before the LLM sees it. Fast, deterministic, and costs zero inference time.

| PII Type | Tag Format | Example Pattern |
|---|---|---|
| Email addresses | `[EMAIL N]` | `\b[\w.-]+@[\w.-]+\.\w{2,}\b` |
| Phone numbers | `[PHONE N]` | Regional patterns (UK, US, intl with `+`) |
| National ID numbers | `[ID N]` | Country-specific (NI number, SSN, etc.) |
| Dates of birth | `[DOB N]` | `\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b` and variants |
| Postcodes / ZIP codes | `[POSTCODE N]` | Regional patterns |
| IP addresses | `[IP N]` | IPv4 / IPv6 patterns |
| URLs | `[URL N]` | `https?://...` patterns |

The regex layer tags each match immediately, adds it to the lookup table, and performs a find-and-replace-all across the entire remaining document before proceeding.

### 5.2 Layer 2: DeBERTa Token Classification (Contextual PII)

Handles PII that requires semantic understanding and cannot be reliably caught with patterns. The model is a fine-tuned mDeBERTa-v3-base encoder (default: ``iiiorg/piiranha-v1-detect-personal-information``) running via HuggingFace ``transformers`` + PyTorch.

| PII Type | Tag Format |
|---|---|
| Person names | `[NAME N]` |
| Organisation names | `[ORG N]` |
| Location names (cities, countries, landmarks, addresses) | `[LOCATION N]` |
| Job titles tied to identifiable individuals | `[JOBTITLE N]` |
| Structured IDs the encoder catches (SSN, account numbers, etc.) | `[ID N]` |

---

## 6. Core Processing Architecture

### 6.1 Pipeline Overview

```
┌─────────────┐
│  Load File   │
│  (parse to   │
│  line list)  │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  Regex Pass       │  For each line:
│  (structured PII) │  → detect, tag, add to lookup,
│                    │    find-and-replace-all in full doc
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  DeBERTa Pass     │  For each line:
│  (contextual PII) │  → token-classify, merge BIO spans,
│                    │    register findings in lookup
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  (Optional)       │
│  Human Review UI  │
└──────┬───────────┘
       │
       ▼
┌─────────────┐
│  Write Output │
│  + Lookup CSV │
└─────────────┘
```

### 6.2 Line-by-Line Processing with Find-and-Replace-All

This is the central design decision. For each line being processed:

1. **Detect** — identify a PII entity (via regex, dictionary, or DeBERTa).
2. **Register** — check the lookup table. If the entity already exists, reuse its tag. If new, assign the next sequential tag (e.g. `[NAME 4]`).
3. **Replace-all** — after every detector layer has run, every lookup entry is replaced across the entire document, longest-first to prevent partial-word contamination.
4. **Proceed** — orchestrator advances to the next layer.

Findings from every layer are aggregated into the lookup table and only applied once at the end of the pipeline (additive replay). This decouples detection from replacement.

### 6.3 DeBERTa Inference

The encoder processes one **line at a time** through a single forward pass (no autoregressive generation):

- Lines longer than ``max_line_chars`` (default 500 characters) are split via ``split_long_lines`` so each chunk fits comfortably inside DeBERTa's 512-token window. Sub-lines are reunified for output.
- Identical line texts share a result cache, so duplicate freetext rows in CSV/XLSX cost only one inference.
- Output is per-token logits; the detector argmaxes to label IDs, applies a confidence threshold (default 0.5), strips B-/I- BIO prefixes, and merges consecutive same-category tokens into character-level spans using offset mapping.
- Spans shorter than 3 characters or matching nothing in the original line are discarded.

### 6.4 Token Classification Output

The detector emits ``(original_text, internal_category)`` pairs. Internal categories are mapped from the model's labels via a lookup (e.g. Piranha's ``I-GIVENNAME`` and ``I-SURNAME`` both map to ``NAME``; ``I-ZIPCODE`` to ``POSTCODE``; ``I-CREDITCARDNUMBER`` and ``I-SOCIALNUM`` to ``ID``). Unmapped labels are dropped, so swapping in a different fine-tuned model is just a matter of updating ``_LABEL_TO_CATEGORY``.

### 6.5 Error Handling

- If a forward pass raises (e.g. CUDA OOM, MPS backend error), the line is flagged and skipped.
- The detector is duck-typed to the orchestrator's contextual-detector slot, so a degraded fallback (e.g. running on CPU) can be swapped in without reaching the orchestrator.
- Rate of inference: ~0.05–0.3 seconds per line on Apple Silicon (MPS) for short documents, ~0.1–0.5s on CPU. A 1,000-line document typically completes in under 5 minutes.

---

## 7. Lookup Table Management

### 7.1 Lifecycle

At session start, the user selects one of:

- **Persist across files** — one lookup table spans all files processed in the session. "Jane Smith" is `[NAME 1]` everywhere.
- **Reset per file** — the lookup table resets when a new file is loaded. "Jane Smith" could be `[NAME 1]` in file A and `[NAME 1]` again (independently) in file B.

### 7.2 Tag Numbering

Each PII category has its own independent counter:

- `[NAME 1]`, `[NAME 2]`, `[NAME 3]` …
- `[LOCATION 1]`, `[LOCATION 2]` …
- `[ORG 1]`, `[ORG 2]` …
- `[EMAIL 1]`, `[EMAIL 2]` …
- etc.

### 7.3 Deduplication

Before assigning a new tag, the engine checks the lookup table for an exact match (case-insensitive). Partial matches (e.g. "Jane" vs "Jane Smith") are not auto-merged — the encoder may surface both, and they receive separate tags. The user can merge them manually in the review step or post-hoc in the CSV.

---

## 8. Optional Human Review Step

Toggled on/off in the UI settings. When enabled:

### 8.1 Review Interface

After processing completes (or in batches during processing), the user sees a table:

| Line | Original | Replacement | Category | Action |
|---|---|---|---|---|
| 3 | Jane Smith | [NAME 1] | NAME | ✅ Accept / ❌ Reject / ✏️ Edit |
| 7 | Acme Corp | [ORG 1] | ORG | ✅ Accept / ❌ Reject / ✏️ Edit |
| 14 | The Red Lion | [LOCATION 3] | LOCATION | ✅ Accept / ❌ Reject / ✏️ Edit |

### 8.2 Behaviours

- **Accept** — replacement stands.
- **Reject** — original text is restored in the document; entry is removed from the lookup table. The find-and-replace-all is reversed for this entity.
- **Edit** — user can change the category or the tag (e.g. re-categorise a LOCATION as an ORG, or merge two tags).
- Bulk accept/reject by category.

---

## 9. User Interface

### 9.1 Main Window

```
┌──────────────────────────────────────────────┐
│  Data Anonymiser                        — □ ✕ │
├──────────────────────────────────────────────┤
│                                              │
│  ┌────────────────────────────────────────┐  │
│  │                                        │  │
│  │       Drag & drop files here           │  │
│  │       (.txt .csv .xlsx .docx)          │  │
│  │                                        │  │
│  │         or click to browse             │  │
│  └────────────────────────────────────────┘  │
│                                              │
│  Settings:                                   │
│  ┌────────────────────────────────────────┐  │
│  │ Output format:  ○ Preserve  ○ CSV/TXT  │  │
│  │ Lookup table:   ○ Persist   ○ Per-file │  │
│  │ Review step:    [toggle on/off]        │  │
│  │ LLM model:      [dropdown / path]     │  │
│  │ Window size:     [2] lines above/below │  │
│  └────────────────────────────────────────┘  │
│                                              │
│  [▶ Start Anonymisation]                     │
│                                              │
│  Progress: ████████░░░░░░  Line 412/1038     │
│  Found: 12 names, 5 locations, 3 orgs...     │
│                                              │
├──────────────────────────────────────────────┤
│  Output files:                               │
│  📄 interview_notes_anon.docx    [Open]      │
│  📊 lookup_table.csv             [Open]      │
└──────────────────────────────────────────────┘
```

### 9.2 Settings Persistence

Settings (output format preference, review toggle, model path, window size) persist between sessions via a local config file (`~/.config/data-anonymiser/config.json` or platform equivalent).

---

## 10. File Processing Detail

### 10.1 Text Extraction by Format

| Format | Strategy |
|---|---|
| `.txt` | Read line by line. Each line is a processing unit. |
| `.csv` | Read row by row. Concatenate all cell values in a row into a single string, separated by ` \| `. Process as one line. On output, map replacements back to individual cells. |
| `.xlsx` | Iterate sheets → rows → cells. Same concatenation strategy as CSV. Preserve sheet/row/cell structure on output. |
| `.docx` | Extract paragraphs via python-docx. Each paragraph is a processing unit (equivalent to a "line"). Preserve formatting on output where possible. |

### 10.2 Output Writing

- **Preserve format**: reconstruct the document using the same library that parsed it, replacing text in-place.
- **CSV/TXT output**: write the anonymised line list as a flat file, one line per row.

Anonymised files are named: `{original_filename}_anon.{ext}`
Lookup tables are named: `{original_filename}_lookup.csv` (per-file mode) or `session_lookup.csv` (persist mode).

---

## 11. Edge Cases & Handling

| Scenario | Handling |
|---|---|
| PII spans multiple cells in a spreadsheet | Unlikely but possible. Each cell is processed independently. If a name is split ("Jane" in A1, "Smith" in B1), the row-concatenation step gives the encoder the full context to detect "Jane Smith", and the replacement maps back to both cells. |
| Same name, different people | They receive the same tag. The tool anonymises text, not identity. If differentiation matters, the user handles it in the review step. |
| Partial matches ("Jane" vs "Jane Smith") | Both are registered separately. No auto-merging. |
| Encoder predicts a span that isn't in the line | The detector validates each merged span against the source line and discards mismatches before registering. |
| Very long lines (e.g. a paragraph in a .txt) | If a line exceeds a configurable character limit (default: 500 chars), split into sub-segments at sentence boundaries. Process each sub-segment as a "line". |
| Empty lines / non-text content | Skip. Log if relevant. |
| Password-protected files | Reject with a user-facing error message. |
| Mixed-language content | The mDeBERTa-v3 backbone is multilingual, so detection works across many languages. Regex patterns may still miss non-Latin structured PII — document this as a known limitation. |

---

## 12. Performance Estimates

Based on the mDeBERTa-v3-base encoder running on Apple Silicon (MPS):

| Document Size | Estimated Processing Time |
|---|---|
| 100 lines | < 30 seconds |
| 500 lines | 1–2 minutes |
| 1,000 lines | 2–5 minutes |
| 5,000 lines | 10–25 minutes |

CPU is roughly 3–5× slower. The regex pass is near-instant and runs before the encoder pass. Progress indication is still helpful for longer documents.

---

## 13. Packaging & Distribution

| Platform | Package Format | Signing |
|---|---|---|
| macOS | `.dmg` containing `.app` | Code-signed + notarised with Apple Developer cert |
| Windows | `.exe` installer (NSIS or similar) | Code-signed with EV or standard cert (optional but recommended) |
| Linux | AppImage | No signing required |

Built via PyInstaller or Nuitka. Single command build per platform via a Makefile or build script.

---

## 14. Future Considerations (Out of Scope for v1)

- Batch processing of entire folders.
- PDF input support (requires OCR layer).
- Custom regex pattern editor in the UI.
- Confidence scoring surfaced per detection.
- Re-identification tool (apply lookup table in reverse to restore original data).
- App Store distribution (would require migration to Tauri shell with Python sidecar).
- ONNX Runtime export for further CPU speedup and a lighter desktop bundle.
