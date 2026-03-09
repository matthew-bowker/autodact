# Data Anonymisation App — Product & Technical Specification

## 1. Overview

A lightweight, self-contained desktop application that anonymises personally identifiable information (PII) in documents using a combination of regex pattern matching and a local LLM. Designed to run entirely on consumer hardware with no cloud dependencies. The app produces anonymised output files alongside a researcher lookup table (CSV) mapping original PII terms to their anonymised replacements.

---

## 2. Goals & Constraints

- **Fully offline** — no data leaves the user's machine.
- **Consumer hardware** — must run comfortably on a machine with 8–16 GB RAM. The LLM should be a small quantised model (e.g. 1–3B parameters, GGUF format).
- **Cross-platform** — Windows, macOS, Linux. Distributed as a self-contained downloadable package (not via app stores).
- **Simple UX** — drag-and-drop files, configure minimal settings, get anonymised output.

---

## 3. Technology Stack

| Layer | Technology |
|---|---|
| UI framework | PyQt6 / PySide6 |
| LLM inference | llama-cpp-python |
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

### 5.2 Layer 2: LLM (Contextual PII)

Handles PII that requires semantic understanding and cannot be reliably caught with patterns.

| PII Type | Tag Format |
|---|---|
| Person names | `[NAME N]` |
| Organisation names | `[ORG N]` |
| Location names (cities, countries, landmarks, addresses) | `[LOCATION N]` |
| Job titles tied to identifiable individuals | `[JOBTITLE N]` |
| Any other contextual identifiers the LLM flags | `[OTHER N]` |

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
│  LLM Pass         │  For each line (sliding window):
│  (contextual PII) │  → detect, tag, add to lookup,
│                    │    find-and-replace-all in full doc
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

1. **Detect** — identify a PII entity (via regex or LLM).
2. **Register** — check the lookup table. If the entity already exists, reuse its tag. If new, assign the next sequential tag (e.g. `[NAME 4]`).
3. **Replace-all** — immediately replace **every occurrence** of this entity across the **entire document** (all lines, not just the current one). This ensures consistency and means the LLM never sees the same PII term twice in future lines.
4. **Proceed** — move to the next detection on the current line, then the next line.

This replace-all-before-advancing approach is critical: it reduces the LLM's exposure to repeated PII and ensures the document is progressively cleaner as processing advances.

### 6.3 Sliding Window for LLM Context

The LLM processes one **focus line** at a time but receives a sliding window for context:

```
[context line n-2]
[context line n-1]
>>> [FOCUS LINE n] <<<
[context line n+1]
[context line n+2]
```

- **Window size**: 2 lines above + 2 lines below (configurable, default 5-line window).
- **Only the focus line is analysed** for new PII. Context lines are read-only for comprehension.
- Context lines will already have earlier PII replaced (due to the replace-all-before-advancing approach), which is desirable — it reduces noise and keeps the prompt small.

### 6.4 LLM Prompt Design

The prompt sent per focus line should be minimal and structured to keep the small model on task:

```
SYSTEM:
You are a PII detector. You will be given a focus line surrounded by
context lines. Identify any person names, organisation names, locations,
job titles, or other identifying information in the FOCUS LINE only.

Respond ONLY with a JSON array. Each entry should have:
- "original": the exact text found
- "category": one of NAME, ORG, LOCATION, JOBTITLE, OTHER

If no PII is found, respond with an empty array: []

USER:
Context:
- [already anonymised line n-2]
- [already anonymised line n-1]

Focus line:
- [current line to analyse]

Context:
- [already anonymised line n+1]
- [already anonymised line n+2]
```

Expected response:

```json
[
  {"original": "Jane Smith", "category": "NAME"},
  {"original": "Manchester", "category": "LOCATION"}
]
```

This keeps the prompt short (well within a 2K context window) and the expected output structured and parseable.

### 6.5 LLM Response Parsing & Error Handling

- Parse the response as JSON. If parsing fails, retry once with the same prompt.
- If retry fails, log the line number and flag it for manual review.
- Validate that each `"original"` string actually appears in the focus line (guard against hallucinated entities).
- Rate of inference: expect ~0.5–2 seconds per line on CPU depending on model size. For a 1,000-line document, this is roughly 8–30 minutes. The UI should show a progress bar with per-line status.

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

Before assigning a new tag, the engine checks the lookup table for an exact match (case-insensitive). Partial matches (e.g. "Jane" vs "Jane Smith") are not auto-merged — the LLM may surface both, and they receive separate tags. The user can merge them manually in the review step or post-hoc in the CSV.

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
| PII spans multiple cells in a spreadsheet | Unlikely but possible. Each cell is processed independently. If a name is split ("Jane" in A1, "Smith" in B1), the row-concatenation step gives the LLM the full context to detect "Jane Smith", and the replacement maps back to both cells. |
| Same name, different people | They receive the same tag. The tool anonymises text, not identity. If differentiation matters, the user handles it in the review step. |
| Partial matches ("Jane" vs "Jane Smith") | Both are registered separately. No auto-merging. |
| LLM hallucinates a PII entity not in the text | Validation step checks that `"original"` exists in the focus line. Hallucinated entries are discarded. |
| Very long lines (e.g. a paragraph in a .txt) | If a line exceeds a configurable character limit (default: 500 chars), split into sub-segments at sentence boundaries. Process each sub-segment as a "line". |
| Empty lines / non-text content | Skip. Log if relevant. |
| Password-protected files | Reject with a user-facing error message. |
| Mixed-language content | The LLM handles this to the extent of its training. Regex patterns may miss non-Latin structured PII. Document this as a known limitation. |

---

## 12. Performance Estimates

Based on a ~1–3B parameter quantised GGUF model on CPU:

| Document Size | Estimated Processing Time |
|---|---|
| 100 lines | 1–3 minutes |
| 500 lines | 4–15 minutes |
| 1,000 lines | 8–30 minutes |
| 5,000 lines | 40 minutes – 2.5 hours |

The regex pass is near-instant and runs before the LLM pass. Progress indication is essential for longer documents.

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
- Confidence scoring per LLM detection.
- Re-identification tool (apply lookup table in reverse to restore original data).
- App Store distribution (would require migration to Tauri shell with Python sidecar).
- GPU acceleration for LLM inference (CUDA / Metal).
