# Technical Reference: Lexical Reduction and Convolution (LRC)

## Overview

**Lexical Reduction and Convolution** is a bidirectional methodology for processing and compressing semantic information. Rather than compressing text at character or token level, this framework operates on meaning.

The process alternates between:

1. Expansion: unpack lexical input into a structured semantic cloud.
2. Convolution: reduce the semantic cloud to the most semantically faithful lexical root.

## Core Mechanisms

### Lexical Expansion (Semantic Unpacking)

- **Process:** Expand a source word, phrase, or sentence into semantic components.
- **Mechanism:** Use dictionary and optional LLM context to produce a structured cloud.
- **Output:** Definitions, connotations, related concepts, constraints, and negative constraints.

### Lexical Convolution (Semantic Reduction)

- **Process:** Compress semantic cloud back into lexical candidates.
- **Mechanism:** Score candidates by definition similarity, lexical relations, keyword overlap, POS compatibility, concision, and ambiguity penalties.
- **Output:** Winner + top-k alternatives with score breakdown and confidence.

## Workflow

1. Input word/phrase/sentence.
2. Expand into structured semantic cloud.
3. Generate and score candidate lexical items.
4. Return winner + alternatives.
5. Optionally recurse over multiple iterations for stability analysis.

## Recursive Dynamics (Milestone 3 Focus)

Recursive expand -> reduce analysis tracks:

- Fixed-point convergence.
- Concept oscillation/cycles.
- Semantic drift.
- Lexical entropy trends.
- Attractor basins across many seed terms.

## Mode Matrix (0-4)

- **Mode 0:** Deterministic baseline only.
- **Mode 1:** LLM expansion + deterministic reduction.
- **Mode 2:** Deterministic expansion + LLM candidate proposal + deterministic rerank.
- **Mode 3:** Deterministic top-k + LLM reranking.
- **Mode 4:** LLM expansion + LLM proposal + LLM rerank + deterministic fallback validation.

All non-zero modes fall back gracefully to deterministic behavior if provider config is missing or provider calls fail.

## Architecture

```text
src/lrc_poc/
  lexicon.py
  expand.py
  candidates.py
  score.py
  reduce.py
  recurse.py
  attractor.py
  pipeline.py
  cli.py
  llm/
  dashboard/app.py
```

## Environment Setup

1. Create conda environment:

```powershell
conda create -n lrc python=3.12 -y
conda activate lrc
```

2. Install package and dev dependencies:

```powershell
pip install -e .[dev]
```

3. Download NLTK resources:

```powershell
python -m nltk.downloader wordnet omw-1.4 punkt averaged_perceptron_tagger
```

## Config Setup (Template -> Runtime)

Tracked template file:

- `config/models.template.yaml`

Runtime file (intentionally untracked):

- `config/models.yaml`

Create runtime file by removing `template` from the filename:

```powershell
Copy-Item config/models.template.yaml config/models.yaml
```

Then place real keys in `config/models.yaml` only.

Template structure:

```yaml
# Default Models
defaults:
  model_name: "" # e.g., "gemini-3-pro-preview"
  image_model_name: "" # e.g., "gemini-3-pro-image-preview"

# API Keys. If you are using all gemini models, you can leave the other keys empty.
api_keys:
  google_api_key: ""
  openai_api_key: ""
  anthropic_api_key: ""
```

If `defaults.model_name` is left empty, the Google provider auto-discovers a compatible Gemini model that supports `generateContent`.

Config loading behavior:

1. Use `config/models.yaml` if present.
2. Otherwise fallback to `config/models.template.yaml`.

## CLI Usage

Run once (sentence input):

```powershell
python -m lrc_poc.cli run-once --text "a feeling of loss tied specifically to death and emotional absence" --mode 0
```

Run recursion (sentence input):

```powershell
python -m lrc_poc.cli recurse --text "a positive emotional state with pleasure and contentment" --iterations 10 --mode 0
```

Run attractor map:

```powershell
python -m lrc_poc.cli map --seed-count 200 --iterations 10 --mode 0 --out artifacts
```

Word and short-phrase inputs are also supported, but sentence inputs are first-class and covered by tests.

## Output Interpretation

- `lexical_entropy` is a normalized uncertainty score over top-k candidate scores at each recursion step.
- Higher entropy means several candidates were similarly plausible.
- Lower entropy means the reducer had a clear winner.

Why a word can map to a different word:

- Milestone 3 studies semantic attractors under expansion->reduction recursion.
- The reducer projects meaning onto the best lexical fit, which can move from the literal input token to a nearby conceptual root.
- In Mode 2 and Mode 4, LLM short-phrase candidates are preserved to support more coherent sentence-level compression.

## Dashboard

```powershell
streamlit run src/lrc_poc/dashboard/app.py
```

The dashboard provides:

- Single reduction with top-k ranking.
- Recursion trajectory and drift/entropy plotting.
- Attractor basin mapping with artifact downloads.

## Testing

Run full suite:

```powershell
pytest
```

Coverage includes:

- Deterministic logic/unit tests.
- Gold-case semantic accuracy regression tests.
- Recursion/cycle/fixed-point validation tests.
- Mode 1-4 fallback and schema parity tests.
- CLI end-to-end tests.
- Dashboard smoke tests.

## Governance and ADRs

ADRs live in:

- [docs/adr/README.md](docs/adr/README.md)

First ADR:

- [docs/adr/0001-change-documentation-and-full-update-rule.md](docs/adr/0001-change-documentation-and-full-update-rule.md)

That ADR enforces documentation and relevant-file updates for all behavioral changes.
