# Technical Reference: Lexical Reduction and Convolution (LRC)

## Overview

**Lexical Reduction and Convolution** is a bidirectional methodology for processing and compressing semantic information. Rather than compressing text at character or token level, this framework operates on meaning.

The process alternates between:

1. Expansion: unpack lexical input into a structured semantic cloud.
2. Convolution: reduce the semantic cloud to the most semantically faithful lexical root.

## LRC Working Hypothesis

LRC is explored as a **language-mediated operator architecture**: language is not only the medium of input and output, but also the map and library through which transformation occurs. Expansion and reduction operate through a lexical-semantic atlas.

**Core framing:** Let `A` = input (word, phrase, sentence, etc.), `Lambda` = the language map / lexical-semantic atlas, `E_Lambda` = expansion through the atlas, `R_Lambda` = reduction through the atlas. The working intuition is that expansion and reduction are directional counterparts, with local recovery:

`R_Lambda(E_Lambda(A)) ≈ A`

The project does not assume this holds in general; it aims to determine whether LRC is a valid architecture, how it behaves under composition and recursion, and under what conditions recovery or stability holds.

**Working hypotheses** (abbreviated; full list in [docs/lrc_working_hypotheses.md](docs/lrc_working_hypotheses.md)):

1. **LRC is a valid language-mediated architecture** — behavior governed by structured operator composition over a lexical-semantic atlas.
2. **Expansion and reduction are directional counterparts** — inverse-like, at least locally for some inputs.
3. **The map is foundational** — definitions, relations, grammar, and lexical neighborhoods form the map that determines what E and R can do.
4. **LRC admits layered composition** — E/R may be composed (e.g. E→R, E→E→R, recursive R until irreducible).
5. **LRC may exhibit dynamical regimes** — fixed-point convergence, oscillation, semantic drift, or collapse to canonical roots under recursion.
6. **Parallel or conjunctive inputs may be reducible jointly** — combined meanings processed and convolved together.
7. **LRC may apply to code** — same operator logic for expand/reduce on programming code.
8. **Reduction may use multiple strategies** — whole-span, recursive, local phrase matching, grammar-sensitive, sliding-window.
9. **Sliding-window reduction may be grammar-sensitive** — left-to-right window with definition matching.
10. **Overlapping windows may matter** — compositional meaning may depend on partial reuse of context.
11. **Recursive reduction may terminate in an irreducible state** — canonical semantic root or limit of atlas and strategy.

See [docs/lrc_working_hypotheses.md](docs/lrc_working_hypotheses.md) for open questions and experimental direction.

## Core Mechanisms

### Lexical Expansion (Semantic Unpacking)

- **Process:** Expand a source word, phrase, or sentence into semantic components.
- **Mechanism:** Use dictionary and optional LLM context to produce a structured cloud.
- **Output:** Definitions, connotations, related concepts, constraints, and negative constraints.

### Lexical Convolution (Semantic Reduction)

- **Process:** Compress semantic cloud back into lexical candidates.
- **Mechanism:** Score candidates by definition similarity, lexical relations, keyword overlap, POS compatibility, concision, and ambiguity penalties (ambiguity from WordNet lemma+POS sense counts).
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

Run once (sentence cycle primary output):

```powershell
python -m lrc_poc.cli run-once --text "The cat jumped over the dog." --mode 4 --definition-style literal_first
```

Run recursion (advanced lexical diagnostics):

```powershell
python -m lrc_poc.cli recurse --text "a positive emotional state with pleasure and contentment" --iterations 10 --mode 0
```

Run attractor map (advanced research diagnostics):

```powershell
python -m lrc_poc.cli map --seed-count 200 --iterations 10 --mode 0 --out artifacts
```

Word and short-phrase inputs are also supported, but sentence inputs are first-class and covered by tests.

## Dashboard Modes

The Streamlit app is split into milestone tabs:

- `Milestone 1 - Expand/Reduce`: core sentence operator (definition expansion + section-wise reduction).
- `Milestone 2 - Sentence Cycles`: repeated sentence-level expansion/reduction.
- `Milestone 3 - Attractor Analysis`: advanced lexical attractor research diagnostics.

## Control Meanings

- `Definition Style`:
  - `literal_first`: first concise literal definition clause.
  - `semantic_relational`: relation-focused meaning phrase (LLM-assisted when available).
  - `literal_raw`: full raw WordNet definition text.
- `Semantic fallback reduction`:
  - `off`: strict token-preserving reduction when exact definition match is not found.
  - `on`: allow semantic replacement for unmatched definition slices.
- `Lexicon Max Entries` (M3 advanced): upper bound on candidate lexicon size for runtime control.
- `Limit Per POS` (M3 advanced): caps entries per part-of-speech class.
- `Seed Count` and `Iterations` (M3 advanced): size and depth of attractor mapping experiments.
- `Use Domain Heuristics` (`--use-domain-heuristics` in CLI): opt-in bootstrap priors for known emotion clusters. Default is `off` to keep deterministic reduction domain-agnostic.

## Dashboard

```powershell
streamlit run src/lrc_poc/dashboard/app.py
```

The dashboard provides:

- Sentence definition expansion/reduction cycle as primary output.
- Definition style controls: `literal_first`, `semantic_relational`, `literal_raw`.
- Optional semantic fallback reduction toggle.
- Optional global compression debug view (disabled by default for sentence inputs).
- Milestone 2 recursive sentence-cycle trace.
- Milestone 3 attractor basin mapping with artifact downloads.

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
