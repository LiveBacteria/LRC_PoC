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
- **Mechanism:** Use deterministic lexical resources (WordNet/dictionary information) to produce a structured cloud.
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

## Deterministic Runtime

- The active runtime is deterministic-only.
- Expansion uses WordNet-backed lexical serialization.
- Reduction uses deterministic scoring and lexical matching.
- No LLM mode switching is used in milestone workflows.

## Expansion Policy

- Expansion is deterministic by design in sentence workflows.
- Expansion should be interpreted as replacing lexical tokens with dictionary/WordNet-grounded definition information.
- LLM expansion was intentionally removed because it introduced unnecessary variance and latency, and it is not required for the core LRC expansion objective.

## M1/M2 Unified Execution Model

M1 and M2 now use the same primitives and only the same primitives:

1. `expand`: token -> context-aware WordNet cloud/definition serialization.
2. `reduce`: greedy left-to-right window reduction using cloud superposition and lexical candidate matching.
3. `recursive_reduce`: repeated `reduce` passes until a pass makes zero replacements.

Formal sketch:

- Input units: `X = (u_1, ..., u_n)`.
- Expansion cloud per unit: `C_i = E(u_i | context)`.
- Expand-on-expand: `C_i^(t+1) = normalize(C_i^t (+) E(top_terms(C_i^t)))`.
- Reduce span cloud: `S_(i:j) = (+)_{k=i..j} C_k`.
- Candidate winner: `w* = argmax_w sim(S_(i:j), L_w)`.
- Accept only if:
  - `score >= tau_score`
  - `margin >= tau_margin`
  - winner is a single lexical word.
- Replacement contract: only `span >= 2 -> 1`.

Recursive stop rule:

- `recursive_reduce` stops on the first pass with zero accepted replacements.
- Because accepted replacements always reduce token count, termination is guaranteed.

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
python -m lrc_poc.cli run-once --text "The cat jumped over the dog." --definition-style literal_first
```

Emit reducer diagnostics for one run:

```powershell
python -m lrc_poc.cli run-once --text "The cat jumped over the dog." --emit-reduction-diagnostics
```

Run recursion (advanced lexical diagnostics):

```powershell
python -m lrc_poc.cli recurse --text "a positive emotional state with pleasure and contentment" --iterations 10
```

Run attractor map (advanced research diagnostics):

```powershell
python -m lrc_poc.cli map --seed-count 200 --iterations 10 --out artifacts
```

Replay a thread JSON and produce an incident report for stalled reductions:

```powershell
python -m lrc_poc.cli investigate-reduction --thread-json "C:/Users/LiveB/Downloads/thread-1 (2).json"
```

Word and short-phrase inputs are also supported, but sentence inputs are first-class and covered by tests.

## Dashboard Milestones

The Streamlit app uses a chat-style workspace with sidebar milestone selection:

- `Milestone 1 - Expand/Reduce`: one sentence cycle with token and segment details.
- `Milestone 2 - Sequence Cycles`: repeated sentence-level operation sequences with declarable order (`expand` / `reduce` / `recursive_reduce`).
- `Milestone 3 - Attractor Analysis`: advanced lexical attractor research diagnostics over seed sets.
- All milestones execute on the same deterministic runtime path.

## Control Meanings

- `Definition Style`:
  - `literal_first`: first concise literal definition clause.
  - `cloud_compact`: compact cloud serialization (definition + lexical relations).
  - `literal_raw`: full raw WordNet definition text.
- `Sequence` (M2):
  - Predefined or custom operation order for each cycle.
  - Supports `expand`, `reduce`, and `recursive_reduce` operators.
  - Custom text accepts comma/space/`->` separators.
  - Built-ins include: `expand -> reduce`, `expand -> expand -> reduce`, `expand -> reduce -> reduce`, `expand -> expand -> reduce -> reduce`, and `expand -> recursive_reduce`.
  - `reduce` scans windows left-to-right, tries longest spans first, superposes span clouds, and replaces accepted spans with the top lexical match.
  - `recursive_reduce` is an explicit LRC function that repeatedly applies reduction until a pass yields zero accepted replacements.
  - Examples:
    - `expand -> recursive_reduce`
    - `recursive_reduce`
    - `expand -> reduce -> reduce` (manual recursive reduction)
- `Lexicon Max Entries` (M3 advanced): upper bound on candidate lexicon size for runtime control.
- `Limit Per POS` (M3 advanced): caps entries per part-of-speech class.
- `Seed Count` and `Iterations` (M3 advanced): size and depth of attractor mapping experiments.
- `Use Domain Heuristics` (`--use-domain-heuristics` in CLI): opt-in bootstrap priors for known emotion clusters. Default is `off` to keep deterministic reduction domain-agnostic.

## Dashboard

```powershell
streamlit run src/lrc_poc/dashboard/app.py
```

The dashboard provides:

- Chat transcript output with card/expander rendering instead of dataframe tables.
- Multi-thread chat history: create, rename, delete, export, and swap between investigation threads.
- Sidebar controls for definition style and M2 sequence selection.
- Milestone 2 cycle-by-cycle and step-by-step operation trace for user-defined sequences.
- Milestone 3 attractor mapping summaries and artifact downloads.

## Investigation Note: Expansion Tail Bug

- Symptom: after `expand -> reduce`, reduced output could include trailing definition fragments.
- Root cause: reduction reconstruction was using expanded text as the reconstruction source while mappings were indexed to the pre-expansion sentence.
- Resolution: reconstruction now uses the source sentence tied to the mapping set, eliminating post-reduction definition tails.

## Testing

Run full suite:

```powershell
pytest
```

Coverage includes:

- Deterministic logic/unit tests.
- Gold-case semantic accuracy regression tests.
- Recursion/cycle/fixed-point validation tests.
- CLI end-to-end tests.
- Dashboard smoke tests.

## Governance and ADRs

ADRs live in:

- [docs/adr/README.md](docs/adr/README.md)

First ADR:

- [docs/adr/0001-change-documentation-and-full-update-rule.md](docs/adr/0001-change-documentation-and-full-update-rule.md)

That ADR enforces documentation and relevant-file updates for all behavioral changes.
