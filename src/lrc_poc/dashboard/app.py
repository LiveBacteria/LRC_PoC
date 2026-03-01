"""Streamlit dashboard for LRC sentence expansion and reduction workflows."""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import streamlit as st

from lrc_poc.attractor import map_attractors
from lrc_poc.lexicon import build_wordnet_lexicon
from lrc_poc.pipeline import LRCPipeline
from lrc_poc.recurse import lexical_entropy_profile, run_pipeline_recursion, summarize_iterations
from lrc_poc.sentence_ops import run_sentence_definition_cycle

MODE_OPTIONS: tuple[tuple[str, int, str], ...] = (
    (
        "Mode 0 - Deterministic baseline",
        0,
        "No LLM calls. WordNet + deterministic scoring only.",
    ),
    (
        "Mode 1 - LLM expansion",
        1,
        "LLM enriches expansion. Reduction remains deterministic.",
    ),
    (
        "Mode 2 - LLM candidate proposal",
        2,
        "Deterministic expansion, LLM proposes candidates, deterministic rerank.",
    ),
    (
        "Mode 3 - LLM rerank/judge",
        3,
        "Deterministic expansion + candidates, LLM reranks top candidates.",
    ),
    (
        "Mode 4 - Full LLM loop",
        4,
        "LLM expansion + proposal + rerank, with deterministic fallback on failure.",
    ),
)

DEFINITION_STYLE_OPTIONS: tuple[str, ...] = (
    "literal_first",
    "semantic_relational",
    "literal_raw",
)
DEFINITION_STYLE_LABELS: dict[str, str] = {
    "literal_first": "literal_first - short first-sense definition",
    "semantic_relational": "semantic_relational - relation phrase (e.g., kind of X)",
    "literal_raw": "literal_raw - full first-sense definition text",
}
DEFINITION_STYLE_HELP: dict[str, str] = {
    "literal_first": "Shortened literal definition. Strips parenthetical/secondary clauses.",
    "semantic_relational": "Compact relation statement, with optional LLM assistance in modes 1-4.",
    "literal_raw": "Unclipped first-sense definition text (longest output).",
}


@st.cache_resource
def prepare_lexicon(max_entries: int = 80000, limit_per_pos: int = 0):
    limit = limit_per_pos if limit_per_pos > 0 else None
    max_items = max_entries if max_entries > 0 else None
    return build_wordnet_lexicon(limit_per_pos=limit, max_entries=max_items)


@st.cache_resource
def prepare_pipeline(max_entries: int = 80000, limit_per_pos: int = 0) -> LRCPipeline:
    lexicon = prepare_lexicon(max_entries=max_entries, limit_per_pos=limit_per_pos)
    return LRCPipeline(lexicon=lexicon)


def _run_sentence_cycle_compat(
    *,
    text: str,
    lexicon,
    pipeline: LRCPipeline,
    mode: int,
    definition_style: str,
    use_semantic_fallback: bool,
) -> dict[str, object]:
    """Call sentence-cycle operator while tolerating older function signatures."""
    signature = inspect.signature(run_sentence_definition_cycle)
    kwargs = {
        "text": text,
        "lexicon": lexicon,
        "pipeline": pipeline,
        "mode": mode,
        "definition_style": definition_style,
        "use_semantic_fallback": use_semantic_fallback,
    }
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return run_sentence_definition_cycle(**supported)


def _render_sentence_cycle(cycle: dict[str, object]) -> None:
    st.markdown("**Expansion**")
    st.code(str(cycle["expanded_sentence"]))
    st.markdown("**Reduction**")
    st.code(str(cycle["reduced_sentence"]))

    mapping_df = pd.DataFrame(cycle["mappings"])
    st.markdown("**Token-Level Mapping**")
    st.dataframe(mapping_df, width="stretch")


def _render_top_candidates(result) -> None:
    table = pd.DataFrame(
        [
            {
                "rank": index + 1,
                "word_or_phrase": item.word,
                "score": item.score_breakdown.total,
                "definition": item.definition,
            }
            for index, item in enumerate(result.top_k)
        ]
    )
    st.markdown("**Global Compression Candidates (Debug)**")
    st.dataframe(table, width="stretch")


def _render_mode_picker(*, key: str, default_mode: int = 0) -> int:
    labels = [item[0] for item in MODE_OPTIONS]
    label_to_mode = {item[0]: item[1] for item in MODE_OPTIONS}
    label_to_help = {item[0]: item[2] for item in MODE_OPTIONS}

    default_index = 0
    for index, item in enumerate(MODE_OPTIONS):
        if item[1] == default_mode:
            default_index = index
            break

    selected_label = st.selectbox(
        "Execution Mode",
        options=labels,
        index=default_index,
        key=key,
        help="Controls where LLMs are used in the pipeline.",
    )
    st.caption(label_to_help[selected_label])
    return label_to_mode[selected_label]


def _render_definition_style_picker(*, key: str, default: str = "literal_first") -> str:
    default_index = DEFINITION_STYLE_OPTIONS.index(default)
    selected = st.selectbox(
        "Definition Style",
        options=list(DEFINITION_STYLE_OPTIONS),
        index=default_index,
        key=key,
        format_func=lambda item: DEFINITION_STYLE_LABELS[item],
        help="Controls how each word is expanded before reduction.",
    )
    st.caption(DEFINITION_STYLE_HELP[selected])
    return selected


def _render_m1(pipeline: LRCPipeline) -> None:
    st.subheader("Milestone 1: Definition Expansion and Reduction")
    st.caption(
        "String in -> expansion string -> reduction string. "
        "Each lexical token is expanded to a definition string, then reduced back to a word/phrase."
    )
    with st.expander("What these controls do", expanded=False):
        st.markdown(
            "- `Execution Mode`: selects deterministic-only vs LLM-assisted processing.\n"
            "- `Definition Style`: selects short literal, relation phrase, or full literal definition.\n"
            "- `Semantic fallback reduction`: if enabled, unmatched definitions are reduced semantically (can change words).\n"
            "- `Show global compression debug output`: optional whole-sentence compressor diagnostics."
        )

    text = st.text_area(
        "Input sentence or phrase",
        value="The cat jumped over the dog.",
        height=100,
        help="Any sentence or phrase. The operator expands each word in this string.",
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        mode = _render_mode_picker(key="m1_mode", default_mode=0)
    with col_b:
        definition_style = _render_definition_style_picker(key="m1_definition_style", default="literal_first")
    with col_c:
        semantic_fallback = st.checkbox(
            "Semantic fallback reduction",
            value=False,
            key="m1_semantic_fallback",
            help="Off = preserve original token when no exact definition match is found. "
            "On = semantic reduction may replace it with a different word/phrase.",
        )

    show_global = st.checkbox(
        "Show global compression debug output",
        value=False,
        key="m1_show_global",
        help="Shows the independent whole-sentence reducer for inspection.",
    )

    if st.button("Run Expansion -> Reduction", width="stretch", key="m1_run"):
        cycle = _run_sentence_cycle_compat(
            text=text,
            lexicon=pipeline.lexicon,
            pipeline=pipeline,
            mode=mode,
            definition_style=definition_style,
            use_semantic_fallback=semantic_fallback,
        )
        st.markdown("**Run Configuration**")
        st.json(
            {
                "mode": mode,
                "definition_style": definition_style,
                "semantic_fallback_reduction": semantic_fallback,
            }
        )
        _render_sentence_cycle(cycle)

        if show_global:
            result = pipeline.run_once(text, mode=mode, top_k=10)
            st.write(f"Global Compression Winner (Debug): `{result.winner.word}`")
            st.write(f"Confidence: `{result.confidence:.4f}`")
            if result.fallback_reason:
                st.warning(result.fallback_reason)
            _render_top_candidates(result)


def _render_m2(pipeline: LRCPipeline) -> None:
    st.subheader("Milestone 2: Recursive Sentence Cycles")
    st.caption(
        "Apply sentence expansion/reduction repeatedly. "
        "This shows string transformations across cycles."
    )
    with st.expander("What these controls do", expanded=False):
        st.markdown(
            "- `Execution Mode`: deterministic vs LLM-assisted behavior.\n"
            "- `Definition Style`: expansion format for each cycle.\n"
            "- `Cycle Count`: number of repeated expand/reduce cycles.\n"
            "- `Semantic fallback reduction`: allow semantic replacement when exact match is missing."
        )

    text = st.text_area(
        "Seed sentence",
        value="A feeling of loss tied specifically to death and emotional absence.",
        height=100,
        key="m2_text",
        help="Initial sentence used for recursive expand/reduce cycles.",
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        mode = _render_mode_picker(key="m2_mode", default_mode=0)
    with col_b:
        definition_style = _render_definition_style_picker(key="m2_definition_style", default="literal_first")
    with col_c:
        iterations = st.slider(
            "Cycle Count",
            min_value=2,
            max_value=20,
            value=5,
            key="m2_iterations",
            help="How many expand/reduce cycles to run in sequence.",
        )
    with col_d:
        semantic_fallback = st.checkbox(
            "Semantic fallback reduction",
            value=False,
            key="m2_semantic_fallback",
            help="Allows non-exact semantic substitution during reduction.",
        )

    if st.button("Run Recursive Sentence Cycles", width="stretch", key="m2_run"):
        current = text
        records: list[dict[str, object]] = []
        for step in range(1, iterations + 1):
            cycle = _run_sentence_cycle_compat(
                text=current,
                lexicon=pipeline.lexicon,
                pipeline=pipeline,
                mode=mode,
                definition_style=definition_style,
                use_semantic_fallback=semantic_fallback,
            )
            records.append(
                {
                    "iteration": step,
                    "input": current,
                    "expanded": str(cycle["expanded_sentence"]),
                    "reduced": str(cycle["reduced_sentence"]),
                }
            )
            current = str(cycle["reduced_sentence"])

        st.markdown("**Sentence Cycle Trace**")
        frame = pd.DataFrame(records)
        st.dataframe(frame, width="stretch")

        # Optional lexical diagnostics from existing recursion module.
        lexical_recursion = run_pipeline_recursion(
            source_text=text,
            pipeline=pipeline,
            iterations=min(iterations, 10),
            mode=mode,
        )
        st.markdown("**Lexical Diagnostics (Reference)**")
        st.json(
            {
                "summary": summarize_iterations(lexical_recursion),
                "candidate_uncertainty_profile": lexical_entropy_profile(lexical_recursion),
            }
        )


def _render_m3() -> None:
    st.subheader("Milestone 3: Attractor Mapping (Advanced)")
    st.caption(
        "Research diagnostics over many seed words. "
        "This is optional and separate from core sentence expansion/reduction."
    )
    with st.expander("What these controls do", expanded=False):
        st.markdown(
            "- `Execution Mode`: deterministic vs LLM-assisted reducer behavior.\n"
            "- `Lexicon Max Entries`: cap on total candidate lexicon size (performance control).\n"
            "- `Limit Per POS`: cap candidates per part-of-speech class (0 means no cap).\n"
            "- `Seed Count`: number of starting words to map.\n"
            "- `Iterations`: recursion depth per seed."
        )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        mode = _render_mode_picker(key="m3_mode", default_mode=0)
    with col_b:
        max_entries = st.number_input(
            "Lexicon Max Entries",
            min_value=1000,
            max_value=250000,
            value=80000,
            help="Upper bound on lexicon entries used in attractor mapping.",
        )
    with col_c:
        limit_per_pos = st.number_input(
            "Limit Per POS (0 = none)",
            min_value=0,
            max_value=10000,
            value=0,
            help="Caps entries per part-of-speech class. 0 disables this cap.",
        )

    col_d, col_e, col_f = st.columns(3)
    with col_d:
        seed_count = st.slider(
            "Seed Count",
            min_value=10,
            max_value=300,
            value=200,
            help="Number of seed words for mapping.",
        )
    with col_e:
        iterations = st.slider(
            "Iterations",
            min_value=2,
            max_value=20,
            value=10,
            help="Recursion steps per seed.",
        )
    with col_f:
        output_dir = st.text_input(
            "Output Directory",
            value="artifacts/dashboard",
            help="Where CSV/JSON artifacts are written.",
        )

    if st.button("Run Attractor Map", width="stretch", key="m3_run"):
        pipeline = prepare_pipeline(max_entries=int(max_entries), limit_per_pos=int(limit_per_pos))
        result = map_attractors(
            lexicon=pipeline.lexicon,
            seed_count=seed_count,
            iterations=iterations,
            output_dir=output_dir,
            mode_used=mode,
            pipeline=pipeline,
        )

        st.json(
            {
                "seed_count": result.seed_count,
                "iterations": result.iterations,
                "fixed_points": result.fixed_points,
                "cycles": result.cycles,
                "average_drift": result.average_drift,
                "average_entropy": result.average_entropy,
            }
        )

        basin_df = pd.DataFrame(
            [{"final_word": word, "count": count} for word, count in result.basin_summary.items()]
        ).sort_values("count", ascending=False)
        st.dataframe(basin_df, width="stretch")
        if not basin_df.empty:
            st.bar_chart(basin_df.set_index("final_word")["count"])

        for file_path in result.output_files:
            path = Path(file_path)
            if not path.exists():
                continue
            st.download_button(
                label=f"Download {path.name}",
                data=path.read_bytes(),
                file_name=path.name,
                mime="application/octet-stream",
                key=f"m3-download-{path.name}",
            )


def main() -> None:
    st.set_page_config(page_title="LRC Engine", layout="wide")
    st.title("LRC Engine")

    pipeline = prepare_pipeline(max_entries=80000, limit_per_pos=0)

    tab_m1, tab_m2, tab_m3 = st.tabs(
        [
            "Milestone 1 - Expand/Reduce",
            "Milestone 2 - Sentence Cycles",
            "Milestone 3 - Attractor Analysis",
        ]
    )

    with tab_m1:
        _render_m1(pipeline)

    with tab_m2:
        _render_m2(pipeline)

    with tab_m3:
        _render_m3()


if __name__ == "__main__":
    main()
