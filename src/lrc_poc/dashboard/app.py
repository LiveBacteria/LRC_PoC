"""Streamlit dashboard for Milestone 3 attractor exploration."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from lrc_poc.attractor import map_attractors
from lrc_poc.lexicon import build_wordnet_lexicon
from lrc_poc.pipeline import LRCPipeline
from lrc_poc.recurse import lexical_entropy_profile, run_pipeline_recursion, summarize_iterations
from lrc_poc.sentence_ops import run_sentence_definition_cycle


@st.cache_resource
def prepare_lexicon(max_entries: int = 80000, limit_per_pos: int = 0):
    limit = limit_per_pos if limit_per_pos > 0 else None
    max_items = max_entries if max_entries > 0 else None
    return build_wordnet_lexicon(limit_per_pos=limit, max_entries=max_items)


@st.cache_resource
def prepare_pipeline(max_entries: int = 80000, limit_per_pos: int = 0) -> LRCPipeline:
    lexicon = prepare_lexicon(max_entries=max_entries, limit_per_pos=limit_per_pos)
    return LRCPipeline(lexicon=lexicon)


def _render_top_candidates(result) -> None:
    table = pd.DataFrame(
        [
            {
                "rank": index + 1,
                "word": item.word,
                "score": item.score_breakdown.total,
                "definition": item.definition,
            }
            for index, item in enumerate(result.top_k)
        ]
    )
    st.subheader("Top Candidates")
    st.dataframe(table, width="stretch")


def _render_sentence_cycle(cycle: dict[str, object]) -> None:
    st.subheader("Definition Expansion/Reduction Cycle")
    st.write("Expanded Sentence:")
    st.code(str(cycle["expanded_sentence"]))
    st.write("Reduced Terms Sentence:")
    st.code(str(cycle["reduced_sentence"]))
    mapping_df = pd.DataFrame(cycle["mappings"])
    st.dataframe(mapping_df, width="stretch")


def _render_recursion(iterations) -> None:
    rows = [
        {
            "iteration": item.iteration,
            "input_text": item.input_text,
            "winner_word": item.winner_word,
            "winner_score": item.winner_score,
            "confidence": item.confidence,
            "drift_from_origin": item.drift_from_origin,
            "entropy": item.entropy,
            "fixed_point": item.fixed_point,
            "cycle_detected": item.cycle_detected,
            "cycle_length": item.cycle_length,
        }
        for item in iterations
    ]
    frame = pd.DataFrame(rows)
    st.subheader("Recursion Trajectory")
    st.dataframe(frame, width="stretch")
    st.line_chart(frame.set_index("iteration")[["winner_score", "drift_from_origin", "entropy"]])


def _render_attractor_result(result) -> None:
    st.subheader("Attractor Summary")
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
    st.subheader("Attractor Basins")
    st.dataframe(basin_df, width="stretch")
    if not basin_df.empty:
        st.bar_chart(basin_df.set_index("final_word")["count"])

    st.subheader("Artifact Downloads")
    for file_path in result.output_files:
        path = Path(file_path)
        if not path.exists():
            continue
        st.download_button(
            label=f"Download {path.name}",
            data=path.read_bytes(),
            file_name=path.name,
            mime="application/octet-stream",
            key=f"download-{path.name}",
        )


def main() -> None:
    st.set_page_config(page_title="LRC Milestone 3 Dashboard", layout="wide")
    st.title("LRC Milestone 3: Sentence Definition Engine")

    with st.sidebar:
        st.header("Run Controls")
        mode = st.selectbox("Mode", options=[0, 1, 2, 3, 4], index=0)
        top_k = st.slider("Top K", min_value=3, max_value=20, value=10)
        iterations = st.slider("Iterations", min_value=2, max_value=20, value=10)
        seed_count = st.slider("Seed Count", min_value=10, max_value=300, value=200)
        max_entries = st.number_input("Lexicon Max Entries", min_value=1000, max_value=250000, value=80000)
        limit_per_pos = st.number_input("Limit Per POS (0 = none)", min_value=0, max_value=10000, value=0)

    pipeline = prepare_pipeline(max_entries=int(max_entries), limit_per_pos=int(limit_per_pos))

    st.subheader("Single Reduction")
    single_text = st.text_input("Input Text", value="grief")
    show_global = st.checkbox("Show Global Compression (Debug)", value=False)
    if st.button("Run Single Reduction", width="stretch"):
        cycle = run_sentence_definition_cycle(
            text=single_text,
            lexicon=pipeline.lexicon,
            pipeline=pipeline,
            mode=mode,
        )
        st.write(f"Reduced Sentence: `{cycle['reduced_sentence']}`")
        _render_sentence_cycle(cycle)

        token_count = len([piece for piece in single_text.strip().split() if piece])
        sentence_input = token_count > 1
        if sentence_input and not show_global:
            st.info("Sentence mode: global single-token compression hidden. Enable debug to view it.")
        else:
            result = pipeline.run_once(single_text, mode=mode, top_k=top_k)
            st.write(f"Global Compression Winner: `{result.winner.word}`")
            st.write(f"Confidence: `{result.confidence:.4f}`")
            if result.fallback_reason:
                st.warning(result.fallback_reason)
            _render_top_candidates(result)

    st.subheader("Recursive Analysis")
    recurse_text = st.text_input("Recursion Seed", value="grief", key="recurse_seed")
    if st.button("Run Recursion", width="stretch"):
        recursion = run_pipeline_recursion(recurse_text, pipeline=pipeline, iterations=iterations, mode=mode)
        st.json(
            {
                "summary": summarize_iterations(recursion),
                "candidate_uncertainty_profile": lexical_entropy_profile(recursion),
            }
        )
        _render_recursion(recursion)

    st.subheader("Attractor Mapping")
    output_dir = st.text_input("Output Directory", value="artifacts/dashboard")
    if st.button("Run Attractor Map", width="stretch"):
        result = map_attractors(
            lexicon=pipeline.lexicon,
            seed_count=seed_count,
            iterations=iterations,
            output_dir=output_dir,
            mode_used=mode,
            pipeline=pipeline,
        )
        _render_attractor_result(result)


if __name__ == "__main__":
    main()
