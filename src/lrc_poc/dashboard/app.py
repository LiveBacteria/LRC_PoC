"""Chat-style Streamlit dashboard for LRC workflows."""

from __future__ import annotations

import html
import json
from pathlib import Path
import re
from typing import Any
from datetime import datetime

import streamlit as st

from lrc_poc.attractor import map_attractors
from lrc_poc.lexicon import build_wordnet_lexicon
from lrc_poc.pipeline import LRCPipeline
from lrc_poc.sentence_ops import (
    DEFAULT_RECURSIVE_REDUCTION_LIMIT,
    parse_operation_sequence,
    run_sentence_definition_cycle,
    run_sentence_operation_sequence,
)

DEFINITION_STYLE_OPTIONS: tuple[str, ...] = (
    "literal_first",
    "cloud_compact",
    "literal_raw",
)
DEFINITION_STYLE_LABELS: dict[str, str] = {
    "literal_first": "literal_first - short first-sense definition",
    "cloud_compact": "cloud_compact - compact lexical cloud serialization",
    "literal_raw": "literal_raw - full first-sense definition text",
}
DEFINITION_STYLE_HELP: dict[str, str] = {
    "literal_first": "Shortened literal definition. Strips parenthetical/secondary clauses.",
    "cloud_compact": "Compact deterministic cloud serialization (definition + lexical relations).",
    "literal_raw": "Unclipped first-sense definition text (longest output).",
}
M2_SEQUENCE_PRESETS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "Classic: Expand -> Reduce",
        ("expand", "reduce"),
        "One expansion followed by one reduction.",
    ),
    (
        "Deep Expansion: Expand -> Expand -> Reduce",
        ("expand", "expand", "reduce"),
        "Two expansion passes before one reduction.",
    ),
    (
        "Recursive Reduce: Expand -> Reduce -> Reduce",
        ("expand", "reduce", "reduce"),
        "Expand once, then recursively reduce twice.",
    ),
    (
        "Wave: Expand -> Expand -> Reduce -> Reduce",
        ("expand", "expand", "reduce", "reduce"),
        "Two expansion passes followed by two reduction passes.",
    ),
    (
        "Recursive Function: Expand -> Recursive Reduce",
        ("expand", "recursive_reduce"),
        "Runs reduction repeatedly until no change, cycle detection, or max recursive steps.",
    ),
)
MILESTONE_OPTIONS: tuple[tuple[str, str, str], ...] = (
    (
        "Milestone 1 - Expand/Reduce",
        "m1",
        "Single sentence cycle with token-level details.",
    ),
    (
        "Milestone 2 - Sequence Cycles",
        "m2",
        "Repeated custom operation sequences (expand/reduce).",
    ),
    (
        "Milestone 3 - Attractor Analysis",
        "m3",
        "Batch lexical recursion diagnostics over seed words.",
    ),
)


@st.cache_resource
def prepare_lexicon(max_entries: int = 6000, limit_per_pos: int = 0):
    limit = limit_per_pos if limit_per_pos > 0 else None
    max_items = max_entries if max_entries > 0 else None
    return build_wordnet_lexicon(limit_per_pos=limit, max_entries=max_items)


@st.cache_resource
def prepare_pipeline(max_entries: int = 6000, limit_per_pos: int = 0) -> LRCPipeline:
    lexicon = prepare_lexicon(max_entries=max_entries, limit_per_pos=limit_per_pos)
    return LRCPipeline(lexicon=lexicon)


def _now_stamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _ensure_thread_state() -> None:
    if "lrc_threads" in st.session_state and "lrc_active_thread" in st.session_state:
        return

    created = _now_stamp()
    st.session_state["lrc_thread_counter"] = 1
    st.session_state["lrc_threads"] = {
        "thread-1": {
            "id": "thread-1",
            "title": "Thread 1",
            "created_at": created,
            "updated_at": created,
            "milestone_hint": "m2",
            "messages": [],
        }
    }
    st.session_state["lrc_thread_order"] = ["thread-1"]
    st.session_state["lrc_active_thread"] = "thread-1"


def _create_thread(*, milestone_hint: str) -> None:
    counter = int(st.session_state.get("lrc_thread_counter", 1)) + 1
    thread_id = f"thread-{counter}"
    created = _now_stamp()
    st.session_state["lrc_thread_counter"] = counter
    st.session_state["lrc_threads"][thread_id] = {
        "id": thread_id,
        "title": f"Thread {counter}",
        "created_at": created,
        "updated_at": created,
        "milestone_hint": milestone_hint,
        "messages": [],
    }
    st.session_state["lrc_thread_order"].append(thread_id)
    st.session_state["lrc_active_thread"] = thread_id


def _delete_active_thread() -> None:
    order = list(st.session_state["lrc_thread_order"])
    if len(order) <= 1:
        return
    active = str(st.session_state["lrc_active_thread"])
    if active in st.session_state["lrc_threads"]:
        st.session_state["lrc_threads"].pop(active, None)
    order = [item for item in order if item != active]
    st.session_state["lrc_thread_order"] = order
    st.session_state["lrc_active_thread"] = order[-1]


def _thread_label(thread: dict[str, Any]) -> str:
    title = str(thread.get("title", "Untitled"))
    count = len(thread.get("messages", []))
    milestone_hint = str(thread.get("milestone_hint", "m2")).upper()
    return f"{title} [{milestone_hint}] ({count})"


def _settings_snapshot(settings: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "milestone_key",
        "definition_style",
        "cycle_count",
        "operation_sequence",
        "recursive_reduce_max_steps",
        "max_entries",
        "limit_per_pos",
        "seed_count",
        "iterations",
        "output_dir",
    )
    return {key: settings.get(key) for key in keys}


def _merge_reduction_diagnostics(items: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if isinstance(value, dict):
                target = merged.setdefault(key, {})
                if not isinstance(target, dict):
                    continue
                for span_len, span_count in value.items():
                    target[str(span_len)] = int(target.get(str(span_len), 0)) + int(span_count)
            elif isinstance(value, (int, float)):
                merged[key] = int(merged.get(key, 0)) + int(value)
    return merged


def _inject_styles() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
:root {
  --lrc-bg-1: #0f1b1d;
  --lrc-bg-2: #1a1110;
  --lrc-card: #19272a;
  --lrc-ink: #f2f7f5;
  --lrc-muted: #d2e4df;
  --lrc-accent: #2dd4bf;
  --lrc-accent-2: #f59e0b;
  --lrc-border: #335158;
  --lrc-sidebar-1: #11272a;
  --lrc-sidebar-2: #17343a;
  --lrc-input-bg: #0f2023;
  --lrc-input-border: #3f6970;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
  font-family: "Space Grotesk", sans-serif;
}
code, pre {
  font-family: "IBM Plex Mono", monospace !important;
}
.material-icons,
.material-icons-round,
.material-icons-outlined,
.material-symbols-rounded,
.material-symbols-outlined,
.material-symbols-sharp {
  font-family: "Material Symbols Rounded", "Material Icons" !important;
  font-weight: normal !important;
  font-style: normal !important;
  letter-spacing: normal !important;
  text-transform: none !important;
  line-height: 1 !important;
}
[data-testid="stAppViewContainer"], .stApp {
  background:
    radial-gradient(circle at 12% 18%, rgba(45, 212, 191, 0.20), transparent 42%),
    radial-gradient(circle at 88% 6%, rgba(245, 158, 11, 0.16), transparent 36%),
    linear-gradient(145deg, var(--lrc-bg-1), var(--lrc-bg-2)) !important;
  color: var(--lrc-ink) !important;
}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
  background: linear-gradient(180deg, var(--lrc-sidebar-1), var(--lrc-sidebar-2)) !important;
  border-right: 1px solid #3d6268 !important;
  color: var(--lrc-ink) !important;
}
[data-testid="stSidebar"] * {
  color: var(--lrc-ink) !important;
}
h1, h2, h3, h4, .stMarkdown p, .stCaption {
  color: var(--lrc-ink) !important;
}
h1 {
  letter-spacing: 0.01em;
  text-shadow: 0 1px 0 rgba(0, 0, 0, 0.35);
}
[data-testid="stChatMessage"] {
  background: rgba(20, 33, 36, 0.72) !important;
  border: 1px solid var(--lrc-border) !important;
  border-radius: 14px !important;
  padding: 0.55rem 0.65rem !important;
}
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
.stTextInput input,
.stTextArea textarea {
  background: var(--lrc-input-bg) !important;
  border: 1px solid var(--lrc-input-border) !important;
  color: #f6fffb !important;
}
[data-baseweb="select"] span, [data-baseweb="select"] div {
  color: #f6fffb !important;
}
div[role="listbox"] {
  background: #102428 !important;
  border: 1px solid #4e767d !important;
  color: #f4fffb !important;
}
div[role="option"] {
  color: #f4fffb !important;
}
div[role="option"][aria-selected="true"] {
  background: #1f4f55 !important;
}
.stSlider [data-baseweb="slider"] > div > div {
  background: #4fe0cf !important;
}
.stSlider [role="slider"] {
  border: 2px solid #123f45 !important;
  background: #c8fff3 !important;
}
.lrc-card {
  background: var(--lrc-card);
  border: 1px solid var(--lrc-border);
  border-left: 4px solid var(--lrc-accent);
  border-radius: 14px;
  padding: 0.9rem 1rem;
  margin: 0.4rem 0 0.7rem;
  box-shadow: 0 5px 16px rgba(13, 42, 38, 0.06);
  animation: lrc-reveal 260ms ease-out;
}
.lrc-card-title {
  font-weight: 700;
  font-size: 0.92rem;
  letter-spacing: 0.01em;
  color: var(--lrc-ink);
}
.lrc-muted {
  color: var(--lrc-muted);
  font-size: 0.9rem;
}
.lrc-badge {
  display: inline-block;
  border: 1px solid #f9cd62;
  background: #4a3307;
  color: #ffd978;
  border-radius: 999px;
  padding: 0.18rem 0.55rem;
  font-size: 0.75rem;
  margin-right: 0.35rem;
}
.lrc-title-wrap {
  background: linear-gradient(90deg, rgba(28, 64, 69, 0.90), rgba(76, 54, 16, 0.85));
  border: 1px solid #3d6469;
  border-left: 4px solid var(--lrc-accent);
  border-radius: 14px;
  padding: 0.65rem 0.95rem;
  margin-bottom: 0.8rem;
}
.lrc-title {
  margin: 0;
  font-size: 1.6rem;
  color: #f5fffb;
  font-weight: 700;
}
@keyframes lrc-reveal {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}
@media (max-width: 900px) {
  .lrc-card { padding: 0.75rem 0.8rem; }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _render_definition_style_picker(*, key: str, default: str = "literal_first") -> str:
    default_index = DEFINITION_STYLE_OPTIONS.index(default)
    selected = st.selectbox(
        "Definition Style",
        options=list(DEFINITION_STYLE_OPTIONS),
        index=default_index,
        key=key,
        format_func=lambda item: DEFINITION_STYLE_LABELS[item],
    )
    st.caption(DEFINITION_STYLE_HELP[selected])
    return selected


def _format_operation_sequence(sequence: tuple[str, ...]) -> str:
    return " -> ".join(step.upper() for step in sequence)


def _render_card(title: str, body: str) -> None:
    st.markdown(
        f"""
<div class="lrc-card">
  <div class="lrc-card-title">{html.escape(title)}</div>
  <div class="lrc-muted">{html.escape(body)}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _parse_m3_seed_prompt(prompt_text: str) -> tuple[str, ...] | None:
    stripped = prompt_text.strip().lower()
    if stripped in {"", "auto", "sample"}:
        return None
    seeds = [token.strip().lower() for token in re.split(r"[,;\n]+", prompt_text) if token.strip()]
    return tuple(seeds) if seeds else None


def _render_sidebar_controls() -> dict[str, Any]:
    with st.sidebar:
        st.markdown("## LRC Settings")
        milestone_labels = [item[0] for item in MILESTONE_OPTIONS]
        label_to_meta = {item[0]: (item[1], item[2]) for item in MILESTONE_OPTIONS}

        selected_milestone_label = st.selectbox(
            "Workspace",
            options=milestone_labels,
            index=1,
        )
        milestone_key, milestone_help = label_to_meta[selected_milestone_label]
        st.caption(milestone_help)

        st.markdown("### Chat Threads")
        thread_order: list[str] = list(st.session_state["lrc_thread_order"])
        threads: dict[str, dict[str, Any]] = st.session_state["lrc_threads"]
        active_thread_id = str(st.session_state["lrc_active_thread"])
        if active_thread_id not in threads:
            active_thread_id = thread_order[0]
            st.session_state["lrc_active_thread"] = active_thread_id

        selected_thread_id = st.selectbox(
            "Active Thread",
            options=thread_order,
            index=thread_order.index(active_thread_id),
            format_func=lambda thread_id: _thread_label(threads[thread_id]),
            key="lrc_active_thread_select",
        )
        if selected_thread_id != active_thread_id:
            st.session_state["lrc_active_thread"] = selected_thread_id
            active_thread_id = selected_thread_id

        col_new, col_del = st.columns(2)
        with col_new:
            if st.button("New Thread", width="stretch"):
                _create_thread(milestone_hint=milestone_key)
                st.rerun()
        with col_del:
            if st.button("Delete Thread", width="stretch", disabled=len(thread_order) <= 1):
                _delete_active_thread()
                st.rerun()

        active_thread = threads[active_thread_id]
        title_key = f"lrc_thread_title_{active_thread_id}"
        title_value = st.text_input(
            "Thread Title",
            value=str(active_thread.get("title", "")),
            key=title_key,
        )
        if st.button("Save Thread Title", width="stretch"):
            cleaned = title_value.strip() or str(active_thread.get("title", "Untitled"))
            active_thread["title"] = cleaned
            active_thread["updated_at"] = _now_stamp()
            st.rerun()

        export_payload = {
            "id": active_thread["id"],
            "title": active_thread["title"],
            "created_at": active_thread["created_at"],
            "updated_at": active_thread["updated_at"],
            "milestone_hint": active_thread.get("milestone_hint", ""),
            "messages": active_thread.get("messages", []),
        }
        st.download_button(
            "Export Thread JSON",
            data=json.dumps(export_payload, indent=2),
            file_name=f"{active_thread['id']}.json",
            mime="application/json",
            key=f"lrc_export_thread_{active_thread_id}",
        )

        st.caption(
            f"Created: {active_thread.get('created_at', '')} | Updated: {active_thread.get('updated_at', '')}"
        )
        st.markdown("---")

        st.caption("Unified deterministic execution is used for all milestones.")
        definition_style = "literal_first"
        cycle_count = 1
        sequence: tuple[str, ...] | None = None
        sequence_error = ""
        recursive_reduce_max_steps = DEFAULT_RECURSIVE_REDUCTION_LIMIT
        max_entries = 6000
        limit_per_pos = 0
        seed_count = 200
        iterations = 10
        output_dir = "artifacts/dashboard"
        matrix_input_text = ""
        run_mode_style_matrix = False

        if milestone_key in {"m1", "m2"}:
            definition_style = _render_definition_style_picker(
                key=f"{milestone_key}_definition_style",
                default="literal_first",
            )
            st.caption(
                "Sentence workflows use deterministic WordNet expansion and windowed reduction."
            )
            st.markdown("### Batch Test")
            matrix_input_text = st.text_area(
                "Batch Input Text",
                value="",
                key=f"{milestone_key}_matrix_input_text",
                placeholder="Enter a sentence to test unified deterministic styles.",
            )
            run_mode_style_matrix = st.button(
                "Run Unified Style Sweep (No Raw)",
                width="stretch",
                key=f"{milestone_key}_run_mode_style_matrix",
                help=(
                    "Runs deterministic execution across literal_first and cloud_compact. "
                    "Posts consolidated results into chat."
                ),
            )

        if milestone_key == "m2":
            cycle_count = st.slider(
                "Cycle Count",
                min_value=1,
                max_value=20,
                value=1,
                help="How many times the entire sequence is repeated.",
            )
            sequence_source = st.radio(
                "Sequence Type",
                options=("Predefined", "Custom"),
                horizontal=True,
                key="m2_sequence_source",
            )
            if sequence_source == "Predefined":
                preset_labels = [item[0] for item in M2_SEQUENCE_PRESETS]
                preset_map = {item[0]: item for item in M2_SEQUENCE_PRESETS}
                selected_preset = st.selectbox(
                    "Predefined Sequence",
                    options=preset_labels,
                    key="m2_sequence_preset",
                )
                _, selected_sequence, selected_help = preset_map[selected_preset]
                sequence = selected_sequence
                st.caption(selected_help)
            else:
                custom_sequence_text = st.text_input(
                    "Custom Sequence",
                    value="expand -> recursive_reduce",
                    key="m2_sequence_custom",
                    help="Use expand/reduce/recursive_reduce separated by commas, spaces, or arrows.",
                )
                try:
                    sequence = parse_operation_sequence(custom_sequence_text)
                except ValueError as error:
                    sequence = None
                    sequence_error = str(error)
                    st.error(sequence_error)

            if sequence is not None:
                st.markdown(
                    f'<span class="lrc-badge">Resolved: {_format_operation_sequence(sequence)}</span>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Each REDUCE performs left-to-right greedy window matching over cloud superposition. "
                    "recursive_reduce repeats REDUCE passes until no replacements are accepted."
                )
            recursive_reduce_max_steps = st.slider(
                "Recursive Reduce Max Steps",
                min_value=2,
                max_value=50,
                value=DEFAULT_RECURSIVE_REDUCTION_LIMIT,
                help="Maximum internal reduce iterations when using recursive_reduce.",
            )
            st.markdown("**Recursive reduction examples:**")
            st.code("expand -> recursive_reduce")
            st.code("recursive_reduce")
            st.code("expand -> reduce -> reduce  (manual recursive reduction)")

        if milestone_key == "m3":
            max_entries = st.number_input(
                "Lexicon Max Entries",
                min_value=1000,
                max_value=250000,
                value=6000,
            )
            limit_per_pos = st.number_input(
                "Limit Per POS (0 = none)",
                min_value=0,
                max_value=10000,
                value=0,
            )
            seed_count = st.slider(
                "Seed Count",
                min_value=10,
                max_value=300,
                value=200,
            )
            iterations = st.slider(
                "Iterations",
                min_value=2,
                max_value=20,
                value=10,
            )
            output_dir = st.text_input(
                "Output Directory",
                value="artifacts/dashboard",
            )

        clear_chat = st.button("Clear Chat", width="stretch")

    return {
        "milestone_label": selected_milestone_label,
        "milestone_key": milestone_key,
        "active_thread_id": active_thread_id,
        "definition_style": definition_style,
        "cycle_count": cycle_count,
        "operation_sequence": sequence,
        "sequence_error": sequence_error,
        "recursive_reduce_max_steps": recursive_reduce_max_steps,
        "max_entries": int(max_entries),
        "limit_per_pos": int(limit_per_pos),
        "seed_count": int(seed_count),
        "iterations": int(iterations),
        "output_dir": output_dir,
        "matrix_input_text": matrix_input_text,
        "run_mode_style_matrix": run_mode_style_matrix,
        "clear_chat": clear_chat,
    }


def _run_m1(
    *,
    text: str,
    pipeline: LRCPipeline,
    definition_style: str,
) -> dict[str, Any]:
    cycle = run_sentence_definition_cycle(
        text=text,
        lexicon=pipeline.lexicon,
        pipeline=pipeline,
        mode=0,
        definition_style=definition_style,
        use_semantic_fallback=False,
    )

    return {
        "kind": "m1",
        "input_text": text,
        "definition_style": definition_style,
        "cycle": cycle,
    }


def _run_m2(
    *,
    text: str,
    pipeline: LRCPipeline,
    definition_style: str,
    cycle_count: int,
    operation_sequence: tuple[str, ...],
    recursive_reduce_max_steps: int,
) -> dict[str, Any]:
    current = text
    cycles: list[dict[str, Any]] = []
    cycle_diagnostics: list[dict[str, Any]] = []
    for cycle_index in range(1, cycle_count + 1):
        sequence_result = run_sentence_operation_sequence(
            text=current,
            lexicon=pipeline.lexicon,
            pipeline=pipeline,
            mode=0,
            definition_style=definition_style,
            use_semantic_fallback=False,
            operation_sequence=operation_sequence,
            recursive_reduce_max_steps=recursive_reduce_max_steps,
        )
        cycles.append(
            {
                "cycle": cycle_index,
                "input": current,
                "final_output": str(sequence_result["final_text"]),
                "last_expanded": str(sequence_result["last_expanded_sentence"] or ""),
                "last_reduced": str(sequence_result["last_reduced_sentence"] or ""),
                "reduction_diagnostics": dict(sequence_result.get("reduction_diagnostics", {})),
                "steps": list(sequence_result["steps"]),
            }
        )
        cycle_diagnostics.append(dict(sequence_result.get("reduction_diagnostics", {})))
        current = str(sequence_result["final_text"])

    return {
        "kind": "m2",
        "input_text": text,
        "definition_style": definition_style,
        "cycle_count": cycle_count,
        "operation_sequence": list(operation_sequence),
        "recursive_reduce_max_steps": recursive_reduce_max_steps,
        "cycles": cycles,
        "reduction_diagnostics": _merge_reduction_diagnostics(cycle_diagnostics),
        "final_output": current,
    }


def _run_m3(
    *,
    prompt_text: str,
    max_entries: int,
    limit_per_pos: int,
    seed_count: int,
    iterations: int,
    output_dir: str,
) -> dict[str, Any]:
    seeds = _parse_m3_seed_prompt(prompt_text)
    pipeline = prepare_pipeline(max_entries=max_entries, limit_per_pos=limit_per_pos)
    result = map_attractors(
        lexicon=pipeline.lexicon,
        seeds=seeds,
        seed_count=seed_count,
        iterations=iterations,
        output_dir=output_dir,
        mode_used=0,
        pipeline=pipeline,
    )
    top_basins = sorted(result.basin_summary.items(), key=lambda item: item[1], reverse=True)[:10]
    return {
        "kind": "m3",
        "seed_prompt": prompt_text,
        "seed_override_used": list(seeds) if seeds else [],
        "summary": {
            "seed_count": result.seed_count,
            "iterations": result.iterations,
            "fixed_points": result.fixed_points,
            "cycles": result.cycles,
            "average_drift": result.average_drift,
            "average_entropy": result.average_entropy,
        },
        "top_basins": top_basins,
        "output_files": list(result.output_files),
    }


def _run_sentence_mode_style_matrix(
    *,
    text: str,
    pipeline: LRCPipeline,
    milestone_key: str,
    cycle_count: int,
    operation_sequence: tuple[str, ...] | None,
    recursive_reduce_max_steps: int,
) -> dict[str, Any]:
    styles = ("literal_first", "cloud_compact")
    matrix_results: list[dict[str, Any]] = []

    for definition_style in styles:
        if milestone_key == "m1":
            result = _run_m1(
                text=text,
                pipeline=pipeline,
                definition_style=definition_style,
            )
            final_output = str(result["cycle"]["reduced_sentence"])
        else:
            if operation_sequence is None:
                raise ValueError("Milestone 2 sequence is invalid. Fix the custom sequence first.")
            result = _run_m2(
                text=text,
                pipeline=pipeline,
                definition_style=definition_style,
                cycle_count=cycle_count,
                operation_sequence=operation_sequence,
                recursive_reduce_max_steps=recursive_reduce_max_steps,
            )
            final_output = str(result["final_output"])

        matrix_results.append(
            {
                "definition_style": definition_style,
                "final_output": final_output,
            }
        )

    return {
        "kind": "matrix",
        "milestone_key": milestone_key,
        "input_text": text,
        "styles_tested": list(styles),
        "cycle_count": cycle_count if milestone_key == "m2" else 1,
        "operation_sequence": list(operation_sequence) if milestone_key == "m2" and operation_sequence else [],
        "recursive_reduce_max_steps": (
            recursive_reduce_max_steps if milestone_key == "m2" else None
        ),
        "results": matrix_results,
    }


def _render_m1_response(payload: dict[str, Any]) -> None:
    cycle = payload["cycle"]
    _render_card("Reduced Output", str(cycle["reduced_sentence"]))
    st.caption(f"definition_style={payload['definition_style']}")
    if cycle.get("reducer_config"):
        st.caption(f"reducer_config={json.dumps(cycle['reducer_config'], sort_keys=True)}")

    with st.expander("Expanded Sentence", expanded=False):
        st.code(str(cycle["expanded_sentence"]))

    with st.expander("Token -> Definition Mapping", expanded=False):
        for item in cycle["token_mappings"]:
            st.markdown(f"- `{item['token']}` -> {item['definition']}")

    with st.expander("Reduction Segments", expanded=False):
        for index, segment in enumerate(cycle["reduction_segments"], start=1):
            confidence = segment.get("confidence")
            confidence_text = "n/a" if confidence is None else f"{float(confidence):.3f}"
            st.markdown(
                f"{index}. `{segment['source_tokens']}` -> `{segment['reduced_phrase']}` "
                f"via `{segment['method']}` (confidence: {confidence_text})"
            )

    with st.expander("Reduction Diagnostics", expanded=False):
        st.json(cycle.get("reduction_diagnostics", {}))

def _render_m2_response(payload: dict[str, Any]) -> None:
    sequence = tuple(str(step) for step in payload["operation_sequence"])
    _render_card(
        "Final Output",
        str(payload["final_output"]),
    )
    st.caption(
        " | ".join(
            [
                f"cycles={payload['cycle_count']}",
                f"sequence={_format_operation_sequence(sequence)}",
                f"recursive_max_steps={payload['recursive_reduce_max_steps']}",
                f"definition_style={payload['definition_style']}",
            ]
        )
    )
    if payload.get("reduction_diagnostics"):
        with st.expander("Sequence Reduction Diagnostics", expanded=False):
            st.json(payload["reduction_diagnostics"])

    with st.expander("Cycle and Step Trace", expanded=False):
        for cycle in payload["cycles"]:
            with st.expander(
                f"Cycle {cycle['cycle']} | input: {cycle['input']} | output: {cycle['final_output']}",
                expanded=False,
            ):
                if cycle.get("reduction_diagnostics"):
                    st.caption("Cycle reduction diagnostics")
                    st.json(cycle["reduction_diagnostics"])
                if cycle["last_expanded"]:
                    st.markdown("**Last Expanded Sentence**")
                    st.code(str(cycle["last_expanded"]))
                if cycle["last_reduced"]:
                    st.markdown("**Last Reduced Sentence**")
                    st.code(str(cycle["last_reduced"]))

                for step in cycle["steps"]:
                    st.markdown(f"**Step {step['step']} - {str(step['operation']).upper()}**")
                    if step.get("implicit_expand"):
                        st.caption("Implicit expansion was applied before this reduction.")
                    if str(step["operation"]) == "recursive_reduce":
                        st.markdown(
                            f"Stop reason: `{step['recursive_stop_reason']}` "
                            f"after `{step['recursive_iteration_count']}` recursive reductions."
                        )
                        with st.expander("Recursive Reduction Trace", expanded=False):
                            for recursive_step in step["recursive_iterations"]:
                                st.markdown(
                                    f"- depth {recursive_step['depth']}: "
                                    f"`{recursive_step['input_text']}` -> `{recursive_step['output_text']}`"
                                )
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("Input")
                        st.code(str(step["input_text"]))
                    with col_b:
                        st.markdown("Output")
                        st.code(str(step["output_text"]))
                    if step.get("reduction_diagnostics"):
                        st.caption("Step reduction diagnostics")
                        st.json(step["reduction_diagnostics"])

def _render_matrix_response(payload: dict[str, Any]) -> None:
    _render_card(
        "Batch Style Test Complete",
        (
            f"runs={len(payload['results'])} | milestone={str(payload['milestone_key']).upper()} | "
            f"styles={', '.join(payload['styles_tested'])}"
        ),
    )
    st.caption(f"input={payload['input_text']}")
    if payload["milestone_key"] == "m2":
        sequence = tuple(str(item) for item in payload["operation_sequence"])
        st.caption(
            " | ".join(
                [
                    f"cycles={payload['cycle_count']}",
                    f"sequence={_format_operation_sequence(sequence)}",
                    f"recursive_max_steps={payload['recursive_reduce_max_steps']}",
                ]
            )
        )

    with st.expander("Matrix Results", expanded=False):
        for item in payload["results"]:
            st.markdown(
                f"**definition_style={item['definition_style']}**"
            )
            st.code(str(item["final_output"]))


def _render_m3_response(payload: dict[str, Any], *, response_key: str) -> None:
    summary = payload["summary"]
    _render_card(
        "Attractor Run Complete",
        (
            f"seeds={summary['seed_count']}, iterations={summary['iterations']}, "
            f"fixed_points={summary['fixed_points']}, cycles={summary['cycles']}"
        ),
    )
    st.caption(
        f"average_drift={summary['average_drift']:.4f} | average_entropy={summary['average_entropy']:.4f}"
    )

    if payload["seed_override_used"]:
        st.markdown("**Seed Override Used**")
        st.code(", ".join(payload["seed_override_used"]))

    if payload["top_basins"]:
        st.markdown("**Top Basins**")
        for word, count in payload["top_basins"]:
            st.markdown(f"- `{word}`: {count}")

    st.markdown("**Artifacts**")
    for index, file_path in enumerate(payload["output_files"]):
        path = Path(file_path)
        if not path.exists():
            continue
        st.download_button(
            label=f"Download {path.name}",
            data=path.read_bytes(),
            file_name=path.name,
            mime="application/octet-stream",
            key=f"{response_key}-download-{index}-{path.name}",
        )


def _render_assistant_response(message: dict[str, Any], *, response_key: str) -> None:
    payload = message["payload"]
    if payload["kind"] == "m1":
        _render_m1_response(payload)
        return
    if payload["kind"] == "m2":
        _render_m2_response(payload)
        return
    if payload["kind"] == "matrix":
        _render_matrix_response(payload)
        return
    _render_m3_response(payload, response_key=response_key)


def _run_prompt(
    *,
    prompt_text: str,
    settings: dict[str, Any],
    pipeline: LRCPipeline,
) -> dict[str, Any]:
    milestone_key = settings["milestone_key"]
    if milestone_key == "m1":
        return _run_m1(
            text=prompt_text,
            pipeline=pipeline,
            definition_style=settings["definition_style"],
        )

    if milestone_key == "m2":
        sequence = settings["operation_sequence"]
        if sequence is None:
            raise ValueError("Milestone 2 sequence is invalid. Fix the custom sequence first.")
        return _run_m2(
            text=prompt_text,
            pipeline=pipeline,
            definition_style=settings["definition_style"],
            cycle_count=settings["cycle_count"],
            operation_sequence=sequence,
            recursive_reduce_max_steps=settings["recursive_reduce_max_steps"],
        )

    return _run_m3(
        prompt_text=prompt_text,
        max_entries=settings["max_entries"],
        limit_per_pos=settings["limit_per_pos"],
        seed_count=settings["seed_count"],
        iterations=settings["iterations"],
        output_dir=settings["output_dir"],
    )


def _chat_input_placeholder(milestone_key: str) -> str:
    if milestone_key == "m1":
        return "Enter a sentence to run one expand/reduce cycle..."
    if milestone_key == "m2":
        return "Enter a seed sentence for recursive sequence cycles..."
    return "Optional: enter comma-separated seed words, or type 'auto' to sample."


def main() -> None:
    st.set_page_config(page_title="LRC Chat Workspace", layout="wide")
    _inject_styles()
    _ensure_thread_state()

    pipeline = prepare_pipeline(max_entries=6000, limit_per_pos=0)
    settings = _render_sidebar_controls()
    thread_id = str(settings["active_thread_id"])
    thread_record: dict[str, Any] = st.session_state["lrc_threads"][thread_id]
    history: list[dict[str, Any]] = list(thread_record.get("messages", []))
    thread_record["milestone_hint"] = settings["milestone_key"]

    if settings["clear_chat"]:
        thread_record["messages"] = []
        thread_record["updated_at"] = _now_stamp()
        st.rerun()

    st.markdown(
        '<div class="lrc-title-wrap"><h1 class="lrc-title">LRC Chat Workspace</h1></div>',
        unsafe_allow_html=True,
    )
    _render_card(
        settings["milestone_label"],
        (
            "Conversational workspace with sidebar controls for sequence and reduction behavior. "
            "Responses are rendered as step cards and expanders (no spreadsheet output)."
        ),
    )

    for idx, message in enumerate(history):
        with st.chat_message(message["role"]):
            stamp = str(message.get("timestamp", "")).strip()
            milestone = str(message.get("milestone", "")).strip().upper()
            if stamp:
                if milestone:
                    st.caption(f"{stamp} | {milestone}")
                else:
                    st.caption(stamp)
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                _render_assistant_response(message, response_key=f"{thread_id}-{idx}")

    if settings["run_mode_style_matrix"]:
        matrix_input = str(settings["matrix_input_text"]).strip()
        if not matrix_input:
            st.error("Enter Batch Input Text before running matrix tests.")
            return
        if settings["milestone_key"] == "m2" and settings["sequence_error"]:
            st.error("Fix the Milestone 2 custom sequence error before running matrix tests.")
            return
        if settings["milestone_key"] not in {"m1", "m2"}:
            st.error("Batch style matrix is available only for Milestone 1 and Milestone 2.")
            return

        user_message = {
            "role": "user",
            "content": f"[Batch Style Test] {matrix_input}",
            "timestamp": _now_stamp(),
            "milestone": settings["milestone_key"],
            "settings": _settings_snapshot(settings),
        }
        history.append(user_message)
        with st.spinner("Running unified style sweep..."):
            payload = _run_sentence_mode_style_matrix(
                text=matrix_input,
                pipeline=pipeline,
                milestone_key=settings["milestone_key"],
                cycle_count=settings["cycle_count"],
                operation_sequence=settings["operation_sequence"],
                recursive_reduce_max_steps=settings["recursive_reduce_max_steps"],
            )
        assistant_message = {
            "role": "assistant",
            "payload": payload,
            "timestamp": _now_stamp(),
            "milestone": settings["milestone_key"],
            "settings": _settings_snapshot(settings),
        }
        history.append(assistant_message)
        thread_record["messages"] = history
        thread_record["updated_at"] = _now_stamp()
        st.rerun()

    prompt = st.chat_input(_chat_input_placeholder(settings["milestone_key"]))
    if prompt:
        if settings["milestone_key"] == "m2" and settings["sequence_error"]:
            st.error("Fix the Milestone 2 custom sequence error before sending a prompt.")
            return

        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": _now_stamp(),
            "milestone": settings["milestone_key"],
            "settings": _settings_snapshot(settings),
        }
        history.append(user_message)
        with st.spinner("Running LRC pipeline..."):
            payload = _run_prompt(prompt_text=prompt, settings=settings, pipeline=pipeline)
        assistant_message = {
            "role": "assistant",
            "payload": payload,
            "timestamp": _now_stamp(),
            "milestone": settings["milestone_key"],
            "settings": _settings_snapshot(settings),
        }
        history.append(assistant_message)
        thread_record["messages"] = history
        thread_record["updated_at"] = _now_stamp()
        st.rerun()


if __name__ == "__main__":
    main()
