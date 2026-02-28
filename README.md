# Technical Reference: Lexical Reduction and Convolution (LRC)

## Overview

**Lexical Reduction and Convolution** is a bidirectional, LLM-mediated methodology for processing and compressing semantic information. Rather than compressing text at the character or token level, this framework operates directly on meaning. It functions through a two-step process:

1. Expand a lexical input into a dense semantic cloud.
2. Convolve that cloud back into an optimized lexical representation.

## Core Mechanisms

The framework consists of two primary, opposing operations.

### Lexical Expansion (Semantic Unpacking)

- **Process:** A source word, phrase, or text is expanded into its semantic components.
- **Mechanism:** Dictionary definitions and LLM-formulated contextual meanings are combined into a comprehensive definition cloud.
- **Output:** A detailed slice of text that captures explicit definitions, implicit connotations, and contextual boundaries.

### Lexical Convolution (Semantic Reduction)

- **Process:** The inverse of expansion. A broad semantic slice is compressed.
- **Mechanism:** The system searches for the best word or highly condensed phrase that encapsulates the expanded meaning.
- **Output:** A dense lexical representation that acts as the root of the expanded concept.

## Workflow

1. **Input:** A baseline word, phrase, or sentence.
2. **Phase 1 (Expand):** Generate a high-dimensional semantic cloud under dictionary and model constraints.
3. **Phase 2 (Convolve):** Reduce the expanded cloud to an optimal lexical fit.
4. **Output:** A refined word or phrase that reformulates the original meaning.

## Theoretical Properties and System Dynamics

While the base mechanics act as a semantic compressor, recursive application unlocks dynamical behaviors.

When the output of Convolution is fed back into Expansion repeatedly, the system can be treated as a topological model of meaning. Key behaviors to test and document:

- **Fixed-Point Convergence:** Whether a concept locks into an immutable root under repeated cycles.
- **Oscillation:** Whether the system loops between related concepts (for example, "sadness" and "grief").
- **Semantic Drift or Collapse:** Whether meaning decays, loses context, or flattens into generic noise.

## Potential Applications

- **Semantic Compression:** Store complex ideas as dense, retrievable seed words or short phrases.
- **Concept Refinement:** Convolve long-form text to core conceptual roots.
- **Latent Space Mapping:** Map semantic attractor basins by observing recursive stability and collapse dynamics.

## Governance and ADRs

- Architectural Decision Records (ADRs) are stored in [docs/adr/README.md](docs/adr/README.md).
- All implementation and process rules should be recorded through ADRs.

## Repository Setup (Milestone 3)

Milestone 3 introduces a modular semantic reduction engine, recursive attractor analysis, mode-selectable pipelines (Mode 0-4), and a Streamlit dashboard.

Detailed setup and run instructions are maintained with the codebase and must be updated alongside implementation changes.