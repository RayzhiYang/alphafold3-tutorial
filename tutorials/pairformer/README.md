# AlphaFold 3: Pairformer Stack

## Role in AlphaFold 3 Architecture

The Pairformer Stack is the computational core of AlphaFold 3's representation refinement network. It sits between the Input Embedder and the Diffusion Module. Its purpose is to take the initial `single` and `pair` representations and iteratively process them to uncover and refine the complex relationships that define a biomolecule's structure.

`single_repr`, `pair_repr` -> **[Pairformer]** -> `refined_single_repr`, `refined_pair_repr`

This module is an evolution of the Evoformer stack from AlphaFold 2. For this tutorial, we focus on a simplified version that emphasizes the key innovations, particularly **Triangular Self-Attention**, while omitting more complex features like MSA processing for clarity.

## How a Pairformer Block Works

The Pairformer Stack consists of a series of identical `PairformerBlock`s. Each block applies a sequence of transformations that allow information to flow within and between the `single` and `pair` representations. The main components you will implement in a block are:

1.  **Pairwise Representation Processing:**
    *   **Triangular Multiplicative Updates:** Before attention, the `pair_repr` is updated using multiplicative interactions. This is done in two ways:
        *   **Outgoing:** For a pair `(i, j)`, it aggregates information from all pairs `(i, k)` and `(j, k)` by summing over `k`.
        *   **Incoming:** For a pair `(i, j)`, it aggregates information from all pairs `(k, i)` and `(k, j)` by summing over `k`.
        This allows the model to start reasoning about geometric relationships even before the main attention mechanism.
    *   **Triangular Self-Attention:** This is the most critical operation. It updates the `pair_repr` by reasoning about geometric relationships. Instead of attending over all pairs, it uses triangles of residues `(i, j, k)` to update the representation for a given pair `(i, j)`. This is an **axial attention** mechanism where for a given pair `(i, j)`, attention is performed along the "axis" of the third residue `k`. This is done in two ways:
        *   **Starting Node Attention:** Updates the edge `(i, j)` by considering all triangles `(i, j, k)` where `i` is the "starting" node. It attends over all `k`, aggregating information from pairs `(i, k)`.
        *   **Ending Node Attention:** Updates the edge `(i, j)` by considering all triangles `(i, j, k)` where `j` is the "ending" node. It attends over all `k`, aggregating information from pairs `(k, j)`.
        This mechanism allows the model to enforce geometric consistency (like the triangle inequality) directly into the `pair_repr`.
    *   **Pairwise Transition:** After attention, a standard feed-forward network (also called a transition layer) is applied independently to each element in the `pair_repr`. This helps to process and integrate the information gathered during the attention step.

2.  **Communication from Pair to Single Representation:**
    *   A key feature of the Pairformer is the flow of information from the `pair_repr` to the `single_repr`. This allows the pairwise interaction information to refine the representation of individual residues.
    *   This is done by a **Pair-to-Single** communication module, which in this simplified version uses **Single Attention with Pair Bias**: For each residue `i`, the model performs self-attention on the `single_repr`, but the attention scores are biased by the corresponding `pair_repr` values `(i, j)`. This allows the `pair_repr` to guide how information is aggregated for the `single_repr`.
    *   After the communication, a transition layer is applied to the `single_repr` to process the newly incorporated information. In the AlphaFold 3 architecture, information does not flow from the single to the pair representation within the Pairformer block.

## What's Simplified in this Tutorial?

The actual AlphaFold 3 architecture is more complex. For educational purposes, we have simplified or omitted the following:
- **MSA Stack:** We do not process a Multiple Sequence Alignment (MSA). The full architecture includes an MSA stack that runs in parallel with the Pairformer and exchanges information with it.
- **Recycling:** We perform a single pass through the Pairformer. The actual model "recycles" the outputs back as inputs for several iterations to further refine the representations.
- **Gating and other small details:** Many of the connections in the real model are gated. We use simple residual connections for clarity.

## Tensor Shape Changes

A key feature of the Pairformer Stack is that the shapes of the representations are preserved throughout the stack.

-   **Input to a `PairformerBlock`:**
    -   `single_repr`: `(N_res, d_single)`
    -   `pair_repr`: `(N_res, N_res, d_pair)`
-   **Output of a `PairformerBlock`:**
    -   `updated_single_repr`: `(N_res, d_single)`
    -   `updated_pair_repr`: `(N_res, N_res, d_pair)`

While the shapes don't change, the values within the tensors are continuously refined with each block, becoming richer in structural information.

## Key Motivation and Purpose

The Pairformer is designed to explicitly model the pairwise interactions between all residues.
-   **Triangular Attention** is the key innovation that allows the model to reason about the geometry of the structure while still working with a 2D representation. It's like solving a big puzzle where each piece (a pair representation) is adjusted based on its neighbors until a globally consistent picture emerges.
-   **Iterative Refinement** through many blocks allows the model to build up a complex and accurate picture of the protein's structure from simple sequence and distance information.

Your task is to implement the `PairformerStack` and the `PairformerBlock` in the `pairformer.py` file. You will need to focus on structuring the block correctly and implementing the placeholder modules for triangular attention and transition layers.