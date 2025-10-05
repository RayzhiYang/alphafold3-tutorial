# AlphaFold 3: Input Embedder

## Role in AlphaFold 3 Architecture

The Input Embedder is the first module in the AlphaFold 3 architecture. Its primary role is to take the raw input features and transform them into the initial `single` and `pair` representations. These representations are the foundation upon which the rest of the model builds its understanding of the biomolecular structure.

This initial embedding is crucial as it sets up the representations with the necessary dimensions and injects fundamental information, like sequence identity and relative positions, before the more complex reasoning in the Pairformer stack begins.

`Input Features` -> **[Input Embedder]** -> `single_repr`, `pair_repr`

## How it Works

The Input Embedder produces two main tensors:

1.  **Single Representation (`single_repr`):** A tensor containing information about each individual residue in the sequence.
2.  **Pair Representation (`pair_repr`):** A tensor containing information about the relationship between every pair of residues.

The process involves two key steps that you will implement:

### 1. Target Sequence Embedding

The model first needs to understand the identity of each amino acid in the sequence.

-   The input `target_seq` is a list of integers, where each integer represents an amino acid.
-   This is converted into a **one-hot encoding**. For a sequence of length `N_res`, this creates a tensor of shape `(N_res, num_amino_acids)`, where each row is a vector of zeros with a single '1' at the index corresponding to the amino acid type.
-   This one-hot representation is then passed through a **linear layer** (a dense neural network layer) to project it into a higher-dimensional space, creating the initial `single_repr`.

### 2. Relative Positional Encoding

The model needs to know how far apart residues are in the sequence. This is vital for understanding the local and global structure of the protein. This is achieved by creating a `pair_repr` that encodes relative positions.

-   First, a matrix of relative distances is computed. For a pair of residues `(i, j)`, the relative distance is simply `i - j`.
-   These distances are then **clipped** to a fixed range (e.g., `[-32, 32]`). This is because the exact distance between very distant residues is less important than the fact that they are far apart.
-   The clipped values are shifted to be non-negative, so they can be used as indices for an **embedding layer**.
-   This embedding layer maps each relative position value to a vector of dimension `d_pair`, creating the initial `pair_repr`.

## Tensor Shape Changes

-   **Input:**
    -   `features['target_seq']`: `(N_res,)`
-   **Intermediate:**
    -   `target_seq_one_hot`: `(N_res, num_amino_acids)`
    -   `initial_single_repr` (after linear projection): `(N_res, d_single)`
    -   `relative_positions`: `(N_res, N_res)`
    -   `embedded_relative_positions`: `(N_res, N_res, d_pair)`
-   **Output:**
    -   `single_repr`: `(N_res, d_single)`
    -   `pair_repr`: `(N_res, N_res, d_pair)`

## Key Motivation and Purpose

The motivation behind the Input Embedder is to create rich, high-dimensional representations of the input sequence and its pairwise relationships.
-   The `single_repr` allows the model to reason about the properties of individual residues.
-   The `pair_repr`, initialized with relative positional encodings, gives the model a built-in "ruler" to measure distances along the polymer chain, which is a fundamental prior for understanding protein folding.

Your task is to implement the `InputEmbedder` module in the `input_embedder.py` file, following the instructions in the template. You will need to define the linear and embedding layers and implement the logic for creating the single and pair representations.