# Introduction to AlphaFold 3

Welcome to the AlphaFold 3 tutorial! This guide will walk you through implementing the key components of the AlphaFold 3 model, a revolutionary AI system from Google DeepMind for predicting the 3D structure of biomolecular complexes.

This repository is inspired by https://github.com/kilianmandon/alphafold-decoded, a great tutorial on the implementation of AlphaFold using PyTorch. This repository is different in the following aspects:

- This repository uses [JAX](https://jax.readthedocs.io/), and the [Flax Linen](https://flax-linen.readthedocs.io/en/latest/) library from the JAX ecosystem.
- This repository aims at the implementation of AlphaFold 3 instead of AlphaFold. AlphaFold 3 includes novel components such as Pairformer and diffusion module.
- This repository doesn't aim at an exact replicate of AlphaFold 3. Instead, this repository is designed to help people with deep learning foundations to learn the model architecture of AlphaFold 3. Components like feature preprocessing are not included in this repository.

Please also be acknowledged that part of the materials in this repository were produced with the help of Gemini. With the help of the world knowledge and research capabilities from Gemini, I found Gemini to be very good at explaining concepts and design tutorial templates with proper hints, which serves at a fantastic starting point to learn the implementation of complex models like AlphaFold 3.

## What is AlphaFold 3?

AlphaFold 3 is a deep learning model that can predict the structure of proteins, DNA, RNA, and their interactions with each other and with small molecules (ligands). It represents a significant leap forward from its predecessor, AlphaFold 2, by moving from predicting single protein structures to modeling large, complex biological systems.

## Model Input and Output

At a high level, AlphaFold 3 takes information about biomolecular sequences and produces their 3D atomic coordinates.

### Input

The primary input to the model is a dictionary of features that describe the molecule(s) you want to predict. For these tutorials, we will focus on the simplest case: a single protein chain.

-   **`features['target_seq']`**: A tensor representing the amino acid sequence of the protein.
    -   **Shape**: `(N_res,)`, where `N_res` is the number of residues.

### Output

The main output of the model is the predicted 3D structure.

-   **`predicted_coords`**: A tensor containing the (x, y, z) coordinates for each atom.
    -   **Shape**: `(N_atoms, 3)`, where `N_atoms` is the total number of atoms.

## High-Level Architecture: A Journey Through the Tutorials

The AlphaFold 3 model is complex, but we can understand it by breaking it down into three main stages. These stages correspond to the tutorial modules you will be implementing.

Here is the data flow through the core components:

`Input Features` -> **[Input Embedder]** -> `single_repr`, `pair_repr` -> **[Pairformer]** -> `refined_reprs` -> **[Diffusion Module]** -> `3D Structure`

### 1. Input Embedding (`tutorials/input_embedder/`)

This is the starting point. The `InputEmbedder` takes raw features (like the protein sequence) and converts them into two initial representations:
-   **`single_repr`**: Holds information about each individual residue.
-   **`pair_repr`**: Holds information about every pair of residues.

### 2. Representation Refinement (`tutorials/pairformer/`)

This is the processing core. The `Pairformer` is a large stack of blocks that iteratively refines the `single_repr` and `pair_repr`. It allows information to flow back and forth between the representations, enabling the model to learn the complex geometric constraints of biomolecular structures.

### 3. Structure Generation (`tutorials/diffusion/`)

This is the final and most novel part of AlphaFold 3. The `DiffusionModule` takes the highly refined representations from the Pairformer and uses them to generate the final 3D atomic coordinates. It employs a powerful generative technique called a "diffusion model" to first build and then denoise a cloud of atoms into a coherent and accurate structure.

## How to Use These Tutorials

We recommend you proceed through the tutorials in the order presented above:

1.  **`input_embedder`**: Start here to understand how the model prepares its data.
2.  **`pairformer`**: Move on to the core logic of the model.
3.  **`diffusion`**: Finish with the structure generation module.

For each component, read the `README.md` file in its subdirectory to understand the concepts before attempting to implement the corresponding Python code. The Python files are templates that guide you through the implementation using the Flax library in JAX. Good luck!