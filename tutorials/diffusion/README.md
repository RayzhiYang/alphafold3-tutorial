# AlphaFold 3: Diffusion Module

## Role in AlphaFold 3 Architecture

The Diffusion Module is the final, generative part of AlphaFold 3. It takes the refined representations from the Pairformer and uses them to generate the 3D coordinates of all atoms in the molecular complex. This is a fundamental shift from AlphaFold 2, which predicted a set of rotations and translations for each residue.

`single_repr`, `pair_repr` -> **[Diffusion Module]** -> `final_atom_coords`

By operating directly on atom coordinates, the diffusion model gives AlphaFold 3 the flexibility to predict the structures of a wide range of molecules, including proteins, nucleic acids, and small ligands. The `single_repr` and `pair_repr` act as a crucial **conditioning signal**, guiding the diffusion process to generate the specific structure encoded in the representations, rather than just a random molecule.

## How the Diffusion Process Works

The diffusion process is a generative modeling technique. It consists of two phases: a fixed "forward process" where noise is progressively added to a known structure, and a learned "reverse process" where the model learns to denoise a random input back into a valid structure.

### 1. The Forward Process (Fixed)

The forward process corrupts the input data (the true atom coordinates `x_0`) by gradually adding Gaussian noise over a series of `T` timesteps. This process is fixed and does not involve any learning.

-   `x_0`: The real, original atom coordinates.
-   `x_t`: The coordinates at timestep `t`.

At each step `t`, we add a small amount of noise according to a predefined schedule `β_t`.

The set of all `β_t` from `t=1` to `T` is called the "noise schedule." These are fixed hyperparameters. The values are chosen such that `β_t` is small and increases with `t`. In this tutorial, we use a **linear schedule**, where `β_t` increases linearly from `β_start = 1e-4` to `β_end = 0.02` over `T=50` timesteps.

A common point of confusion is whether `β_T` needs to be 1 to ensure the final state `x_T` is pure noise. This is not the case. With `β_T = 0.02`, the final `α_T` is `0.98`. However, the crucial factor is the cumulative product `α_bar_T`, which will be very close to zero due to the repeated multiplication of many numbers less than 1. This ensures that `x_T ≈ ε`, meaning the final noised structure is approximately a sample from a standard normal distribution, effectively erasing the original information.

The per-step noising formula is:

`x_t = sqrt(α_t) * x_{t-1} + sqrt(1 - α_t) * ε_{t-1}`

Here, `α_t = 1 - β_t`, and `ε_{t-1}` is a noise vector sampled from a standard normal distribution, `N(0, I)`.

A powerful property of this process is that we can sample `x_t` at any arbitrary timestep `t` directly from `x_0`, without iterating through the intermediate steps. This is achieved using the following formula:

`x_t = sqrt(α_bar_t) * x_0 + sqrt(1 - α_bar_t) * ε`

In this "closed-form" equation:
-   `α_bar_t` is the cumulative product of all `α` values up to `t` (`α_bar_t = Π_{i=1 to t} α_i`).
-   `ε` is the **cumulative noise** vector, which is also sampled from a standard normal distribution, `N(0, I)`. It represents the total noise accumulated from step 0 to `t`.

This gives us the conditional distribution `q(x_t | x_0)`, which is a Gaussian:

`q(x_t | x_0) = N(x_t; sqrt(α_bar_t) * x_0, (1 - α_bar_t) * I)`

This formula states that the noised structure `x_t` is a sample from a normal distribution with:
-   **Mean:** `sqrt(α_bar_t) * x_0`. This is the original structure, scaled down. As `t` increases, `α_bar_t` decreases, so the mean moves closer to zero.
-   **Covariance:** `(1 - α_bar_t) * I`. Here, `I` is the **identity matrix**. This means the noise added to each coordinate is independent, and the variance of the noise is `(1 - α_bar_t)`. As `t` increases, this variance grows, and the structure becomes more dominated by noise.

### 2. The Reverse Process (Learned)

The goal of training is to learn the reverse of the forward process: to start from a noisy input `x_t` and gradually remove the noise to recover the original structure `x_0`.

While we could try to predict the slightly less noisy `x_{t-1}` from `x_t`, it has been shown to be more effective to train a neural network, `ε_θ`, to predict the **total cumulative noise** `ε` that was added to `x_0` to produce `x_t`. The `θ` represents the trainable parameters of the network.

The network `ε_θ` takes the noisy structure `x_t` and the current timestep `t` as input and outputs a prediction of the noise, `ε_pred`.

**What is `ε_θ` predicting?**
From the forward process formula, we can express the true cumulative noise `ε` as:

`ε = (x_t - sqrt(α_bar_t) * x_0) / sqrt(1 - α_bar_t)`

This is the target that our network `ε_θ(x_t, t)` learns to predict. It is not simply `x_t - x_0`, but a scaled version of the difference that isolates the noise component.

By predicting the noise `ε`, we can get a direct estimate of the original, clean structure `x_0` at any timestep by rearranging the forward process formula:

`x_0_pred = (x_t - sqrt(1 - α_bar_t) * ε_θ(x_t, t)) / sqrt(α_bar_t)`

This ability to estimate the final "denoised" structure at any point in the process is a key part of the sampling algorithm.

-   **Input:** The noised coordinates `x_t` and the timestep `t`.
-   **Conditioning:** The `single_repr` and `pair_repr` from the Pairformer.
-   **Output:** The predicted total cumulative noise `ε_θ(x_t, t, conditioning)`.

### 3. The Role of Timestep Embedding

How does the network know whether it's at the beginning (lots of noise) or near the end (a little noise) of the process? It needs to be explicitly told the timestep `t`. However, feeding `t` as a simple integer is not effective for neural networks.

Instead, we convert `t` into a high-dimensional **timestep embedding**. We use the same sinusoidal embedding technique from the original Transformer paper, which maps the integer `t` into a continuous vector. This allows the network to easily distinguish between different timesteps and understand its position in the denoising trajectory. This embedding is then processed by a small MLP and fed into the main denoising network.

### 4. Training Objective

The network is trained with a simple objective: make the predicted noise match the actual noise that was added during the forward process. We use a simple Mean Squared Error (MSE) loss:

`Loss = E[ ||ε - ε_θ(x_t, t, conditioning)||^2 ]`

Where:
-   `ε` is the actual Gaussian noise that was sampled to generate `x_t`.
-   `ε_θ` is the noise predicted by our network.
-   The expectation `E` is taken over all training examples, timesteps, and noise samples.

### 5. Sampling (Generating a Structure)

At inference time, we start the reverse process from timestep `T`. This is where a key approximation is made. The forward process on real data `x_0` produces an `x_T` that is only *approximately* a standard normal distribution, while for sampling, we begin by drawing `x_T` from a *pure* standard normal distribution (`N(0, I)`).

This slight mismatch between the training and inference starting points is a fundamental aspect of DDPMs. The noise schedule is intentionally designed so that for a sufficiently large `T`, the distribution of noised data `q(x_T|x_0)` is indistinguishable from a pure Gaussian `N(0, I)`. Therefore, we can confidently start the reverse process from pure noise.

With this starting point, we iteratively apply our learned denoising network to reverse the diffusion. This algorithm is based on the Denoising Diffusion Probabilistic Models (DDPM) paper.

Here is a more detailed look at the sampling process for each step `t` from `T-1` down to `0`:

1.  **Predict the total noise `ε_θ`** using the current coordinates `x_t`, the timestep `t`, and the conditioning representations.

2.  **Estimate the original structure `x_0`** using the predicted noise. This is the "denoised" `x_0`:
    `x_0_pred = (x_t - sqrt(1 - α_bar_t) * ε_θ) / sqrt(α_bar_t)`

3.  **Calculate the mean of the posterior distribution `q(x_{t-1} | x_t, x_0)`**. This is the crucial reverse step. We use our `x_0_pred` and `x_t` to calculate the mean of the distribution of `x_{t-1}`. The formula is:
    `μ_{t-1} = (sqrt(α_bar_{t-1}) * β_t) / (1 - α_bar_t) * x_0_pred + (sqrt(α_t) * (1 - α_bar_{t-1})) / (1 - α_bar_t) * x_t`

4.  **Sample `x_{t-1}`**. We then sample `x_{t-1}` from a Gaussian with the mean `μ_{t-1}` and a pre-calculated variance `σ_t^2`. This re-introduces a small amount of randomness to guide the sampling process. For the final step (from `t=1` to `t=0`), we do not add this noise. For simplicity and effectiveness, this pre-calculated variance σ_t^2 is set to β_t, an effective simplification from the DDPM paper.
    `x_{t-1} ~ N(μ_{t-1}, σ_t^2 * I)`

After iterating through all timesteps, the final result `x_0` is the model's predicted 3D structure.

#### Why is Iterative Refinement Necessary?

If we can predict `x_0` from `x_t`, why not just start from the pure noise `x_T` and jump directly to our final prediction of `x_0`?

The reason is that the accuracy of our noise prediction `ε_θ(x_t, t)` depends heavily on the timestep `t`. 

-   When `t` is large (e.g., `t ≈ T`), `x_t` is almost pure noise. The model has very little information to work with, so its prediction of the noise (and therefore its estimate of `x_0`) is very inaccurate and unreliable.
-   When `t` is small, `x_t` is much closer to the real structure. The model can make a much more accurate prediction of the small amount of remaining noise.

The iterative process works by **gradually refining the structure**. Each step is a small correction that pushes the noisy structure in the right direction. The model makes a rough guess at the beginning and then improves upon it at each step as the input becomes cleaner and its predictions become more confident. A single, large leap from pure noise to a final structure would be too difficult for the network to learn effectively.

## What's Simplified in this Tutorial?

-   **Conditioning:** The way we incorporate the `single_repr` and `pair_repr` is highly simplified. In the real model, this is a complex network that projects and integrates these representations to provide a detailed, per-atom conditioning signal. Here, you will create a basic conditioning vector by simply averaging the pair representation and concatenating it with the single representation.
-   **Denoising Network:** Our denoising network will be a simple Multi-Layer Perceptron (MLP). The actual network in AlphaFold 3 is a sophisticated equivariant transformer that processes the geometry of the molecule, the timestep embedding, and the conditioning information.
-   **Atom Representation:** We treat all atoms generically. The real model has specific embeddings for different atom types, which are crucial for handling diverse chemistry.
-   **Sampling:** We will implement the standard DDPM sampling algorithm. The actual sampler in AlphaFold 3 might use more advanced techniques (like DDIM) for faster and more accurate generation.

## Your Task

Your task is to implement the core components of this diffusion process in `diffusion.py`:
1.  **`generate_noise_schedule`:** A function to compute the `betas`, `alphas`, and `alphas_cumprod` tensors.
2.  **`TimestepEmbedding`:** A module to convert the integer timestep `t` into a continuous vector embedding.
3.  **`DenoisingNetwork`:** The neural network that predicts the noise.
4.  **`DiffusionModule`:** The main module that orchestrates the process.
5.  **`train_step`:** A function to perform a single training step (forward pass, loss calculation).
6.  **`sample`:** A function to generate a structure from random noise using the trained model.