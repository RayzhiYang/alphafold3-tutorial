import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any
from dataclasses import dataclass
from functools import partial

# Flax and Optax for training
import optax
from flax.training import train_state


@dataclass(frozen=True)
class DiffusionConfig:
    """Configuration for the Diffusion Module."""

    d_single: int = 384
    d_pair: int = 128
    num_timesteps: int = 50  # Reduced for tutorial speed
    beta_start: float = 1e-4
    beta_end: float = 0.02
    d_timestep: int = 64
    d_hidden: int = 256
    n_atoms_per_res: int = 5  # Simplified assumption


class TimestepEmbedding(nn.Module):
    """Converts integer timesteps into a continuous embedding."""

    config: DiffusionConfig

    @nn.compact
    def __call__(self, timestep: jnp.ndarray) -> jnp.ndarray:
        """
        Creates a sinusoidal embedding for the given timestep.

        Args:
            timestep: A batch of integer timesteps, shape (B,).

        Returns:
            A batch of embeddings, shape (B, d_timestep).
        """
        # This gives the model information about its position in the diffusion process.

        # 1. Create a vector of frequencies based on the formula from the Transformer paper.
        #    The formula for the `i`-th frequency is `1 / (10000^(2i / d_timestep))`.
        #    The shape should be (d_timestep / 2,).
        #    Hint: You can express `10000` as `jnp.exp(jnp.log(10000.0))` to help formulate the expression.
        frequencies = 10000.0 ** (
            -2.0 * jnp.arange(self.config.d_timestep // 2) / self.config.d_timestep
        )

        # 2. Combine the timesteps and frequencies.
        #    Remember that `timestep` has shape (B,) and frequencies have shape (d_timestep / 2,).
        #    You'll need to adjust their dimensions to perform element-wise multiplication that results in a shape of (B, d_timestep / 2).
        #    Hint: `jnp.expand_dims` or using `None` for indexing can add a new axis.
        timestamp_embeddings = timestep[:, None] * frequencies[None, :]

        # 3. Create the final embedding by applying `sin` and `cos` to the arguments and concatenating them.
        #    The final shape should be (B, d_timestep).
        #    Hint: `jnp.concatenate` is the function you need.
        timestamp_embeddings = jnp.concatenate(
            [jnp.sin(timestamp_embeddings), jnp.cos(timestamp_embeddings)], axis=-1
        )

        # 4. To allow for a more expressive representation of time, pass the embedding through a small MLP.
        #    This MLP should consist of two linear layers with a ReLU activation in between.
        #    Hint: Use `nn.Dense` for the linear layers and `nn.relu` for the activation.
        timestamp_embeddings = nn.Dense(self.config.d_timestep)(timestamp_embeddings)
        timestamp_embeddings = nn.relu(timestamp_embeddings)
        timestamp_embeddings = nn.Dense(self.config.d_timestep)(timestamp_embeddings)

        return timestamp_embeddings


class DenoisingNetwork(nn.Module):
    """A simplified denoising network that predicts noise."""

    config: DiffusionConfig

    @nn.compact
    def __call__(
        self,
        noised_coords: jnp.ndarray,
        t_embedding: jnp.ndarray,
        conditioning: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Predicts noise based on the current state and conditioning.

        Args:
            noised_coords: The noisy atom coordinates, shape (N_atoms, 3).
            t_embedding: The timestep embedding for a single timestep, shape (1, d_timestep).
            conditioning: The conditioning vector for each atom, shape (N_atoms, d_hidden).

        Returns:
            The predicted noise, shape (N_atoms, 3).
        """
        n_atoms = noised_coords.shape[0]

        # This network learns to predict the noise that was added to the original coordinates.

        # 1. The timestep embedding `t_embedding` has shape (1, d_timestep), but it needs to be combined
        #    with the per-atom `noised_coords` and `conditioning` tensors.
        #    Broadcast the timestep embedding so that it has a shape of (n_atoms, d_timestep).
        #    Hint: `jnp.tile` can be used to repeat an array.
        t_embedding = jnp.tile(t_embedding, (n_atoms, 1))

        # 2. Combine the `noised_coords`, broadcasted timestep embedding, and `conditioning` tensors
        #    into a single input tensor for the MLP.
        #    Hint: Concatenate the tensors along the feature dimension (`axis=-1`).
        combined_input = jnp.concatenate(
            [noised_coords, t_embedding, conditioning], axis=-1
        )

        # 3. Process the combined input through a simple MLP to predict the noise.
        #    The network should have two hidden layers with ReLU activations and an output layer.
        #    The output dimension should match the dimension of the noise (3 for 3D coordinates).
        #    Hint: Use `nn.Dense` for the layers and `nn.relu` for activations.
        noise = nn.Dense(self.config.d_hidden)(combined_input)
        noise = nn.relu(noise)
        noise = nn.Dense(self.config.d_hidden)(noise)
        noise = nn.relu(noise)
        noise = nn.Dense(3)(noise)
        return noise


class DiffusionModule(nn.Module):
    """The main module for the diffusion process."""

    config: DiffusionConfig

    def setup(self):
        self.timestep_embedding = TimestepEmbedding(
            config=self.config, name="timestep_embedding"
        )
        self.denoising_net = DenoisingNetwork(config=self.config, name="denoising_net")
        self.conditioning_projection = nn.Dense(
            self.config.d_hidden, name="conditioning_projection"
        )

    def __call__(
        self,
        noised_coords: jnp.ndarray,
        timestep: jnp.ndarray,
        single_repr: jnp.ndarray,
        pair_repr: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Predicts the noise added to the coordinates at a given timestep.
        This is the core function called during both training and sampling.
        """
        n_atoms = noised_coords.shape[0]
        n_residues = single_repr.shape[0]

        # 1. Timestep Embedding
        # The timestep input is a scalar for inference, but a vector for training.
        # Ensure it's always an array to handle both cases.
        t_embedding = self.timestep_embedding(jnp.atleast_1d(timestep))

        # 2. Conditioning Information
        # This is a highly simplified conditioning model for the tutorial.
        # The goal is to create a vector for each atom that contains information
        # about the residue it belongs to.

        # a. To create a basic residue-level conditioning vector, you can combine information
        #    from both `single_repr` and `pair_repr`.
        #    A simple approach is to average the `pair_repr` over one of its dimensions
        #    and then concatenate the result with `single_repr`.
        #    Hint: `jnp.mean` and `jnp.concatenate` will be useful here.
        conditioning = jnp.concatenate(
            [single_repr, jnp.mean(pair_repr, axis=1)], axis=-1
        )

        # b. Use the `self.conditioning_projection` layer to transform the residue-level
        #    conditioning vector into the desired hidden dimension for the denoising network.
        conditioning = self.conditioning_projection(conditioning)

        # c. The conditioning is at the residue level, but the denoising happens at the atom level.
        #    Expand the residue-level conditioning so that each atom in a residue receives the same signal.
        #    Hint: `jnp.repeat` can be used to duplicate entries along an axis.
        conditioning = jnp.repeat(conditioning, self.config.n_atoms_per_res, axis=0)

        # 3. Denoising Network
        # Predict the noise from the noised coordinates, timestep, and conditioning.
        predicted_noise = self.denoising_net(noised_coords, t_embedding, conditioning)

        return predicted_noise


def generate_noise_schedule(config: DiffusionConfig) -> Dict[str, jnp.ndarray]:
    """
    Generates the noise schedule (betas, alphas, and alpha cumulative products)
    for the forward diffusion process. This is a fixed, not learned, schedule.
    """
    # These values determine how much noise is added at each step of the forward process.

    # 1. `beta` is typically scheduled to increase linearly over the timesteps.
    #    Create a tensor of `beta` values starting from `beta_start` and ending at `beta_end`.
    #    Hint: `jnp.linspace` is perfect for creating linearly spaced points.
    betas = jnp.linspace(config.beta_start, config.beta_end, config.num_timesteps)

    # 2. `alpha_t` is defined as `1.0 - beta_t`. Calculate this for all betas.
    alphas = 1.0 - betas

    # 3. `alpha_bar_t` is the cumulative product of `alpha` values up to timestep `t`.
    #    This value is crucial for the "closed-form" noising formula.
    #    Hint: Look for a JAX function that computes the cumulative product of an array.
    alphas_cumprod = jnp.cumprod(alphas)

    # 4. Return a dictionary containing these three tensors.
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
    }


@partial(jax.jit, static_argnums=(0,))
def train_step(
    model: DiffusionModule,
    state: train_state.TrainState,
    batch: Dict[str, jnp.ndarray],
    key: jax.random.PRNGKey,
) -> Tuple[train_state.TrainState, jnp.ndarray]:
    """Performs a single training step."""

    def loss_fn(params):
        # The goal is to predict the noise that was added to the true coordinates.

        # 1. Unpack the batch.
        true_coords, single_repr, pair_repr = (
            batch["coords"],
            batch["single"],
            batch["pair"],
        )
        batch_size = true_coords.shape[0]

        # 2. For each item in the batch, we need to sample a random timestep `t`.
        #    This ensures the model is trained to denoise across the entire diffusion process.
        #    Hint: Use `jax.random.randint` to generate random integers within the valid timestep range.
        #    Remember to split the PRNG key.
        t_key, noise_key = jax.random.split(key)
        timesteps = jax.random.randint(
            t_key, (batch_size,), 0, model.config.num_timesteps
        )

        # 3. Generate a tensor of random noise from a standard normal distribution.
        #    This noise will be added to the true coordinates and will also serve as the prediction target for the model.
        #    Its shape should match the `true_coords`.
        #    Hint: Use `jax.random.normal`.
        noise = jax.random.normal(noise_key, true_coords.shape)

        # 4. Apply the forward diffusion process "closed-form" formula to get the noised coordinates `x_t`.
        #    The formula is: `x_t = sqrt(α_bar_t) * x_0 + sqrt(1 - α_bar_t) * ε`.
        #    You will need to get the `alphas_cumprod` from the noise schedule and select the values corresponding to the sampled timesteps `t`.
        #    Hint: Pay close attention to the shapes of the tensors. You'll need to reshape the schedule-derived tensors
        #    to broadcast correctly with the coordinates tensor, which has a shape of (B, N, 3).
        alphas_cumprod = generate_noise_schedule(model.config)["alphas_cumprod"]
        alphas_cumprod = alphas_cumprod[timesteps]
        alphas_cumprod = alphas_cumprod[:, None, None]  # (B, 1, 1)
        noised_coords = (
            jnp.sqrt(alphas_cumprod) * true_coords
            + jnp.sqrt(1 - alphas_cumprod) * noise
        )

        # 5. Pass the `noised_coords`, `t`, `single_repr`, and `pair_repr` to the model to get the predicted noise.
        #    Since our model's `__call__` method is designed for a single example, but we have a batch,
        #    we need to use `vmap` to apply the model to each item in the batch.
        #    Hint: `nn.vmap` is the Flax version of `jax.vmap`. You need to specify which input axes correspond to the batch dimension.
        #    Model parameters should not be mapped.
        batched_model_cls = nn.vmap(
            model.__class__,
            in_axes=0,
            out_axes=0,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )
        predicted_noise = batched_model_cls(config=model.config).apply(
            {"params": params},
            noised_coords,
            timesteps,
            single_repr,
            pair_repr,
        )

        # 6. The loss function for a simple diffusion model is typically the mean squared error (MSE)
        #    between the true noise `eps` and the `predicted_noise`.
        #    Hint: The formula for MSE is the mean of the squared differences.
        return jnp.mean(jnp.square(noise - predicted_noise))

    # Calculate the gradients and update the state.
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@partial(jax.jit, static_argnums=(0, 2))
def sample(
    model: DiffusionModule,
    params: Any,
    n_atoms: int,
    single_repr: jnp.ndarray,
    pair_repr: jnp.ndarray,
    key: jax.random.PRNGKey,
) -> jnp.ndarray:
    """Generates a structure from random noise using the DDPM sampling algorithm."""

    # 1. Get the fixed noise schedule.
    noise_schedule = generate_noise_schedule(model.config)
    alphas = noise_schedule["alphas"]
    alphas_cumprod = noise_schedule["alphas_cumprod"]
    betas = noise_schedule["betas"]

    # 2. Start with random noise for the coordinates `x_T`.
    key, noise_key = jax.random.split(key)
    coords = jax.random.normal(noise_key, (n_atoms, 3))

    # 3. Loop from `T-1` down to `0`, iteratively denoising the structure.
    for t in reversed(range(model.config.num_timesteps)):
        # a. Retrieve the `alpha_t`, `alpha_bar_t`, and `beta_t` for the current timestep `t`
        #    from the pre-computed noise schedule.
        alpha_t = alphas[t]
        alpha_bar_t = alphas_cumprod[t]
        beta_t = betas[t]

        # b. Call the model to predict the noise from the current `coords`, the timestep `t`,
        #    and the provided conditioning representations.
        #    Hint: Ensure the timestep `t` is passed as a JAX array.
        noise = model.apply(
            {"params": params}, coords, jnp.array(t), single_repr, pair_repr
        )

        # c. Rearrange the forward process formula to estimate the original coordinates `x_0`
        #    based on the current `x_t` (which is `coords`) and the `predicted_noise`.
        #    The formula is given in the README.
        # x_0_pred = (x_t - sqrt(1 - α_bar_t) * ε_θ) / sqrt(α_bar_t)
        x_0 = (coords - jnp.sqrt(1 - alpha_bar_t) * noise) / jnp.sqrt(alpha_bar_t)

        # d. Calculate the mean of the posterior distribution `q(x_{t-1} | x_t, x_0)`, which will be our `x_{t-1}` before adding noise.
        #    The formula for this involves coefficients derived from the noise schedule and both `x0_pred` and `coords` (`x_t`).
        #    Refer to the DDPM paper or the README for the exact formula.
        #    Hint: Be careful with the `t-1` index, especially when `t` is 0.
        # μ_{t-1} = (sqrt(α_bar_{t-1}) * β_t) / (1 - α_bar_t) * x_0_pred + (sqrt(α_t) * (1 - α_bar_{t-1})) / (1 - α_bar_t) * x_t
        alpha_cumprod_t_minus_1 = jnp.where(
            t > 0, alphas_cumprod[t - 1], jnp.array(1.0)
        )
        mu_t_minus_1 = (jnp.sqrt(alpha_cumprod_t_minus_1) * beta_t) / (
            1 - alpha_bar_t
        ) * x_0 + (jnp.sqrt(alpha_t) * (1 - alpha_cumprod_t_minus_1)) / (
            1 - alpha_bar_t
        ) * coords

        # e. For all steps except the last one (t > 0), add scaled noise to the calculated mean `mu_t_minus_1`.
        #    The variance of this noise is also derived from the noise schedule.
        #    This re-introduces stochasticity, which is key to the generative process.
        #    Hint: Use `jax.random.normal` to generate the noise `z`. The variance formula is in the README.
        #    Use `jnp.where` to conditionally add noise only when `t > 0`.
        key, noise_key = jax.random.split(key)
        z = jax.random.normal(noise_key, coords.shape)
        coords = mu_t_minus_1
        if t > 0:
            coords += jnp.sqrt(beta_t) * z

    # 4. Return the final coordinates `x_0`.
    return coords


# Example usage
def main():
    config = DiffusionConfig()
    model = DiffusionModule(config)

    key = jax.random.PRNGKey(42)
    n_res = 10
    n_atoms = n_res * config.n_atoms_per_res
    batch_size = 4

    # Dummy inputs with a batch dimension
    true_coords = jnp.zeros((batch_size, n_atoms, 3))
    single_repr = jnp.zeros((batch_size, n_res, config.d_single))
    pair_repr = jnp.zeros((batch_size, n_res, n_res, config.d_pair))
    batch = {"coords": true_coords, "single": single_repr, "pair": pair_repr}
    params = {}

    print("--- DiffusionModule Initialization Test ---")
    try:
        # Initialize the model parameters.
        # We need to use a single example from the batch to initialize.
        params = model.init(
            key, true_coords[0], jnp.array(10), single_repr[0], pair_repr[0]
        )["params"]
        print("Model initialized successfully.")

    except Exception as e:
        print(
            f"An error occurred during initialization. This is expected if modules are not implemented."
        )
        print(f"Error: {e}")

    # Only proceed with further tests if initialization was successful
    if params:
        print("--- Noise Schedule Test ---")
        try:
            noise_schedule = generate_noise_schedule(config)
            print("Noise schedule 'betas' shape:", noise_schedule["betas"].shape)
            assert noise_schedule["betas"].shape == (config.num_timesteps,)
            print("All schedule shapes are correct.")
            assert jnp.all(noise_schedule["betas"] > 0), "Betas should be positive."
            assert jnp.all(noise_schedule["alphas"] > 0), "Alphas should be positive."
            assert jnp.all(noise_schedule["alphas_cumprod"] > 0), (
                "Alphas cumulative product should be positive."
            )
            print("Noise schedule content checks passed.")
        except Exception as e:
            print(f"generate_noise_schedule not implemented yet. Error: {e}")

        print("--- Training and Sampling Test ---")
        try:
            # Create a dummy training state
            tx = optax.adam(1e-3)
            state = train_state.TrainState.create(
                apply_fn=model.apply, params=params, tx=tx
            )

            # Test train_step
            state, loss = train_step(model, state, batch, key)
            print(f"Train step loss: {loss:.4f}")

            # Test sample
            generated_coords = sample(
                model, params, n_atoms, single_repr[0], pair_repr[0], key
            )
            print("Generated coords shape:", generated_coords.shape)
            assert generated_coords.shape == true_coords[0].shape
            print("Sampling shape check passed!")

        except Exception as e:
            print(f"Training/sampling functions not implemented yet. Error: {e}")

    print("Test script finished.")


if __name__ == "__main__":
    main()


# ---
# Questions for students:
# 1. In `TimestepEmbedding`, why do we use both `sin` and `cos`? What would happen if we only used one? (Hint: Think about phase shifts and unique representations for each timestep).
# 2. In `DenoisingNetwork`, why is it important to include the timestep embedding `t_embedding` as an input? What information does it provide to the network?
# 3. What is the role of the conditioning signal (`single_repr`, `pair_repr`)? What would happen if we trained the model without it?
# 4. The sampling process seems complex. Why can't we just use our `x0_pred` at each step as the input for the next step? Why do we need the full `mu_t_minus_1` formula that also includes the previous noisy state `x_t`?
# 5. In the `sample` function, we add a random noise vector `z` at every step except the last one. What is the purpose of re-introducing noise during the reverse process?
# ---
