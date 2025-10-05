import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Tuple

# A config dataclass is a good practice for managing hyperparameters.
from dataclasses import dataclass


@dataclass
class InputEmbedderConfig:
    """Configuration for the InputEmbedder."""

    d_single: int = 384
    d_pair: int = 128
    num_amino_acids: int = 21  # 20 standard + 1 unknown
    max_relative_pos: int = 32  # As used in AlphaFold2


class InputEmbedder(nn.Module):
    """Embeds the input features into single and pair representations."""

    config: InputEmbedderConfig

    @nn.compact
    def __call__(self, features: dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Computes the initial single and pair representations.

        Args:
            features: A dictionary of input features. It must contain 'target_seq',
                      a tensor of amino acid indices of shape (N_res,).

        Returns:
            A tuple containing:
            - single_repr: The single representation of shape (N_res, d_single).
            - pair_repr: The pair representation of shape (N_res, N_res, d_pair).
        """
        target_seq = features["target_seq"]
        target_seq_one_hot = jax.nn.one_hot(target_seq, self.config.num_amino_acids)

        # TODO: Implement the single representation projection.
        # 1. Define a flax.linen.Dense layer with `d_single` features.
        # 2. Apply the layer to `target_seq_one_hot`.
        single_repr = nn.Dense(self.config.d_single)(target_seq_one_hot)

        # TODO: Implement the relative positional encoding for the pair representation.
        # See the README for the required steps:
        # 1. Get the number of residues, N_res.
        # 2. Create a 1D array of residue indices: [0, 1, ..., N_res-1].
        # 3. Create a 2D matrix of relative positions by subtracting the indices
        #    from each other (hint: use broadcasting). Shape: (N_res, N_res).
        # 4. Clip the relative positions to the range [-max_relative_pos, max_relative_pos].
        # 5. Shift the values to be non-negative (from 0 to 2*max_relative_pos).
        # 6. Define a flax.linen.Embed layer with `d_pair` features.
        # 7. Apply the embedding layer to the shifted relative positions.
        N_res = target_seq.shape[0]
        residue_indices = jnp.arange(N_res)
        relative_positions = jnp.expand_dims(residue_indices, 1) - jnp.expand_dims(
            residue_indices, 0
        )
        relative_positions = jnp.clip(
            relative_positions,
            -self.config.max_relative_pos,
            self.config.max_relative_pos,
        )
        relative_positions = (relative_positions + self.config.max_relative_pos).astype(
            jnp.int32
        )
        pair_repr = nn.Embed(2 * self.config.max_relative_pos + 1, self.config.d_pair)(
            relative_positions
        )

        return single_repr, pair_repr


# Example usage (for testing your implementation)
if __name__ == "__main__":
    config = InputEmbedderConfig()
    model = InputEmbedder(config)

    key = jax.random.PRNGKey(42)
    n_res = 10
    example_features = {
        "target_seq": jax.random.randint(key, (n_res,), 0, config.num_amino_acids)
    }

    print("--- InputEmbedder Test ---")
    try:
        params = model.init(key, example_features)["params"]
        single_repr, pair_repr = model.apply({"params": params}, example_features)

        print("Input sequence shape:", example_features["target_seq"].shape)
        print("Single representation shape:", single_repr.shape)
        print("Pair representation shape:", pair_repr.shape)
        assert single_repr.shape == (n_res, config.d_single)
        assert pair_repr.shape == (n_res, n_res, config.d_pair)
    except Exception as e:
        print(f"InputEmbedder not implemented yet. Error: {e}")

    print("Test script finished.")
