import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Tuple
from dataclasses import dataclass


@dataclass
class PairformerConfig:
    """Configuration for the Pairformer."""

    d_single: int = 384
    d_pair: int = 128
    num_blocks: int = 48
    num_heads: int = 16
    d_head: int = 32


class TriangularSelfAttention(nn.Module):
    """
    Computes triangular self-attention, either for starting or ending nodes.
    This is a key component of the Pairformer, allowing pair representations
    to be updated based on information from other pairs.
    """

    config: PairformerConfig
    is_starting_node: bool

    @nn.compact
    def __call__(self, pair_repr: jnp.ndarray) -> jnp.ndarray:
        """
        Performs triangular self-attention on the pair representation.

        Args:
            pair_repr: The pair representation of shape (N_res, N_res, d_pair).

        Returns:
            An updated pair representation of the same shape.
        """
        # TODO: Implement the Triangular Self-Attention logic.
        # This is an axial attention mechanism. For a given pair (i, j), we attend
        # along the "axis" of the third residue k. The goal is to perform attention
        # over rows (for starting nodes) or columns (for ending nodes) of the pair matrix.
        #
        # - For starting node attention, we treat the row index `i` as a batch dimension
        #   and perform attention over the column index `k`.
        # - For ending node attention, we treat the column index `j` as a batch dimension
        #   and perform attention over the row index `k`.
        #
        # This logic can be handled directly by the `einsum` expressions in the steps below,
        # so no explicit `jnp.transpose` of the whole `pair_repr` is needed beforehand.

        # 1. Linearly project `pair_repr` to get queries, keys, and values.
        #    - Use a `nn.Dense` layer for each projection (q, k, v).
        #    - The output dimension for each should be `num_heads * d_head`.

        # 2. Reshape q, k, v to have a separate head dimension.
        #    - The new shape should be (N_res, N_res, num_heads, d_head).

        # 3. Compute attention logits using `jnp.einsum`.
        #    - For starting node attention (row-wise), the einsum should be:
        #      'ijhc,ikhc->ijkh'
        #      i: batch dim (row index), j: query sequence, k: key sequence, h: head, c: channel
        #    - For ending node attention (column-wise), the einsum should be:
        #      'ijhc,kjhc->ikjh'
        #      j: batch dim (column index), i: query sequence, k: key sequence, h: head, c: channel
        #    - Don't forget to scale by `1 / sqrt(d_head)`.

        # 4. Create and apply the triangular attention mask.
        #    - This is the key to "triangular" attention. It ensures that information
        #      only flows from residues `k` that come "before" the current residue
        #      in the relevant axis, preventing information leakage from the future.
        #    - If `is_starting_node` (row-wise), a query at `j` should only attend to keys at `k` where `k <= j`.
        #    - If not `is_starting_node` (column-wise), a query at `i` should only attend to keys at `k` where `k <= i`.
        #    - The mask should be applied to the logits before the softmax step, for example by
        #      adding a large negative number (e.g., -1e9) to the masked-out positions.

        # 5. Apply softmax to get attention weights (over the key sequence dimension `k`).

        # 6. Compute the output by taking the weighted sum of values using `jnp.einsum`.
        #    - For starting node attention, the einsum should be:
        #      'ijkh,ikhc->ijhc'
        #    - For ending node attention, the einsum should be:
        #      'ikjh,kjhc->ijhc'

        # 7. Reshape the output and project it back to `d_pair`.
        return pair_repr + 0  # Placeholder, remove this


class Transition(nn.Module):
    """A feed-forward network applied to a representation."""

    config: PairformerConfig
    widening_factor: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies a two-layer feed-forward network to the input representation.
        """
        # TODO: Implement the Transition layer.
        # 1. Get the input dimension from the last axis of `x`.
        # 2. Define the first `nn.Dense` layer. The number of features should be
        #    `widening_factor * input_dim`.
        # 3. Apply the ReLU activation function (`jax.nn.relu`).
        # 4. Define the second `nn.Dense` layer that projects the representation
        #    back to its original dimension (`input_dim`).
        return x + 0  # Placeholder, remove this


class PairToSingle(nn.Module):
    """Aggregates information from the pair_repr to update the single_repr."""

    config: PairformerConfig

    @nn.compact
    def __call__(self, pair_repr: jnp.ndarray, single_repr: jnp.ndarray) -> jnp.ndarray:
        """
        Computes updates for the single representation from the pair representation.
        """
        # TODO: Implement the pair-to-single communication with attention.
        # 1. Linearly project `single_repr` to queries, keys, and values (q, k, v).
        #    - Use `nn.Dense` for each. Output dim: `num_heads * d_head`.

        # 2. Reshape q, k, v to have a separate head dimension.
        #    - Shape: (N_res, num_heads, d_head)

        # 3. Linearly project `pair_repr` to get the pair bias.
        #    - The output dimension should be `num_heads`.
        #    - Shape: (N_res, N_res, num_heads)

        # 4. Compute attention logits.
        #    - `logits = jnp.einsum('ihc,jhc->ijh', q, k)`
        #    - Add the pair bias.
        #    - Scale the logits by `1 / sqrt(d_head)`.

        # 5. Apply softmax to get weights.

        # 6. Compute the weighted sum of values.
        #    - `output = jnp.einsum('ijh,jhc->ihc', weights, v)`

        # 7. Reshape and project the output back to `d_single`.
        return single_repr + 0  # Placeholder, remove this


class TriangularMultiplicativeUpdate(nn.Module):
    """
    Applies a triangular multiplicative update to the pair representation.
    """

    config: PairformerConfig
    is_outgoing: bool

    @nn.compact
    def __call__(self, pair_repr: jnp.ndarray) -> jnp.ndarray:
        """
        Performs a triangular multiplicative update.
        """
        # TODO: Implement the Triangular Multiplicative Update.
        # 1. Linearly project `pair_repr` into two separate tensors, `a` and `b`.
        #    - Use `nn.Dense` with `d_pair` features for each.
        #    - These will be the inputs to the multiplication.

        # 2. Apply a sigmoid activation to one tensor and use the other as is.
        #    Let's say `g = sigmoid(a)` and `x = b`.

        # 3. Perform the multiplication using `jnp.einsum`.
        #    - If `is_outgoing`: `update = jnp.einsum('ikc,jkc->ijc', g, x)`
        #      This sums over k for pairs (i, k) and (j, k).
        #    - If not `is_outgoing`: `update = jnp.einsum('kic,kjc->ijc', g, x)`
        #      This sums over k for pairs (k, i) and (k, j).
        #    The intuition here is to first check how strongly i and k are related (g_ik).
        #    If they are, then and only then, pay attention to what k has to with j (x_jk).

        # 4. The result of the multiplication should be projected back to `d_pair`
        #    features using another `nn.Dense` layer.
        return pair_repr + 0  # Placeholder, remove this


class PairformerBlock(nn.Module):
    """A single block of the Pairformer stack."""

    config: PairformerConfig

    @nn.compact
    def __call__(
        self, single_repr: jnp.ndarray, pair_repr: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Performs one round of representation refinement.
        """
        # 1. Apply Triangular Multiplicative Updates.
        pair_repr = TriangularMultiplicativeUpdate(
            config=self.config, is_outgoing=True, name="outgoing_update"
        )(pair_repr)
        pair_repr = TriangularMultiplicativeUpdate(
            config=self.config, is_outgoing=False, name="incoming_update"
        )(pair_repr)

        # 2. Apply Triangular Self-Attention.
        pair_repr = TriangularSelfAttention(
            config=self.config, is_starting_node=True, name="start_node_attn"
        )(pair_repr)
        pair_repr = TriangularSelfAttention(
            config=self.config, is_starting_node=False, name="end_node_attn"
        )(pair_repr)

        # 3. Apply a Transition layer to the pair representation.
        pair_repr = Transition(config=self.config, name="pair_transition")(pair_repr)

        # 4. Communication from pair to single representation.
        single_repr = PairToSingle(config=self.config, name="pair_to_single")(
            pair_repr, single_repr
        )

        # 5. Apply a Transition layer to the single representation.
        single_repr = Transition(config=self.config, name="single_transition")(
            single_repr
        )

        return single_repr, pair_repr


class PairformerStack(nn.Module):
    """The main stack of Pairformer blocks."""

    config: PairformerConfig

    @nn.compact
    def __call__(
        self, single_repr: jnp.ndarray, pair_repr: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Iteratively refines the single and pair representations.
        """
        # We use a loop for the blocks. In Flax, to have different parameters for
        # each block, we can create a list of block instances.
        for i in range(self.config.num_blocks):
            single_repr, pair_repr = PairformerBlock(
                config=self.config, name=f"block_{i}"
            )(single_repr, pair_repr)

        return single_repr, pair_repr


# Example usage
if __name__ == "__main__":
    config = PairformerConfig()
    model = PairformerStack(config)

    key = jax.random.PRNGKey(42)
    n_res = 16
    initial_single_repr = jnp.zeros((n_res, config.d_single))
    initial_pair_repr = jnp.zeros((n_res, n_res, config.d_pair))

    print("--- PairformerStack Test ---")
    try:
        # Initialize the model parameters.
        params = model.init(key, initial_single_repr, initial_pair_repr)["params"]

        # Apply the model.
        final_single_repr, final_pair_repr = model.apply(
            {"params": params}, initial_single_repr, initial_pair_repr
        )

        print("Initial single repr shape:", initial_single_repr.shape)
        print("Initial pair repr shape:", initial_pair_repr.shape)
        print("Final single repr shape:", final_single_repr.shape)
        print("Final pair repr shape:", final_pair_repr.shape)

        assert final_single_repr.shape == initial_single_repr.shape
        assert final_pair_repr.shape == initial_pair_repr.shape
        print("Shape check passed!")

    except Exception as e:
        print(
            f"An error occurred. This is expected if the modules are not yet implemented."
        )
        print(f"Error: {e}")

    print("Test script finished.")
