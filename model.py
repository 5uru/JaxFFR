import jax.numpy as jnp
import flax.linen as nn

class ShapeChecker:
    def __call__(self, tensor, expected_shape):
        assert tensor.shape == expected_shape, f"Expected {expected_shape}, got {tensor.shape}"

class Encoder(nn.Module):
    input_vocab_size: int
    embedding_dim: int
    enc_units: int

    @nn.compact
    def __call__(self, tokens,):

        shape_checker = ShapeChecker()

        embedding = nn.Embed(num_embeddings=self.input_vocab_size, features=self.embedding_dim)
        gru = nn.RNN(nn.GRUCell(self.enc_units),return_carry=True)
        batch_size, seq_length = tokens.shape
        shape_checker(tokens, (batch_size, seq_length))

        vectors = embedding(tokens)
        shape_checker(vectors, (batch_size, seq_length, self.embedding_dim))
        _, enc_output = gru(vectors, seq_lengths=seq_length)
        shape_checker(enc_output, (batch_size, seq_length, self.enc_units))
        return enc_output