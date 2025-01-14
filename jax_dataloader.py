import jax
import jax.numpy as jnp

class CustomDataset:
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.num_samples = data.shape[0]
        self.indices = jnp.arange(self.num_samples)

    def __iter__(self):
        self.indices = jax.random.permutation(jax.random.PRNGKey(0), self.indices)
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= self.num_samples:
            raise StopIteration
        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        batch_data = self.data[batch_indices]
        batch_labels = self.labels[batch_indices]
        self.current_index += self.batch_size
        return batch_data, batch_labels