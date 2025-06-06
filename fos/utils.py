import dataclasses
import jax
import jax.numpy as jnp
import warnings

from jax import tree_map
from jax.tree_util import tree_reduce
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
warnings.filterwarnings("ignore")


# Helper function to make a dataclass a JAX PyTree
def register_pytree_node_dataclass(cls):
  _flatten = lambda obj: jax.tree_flatten(dataclasses.asdict(obj))
  _unflatten = lambda d, children: cls(**d.unflatten(children))
  jax.tree_util.register_pytree_node(cls, _flatten, _unflatten)
  return cls


def convex_combo(pytree1, pytree2, stepsize):
    f = lambda x, y: (1 - stepsize) * x + stepsize * y
    return tree_map(f, pytree1, pytree2)


def tree_add(pytree1, pytree2, scale=1.0):
    return tree_map(lambda x, y: x + scale * y, pytree1, pytree2)


def tree_dot(pytree1, pytree2):
    return tree_reduce(jnp.add,
                       tree_map(lambda x, y: jnp.sum(x * y), pytree1, pytree2),
                       0.0)


def scale_counts(counts, initial_params=None):
    """Scale counts data using z-score normalization, preserving zeros.
    
    Args:
        counts: Array of count data
        initial_params: Optional SemiNMFParams object to scale along with the data
        
    Returns:
        If initial_params is None:
            tuple: (scaled_counts, scaling_params)
        If initial_params is provided:
            tuple: (scaled_counts, scaled_params, scaling_params)
    """
    # Only compute statistics on non-zero counts
    non_zero_mask = counts > 0
    data_mean = jnp.mean(counts[non_zero_mask])
    data_std = jnp.std(counts[non_zero_mask])
    
    # Scale the data, keeping zeros as zeros
    scaled_counts = jnp.where(non_zero_mask,
                            (counts - data_mean) / data_std,
                            0.0)
    
    scaling_params = {
        'mean': data_mean,
        'std': data_std
    }
    
    if initial_params is not None:
        # Scale the parameters to match the data scaling
        scaled_params = dataclasses.replace(
            initial_params,
            factors=initial_params.factors / data_std,
            count_loadings=initial_params.count_loadings * data_std,
            count_row_effects=(initial_params.count_row_effects - data_mean) / data_std,
            count_col_effects=initial_params.count_col_effects / data_std
        )
        return scaled_counts, scaled_params, scaling_params
    
    return scaled_counts, scaling_params


def unscale_counts(scaled_counts, scaling_params):
    """Reverse the scaling of counts data.
    
    Args:
        scaled_counts: Scaled count data
        scaling_params: Dict containing 'mean' and 'std' used for scaling
        
    Returns:
        Array of unscaled counts
    """
    # Only compute statistics on non-zero counts
    non_zero_mask = scaled_counts > 0
    return jnp.where(non_zero_mask,
                    scaled_counts * scaling_params['std'] + scaling_params['mean'],
                    0.0)
