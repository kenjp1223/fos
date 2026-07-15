import os
import dataclasses
import pickle

import click
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import wandb

from fos import seminmf_full as seminmf


@click.command()
@click.option('--data_file', required=True, help='Path to .npz with raw_counts, bg_counts, and optional mask.')
@click.option('--results_dir', required=True, help='Directory to save results.')
@click.option('--mask_key', default=None, help='Optional key in NPZ file for training mask.')
@click.option('--max_num_factors', default=25, help='Largest number of factors to try.')
@click.option('--num_iters', default=250, help='Number of EM iterations.')
@click.option('--num_coord_ascent_iters', default=1, help='Coordinate-ascent steps per iteration.')
@click.option('--elastic_net_frac', default=1.0, help='Elastic net mixing fraction.')
@click.option('--wandb_project', default='fos-counts-poisson-search', help='Weights & Biases project name.')
@click.option('--seed', default=0, help='Random seed.')
def main(data_file, results_dir, mask_key, max_num_factors, num_iters,
         num_coord_ascent_iters, elastic_net_frac,
         wandb_project, seed):
    os.makedirs(results_dir, exist_ok=True)

    data = np.load(data_file)
    raw_counts = data['raw_counts']
    bg_counts = data['bg_counts']
    counts = data.get('counts')
    if counts is None:
        counts = raw_counts - bg_counts
    mask = data[mask_key] if mask_key is not None and mask_key in data else None

    if bg_counts.shape != counts.shape:
        raise ValueError(
            f"bg_counts shape {bg_counts.shape} does not match counts shape {counts.shape}"
        )

    key = jr.PRNGKey(seed)
    mean_func = 'softplus'
    full_initial_params = seminmf.initialize_random(key, counts, max_num_factors, mean_func)

    all_sparsity_penalties = jnp.array([1e-4, 1e-3, 1e-2, 1e-1])
    all_num_factors = jnp.arange(8, max_num_factors + 1, 2)
    all_heldout_loglikes = -jnp.inf * jnp.ones((len(all_sparsity_penalties), len(all_num_factors)))

    for i, sparsity_penalty in enumerate(all_sparsity_penalties):
        for j, num_factors in enumerate(all_num_factors):
            run = wandb.init(
                dir=results_dir,
                project=wandb_project,
                job_type='train',
                config=dict(
                    sparsity_penalty=float(sparsity_penalty),
                    num_factors=int(num_factors),
                    elastic_net_frac=elastic_net_frac,
                    max_num_iters=num_iters,
                    num_coord_ascent_iters=num_coord_ascent_iters,
                    model='poisson_bg_offset',
                    initialization='random',
                    data_file=data_file,
                    mask_key=mask_key,
                ),
            )

            print(f'Fitting model with {float(sparsity_penalty)} sparsity and {int(num_factors)} factors')

            initial_params = dataclasses.replace(
                full_initial_params,
                factors=full_initial_params.factors[:num_factors],
                loadings=full_initial_params.loadings[:, :num_factors],
            )

            params, losses = seminmf.fit_poisson_seminmf(
                counts,
                initial_params,
                mask=mask,
                mean_func=mean_func,
                sparsity_penalty=float(sparsity_penalty),
                elastic_net_frac=elastic_net_frac,
                num_iters=num_iters,
                num_coord_ascent_iters=num_coord_ascent_iters,
                bg_counts=bg_counts,
            )

            heldout_mask = ~mask if mask is not None else jnp.ones_like(counts, dtype=bool)
            heldout_ll = -seminmf.smooth_loss(
                params,
                counts,
                heldout_mask,
                mean_func,
                distribution='poisson',
                bg_counts=bg_counts,
            )
            all_heldout_loglikes = all_heldout_loglikes.at[i, j].set(heldout_ll)

            result_file = os.path.join(results_dir, f'params_{i}_{j}.pkl')
            with open(result_file, 'wb') as f:
                pickle.dump(dict(params=params,
                                 losses=np.array(losses),
                                 sparsity_penalty=float(sparsity_penalty),
                                 num_factors=int(num_factors),
                                 heldout_loglike=float(heldout_ll)), f)

            run.summary['final_loss'] = float(losses[-1])
            run.summary['heldout_loglike'] = float(heldout_ll)
            artifact = wandb.Artifact(name=f'params_{i}_{j}', type='model')
            artifact.add_file(result_file)
            run.log_artifact(artifact)
            wandb.finish()

    np.savez(
        os.path.join(results_dir, 'heldout_loglikes.npz'),
        heldout_loglikes=np.array(all_heldout_loglikes),
        sparsity_penalties=np.array(all_sparsity_penalties),
        num_factors=np.array(all_num_factors),
    )


if __name__ == '__main__':
    main()
