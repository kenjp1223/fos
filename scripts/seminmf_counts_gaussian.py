import os
import pickle

import click
import numpy as np
import wandb
import jax.random as jr

from fos import seminmf_full as seminmf


@click.command()
@click.option('--data_file', required=True, help='Path to .npz with counts, raw_counts, bg_counts.')
@click.option('--results_dir', required=True, help='Directory to save results.')
@click.option('--num_factors', default=10, help='Number of latent factors.')
@click.option('--sparsity_penalty', default=0.01, help='Sparsity penalty weight.')
@click.option('--elastic_net_frac', default=1.0, help='Elastic net mixing fraction.')
@click.option('--num_iters', default=250, help='Number of EM iterations.')
@click.option('--num_coord_ascent_iters', default=1, help='Number of coordinate ascent steps per iteration.')
@click.option('--mean_func', default='softplus', help='Inverse link function.')
@click.option('--gaussian_var', default=None, type=float, help='Observation variance for Gaussian likelihood.')
@click.option('--seed', default=0, help='Random seed.')
@click.option('--wandb_project', default='fos-counts-gaussian', help='Weights & Biases project name.')
def main(data_file, results_dir, num_factors, sparsity_penalty, elastic_net_frac,
         num_iters, num_coord_ascent_iters, mean_func, gaussian_var, seed,
         wandb_project):
    os.makedirs(results_dir, exist_ok=True)

    data = np.load(data_file)
    counts = data['counts']
    raw_counts = data.get('raw_counts')
    bg_counts = data.get('bg_counts')

    if gaussian_var is None:
        gaussian_var = float(np.var(counts))

    key = jr.PRNGKey(seed)
    init_params = seminmf.initialize_random(key, counts, num_factors, mean_func)

    run = wandb.init(
        project=wandb_project,
        config=dict(
            data_file=data_file,
            num_factors=num_factors,
            sparsity_penalty=sparsity_penalty,
            elastic_net_frac=elastic_net_frac,
            num_iters=num_iters,
            num_coord_ascent_iters=num_coord_ascent_iters,
            mean_func=mean_func,
            gaussian_var=gaussian_var,
            seed=seed,
        ),
    )

    params, losses = seminmf.fit_gaussian_seminmf(
        counts,
        init_params,
        mean_func=mean_func,
        sparsity_penalty=sparsity_penalty,
        elastic_net_frac=elastic_net_frac,
        num_iters=num_iters,
        num_coord_ascent_iters=num_coord_ascent_iters,
        gaussian_var=gaussian_var,
    )

    result_file = os.path.join(results_dir, 'params.pkl')
    with open(result_file, 'wb') as f:
        pickle.dump(dict(params=params, losses=np.array(losses),
                         raw_counts=raw_counts, bg_counts=bg_counts,
                         gaussian_var=gaussian_var), f)

    artifact = wandb.Artifact(name='params', type='model')
    artifact.add_file(result_file)
    run.log_artifact(artifact)

    wandb.run.summary['final_loss'] = losses[-1]
    wandb.finish()


if __name__ == '__main__':
    main()
