"""
Evaluate a trained Top-K Sparse Autoencoder (SAE) on language model
activations.

The script computes feature activation statistics, with a focus on the two
primary SAE evaluation metrics:
    - Dead features: features that never activate.
    - Alive feature density: the activation density of non-dead features.

It also reports summary activation statistics and the most highly activated
features.
"""

from topk_sae import TopKSAE
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def inspect_neurons(model, df, hidden_dim, mute=True, device="cuda", top_k=20):
    xs = []
    for arr in df["mean_pooling"].values:
        x = torch.tensor(
            arr,
            dtype=torch.float32
        ).view(-1)
        xs.append(x)

    x = torch.stack(xs).to(device)
    model.eval()
    _, z = model(x)


    activation_sums = []
    for neuron_id in tqdm(range(0, hidden_dim)):
        
        scores = z[:, neuron_id]
        topk = torch.topk(scores, k=top_k)

        activation_sums.append(sum(topk.values))
        if not mute:
            if(sum(topk.values) > 10):
                print(f"    === Neuron {neuron_id} ===")

                print(f"total topk = {sum(topk.values):.4f}")
                for idx, score in zip(topk.indices, topk.values):
                    
                    idx = idx.item()
                    print(f"{score.item():.4f} -> {df.iloc[idx]['prompt']}")
        
    return activation_sums, z

def evaluate_sae(model_name, k, N, df, input_dim=2048, threshold=0.0):

    n = N*input_dim
    model = TopKSAE(d_in=input_dim, d_sae=n, k=k).to("cuda")
    model.load_state_dict(torch.load(model_name))
    model.eval()

    df = pd.read_pickle(df)

    activations_sums, z = inspect_neurons(model, df, hidden_dim=n, mute=True)

    with torch.no_grad():
        active = (z > threshold)
        densities = active.float().mean(dim=0).cpu().numpy()

    nonzero_densities = densities[densities > 0]

    if len(nonzero_densities) == 0:
        raise ValueError("All features have zero density.")

    print(f"=== Evaluating SAE with Top k: {k}, n: {n} ===")
    print(f"Features:               {len(densities)}")
    print(f"Dead features:          {((densities == 0).sum() / len(densities)):.2f} ({(densities == 0).sum()})")
    print(f"Alive features:         {n - (densities == 0).sum()}")
    print(f"Mean density:           {densities.mean():.6f}")
    print(f"Median density:         {np.median(densities):.6f}")
    # for q in [0.5, 0.9, 0.95, 0.99]:
    #     print(f"Q. {q} = {np.quantile(densities, q):.3f}")
    print(f"Nonzero mean density:   {nonzero_densities.mean():.6f}")
    print(f"Nonzero median density: {np.median(nonzero_densities):.6f}")

    print(f"Top 10 neurons: {torch.topk(torch.tensor(activations_sums), k=10)[1].tolist()}")
    return z

def plot_feature_density_histogram(
    z,
    threshold=0.0,
    bins=100,
    use_log_densities=False,
    use_non_zero=True
):
    """
    Plot histogram of SAE feature densities.

    Parameters
    ----------
    z : torch.Tensor
        Shape [num_samples, hidden_dim]
        Latent activations.

    threshold : float
        Activation threshold for considering a feature active.
        Use 0 for ReLU SAEs, e.g. 1e-3 for noisy activations.

    bins : int
        Number of histogram bins.
    """

    with torch.no_grad():

        # active[sample, feature]
        active = (z > threshold)

        # density per feature
        densities = active.float().mean(dim=0).cpu().numpy()

    # remove dead features to avoid log10(0)
    nonzero_densities = densities[densities > 0]

    if len(nonzero_densities) == 0:
        raise ValueError("All features have zero density.")

    log_densities = np.log10(nonzero_densities)

    if use_log_densities:
        plot_densities = log_densities
        x_label = "log10(feature density)"
    elif use_non_zero:
        plot_densities = nonzero_densities
        x_label = "feature density (nonzero)"
    else:
        plot_densities = densities
        x_label = "feature density"

    print(f"Features: {len(densities)}")
    print(f"Dead features: {(densities == 0).sum()} -> (%) {((densities == 0).sum() / len(densities)):.6f}")
    print(f"Mean density: {densities.mean():.6f}")
    print(f"Median density: {np.median(densities):.6f}")

    
    for q in [0.5, 0.9, 0.95, 0.99]:
        print(f"Q. {q} = {np.quantile(densities, q):.3f}")

    print(f"Nonzero mean density: {nonzero_densities.mean():.6f}")
    print(f"Nonzero median density: {np.median(nonzero_densities):.6f}")

    plt.figure(figsize=(8, 5))

    plt.hist(
        plot_densities,
        bins=bins,
    )

    plt.xlabel(x_label)
    plt.ylabel("Count")
    plt.title("Distribution of SAE Feature Densities")

    plt.grid(alpha=0.3)

    plt.savefig("feature_density_histogram.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()



def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train a TopK Sparse Autoencoder on MLP activations.",
                                     usage="python eval_sae.py --model_path <model path> --df_path <activation df path> --k <SAE topK> --N <hidden dim factor>")
    parser.add_argument("--model_path",     type=str,   required=True,        help="Path to SAE model save point.")
    parser.add_argument("--df_path",        type=str,   required=True,        help="Path to the activations dataset")
    parser.add_argument("--k",              type=int,   required=True,        help="Top k used to train the SAE")
    parser.add_argument("--N",              type=int,   required=True,        help="Size (Factor) of the hidden dimension of the SAE")
    parser.add_argument("--plot_feat",      type=bool,  choices=[True, False],  default=True,   help="Plot the feature density historgram")

    args = parser.parse_args()
    
    z = evaluate_sae(args.model_path, args.k, args.N, args.df_path)
    if args.plot_feat:
        plot_feature_density_histogram(z)

if __name__ == "__main__":
    main()