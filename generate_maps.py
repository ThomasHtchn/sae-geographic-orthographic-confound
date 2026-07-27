"""
Generate geographic activation maps and Local Moran's I cluster maps for
Sparse Autoencoder (SAE) feature extracted from language model activations.
"""

from tqdm import tqdm
import torch
from topk_sae import TopKSAE
import pandas as pd

def load_model(model_name, input_dim=2048):

    k = int(model_name.split("topk_")[1][:3])
    n = int(model_name.split("_dim")[2][:-3])

    print(f"Loading model for k = {k} and n = {n}")
    model = TopKSAE(d_in=input_dim, d_sae=n, k=k).to("cuda")
    model.load_state_dict(torch.load(model_name))
    model.eval()

    return model

def inspect_neurons(model, df, col_name, mute=True, device="cuda", top_k=20):
    xs = []

    for arr in df[col_name].values:
        x = torch.tensor(
            arr,
            dtype=torch.float32
        ).view(-1)
        xs.append(x)

    x = torch.stack(xs).to(device)
    model.eval()
    _, z = model(x)

    activation_sums = []
    for neuron_id in tqdm(range(0, z.shape[1])):
        
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

from libpysal.weights import KNN
from esda.moran import Moran_Local
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def plot_local_moran_clusters(
    df,
    z,
    neuron_id,
    num_layer,
    save_path,
    k=8,
    p_threshold=0.05,
    figsize=(10, 10),
    title=True,
    legend=True,
    title_prefix=""
):
    """
    Plot Local Moran's I cluster map.
    Parameters
    ----------
    df : DataFrame
        Must contain columns:
            lat
            lon
    z : array-like
        Activation values of hidden embedding of the SAE.
        Length must match shape : len(df), size hidden embedding.
    neuron_id : int
        id of the neuron.

    k : int
        Number of nearest neighbors.
    p_threshold : float
        Significance threshold.
    figsize : tuple
    title, legend : booleans
        display title and / or legend
    """
    scores = np.asarray(z[:, neuron_id].detach().cpu().numpy(), dtype=np.float64)
    work_df = df.copy()
    work_df["activation"] = scores
    gdf = gpd.GeoDataFrame(
        work_df,
        geometry=gpd.points_from_xy(
            work_df["longitude"],
            work_df["latitude"]
        ),
        crs="EPSG:4326"
    )
    gdf = gdf.to_crs("EPSG:3857") # WORLD
    w = KNN.from_dataframe(gdf, k=k)
    w.transform = "R"
    lisa = Moran_Local(gdf["activation"].values, w)
    sig = lisa.p_sim < p_threshold
    cluster = np.full(len(gdf), "Not significant", dtype=object)


    cluster[(lisa.q == 1) & sig] = "High-High"
    cluster[(lisa.q == 2) & sig] = "Low-High"
    cluster[(lisa.q == 3) & sig] = "Low-Low"
    cluster[(lisa.q == 4) & sig] = "High-Low"
    gdf["cluster"] = cluster
    colors = {
        "High-High": "#d7191c",
        "Low-Low": "#2c7bb6",
        "High-Low": "#fdae61",
        "Low-High": "#abd9e9",
        "Not significant": "#d9d9d9"
    }
    fig, ax = plt.subplots(
        figsize=figsize
    )
    alphas = {"High-High": 1, "Low-Low": 1, "High-Low": 1, "Low-High": 1, "Not significant": 0.7}
    for name, color in colors.items():
        subset = gdf[
            gdf["cluster"] == name
        ]
        if len(subset) == 0:
            continue

        subset.plot(
            ax=ax,
            color=color,
            markersize=1,
            label=name,
            alpha=alphas[name]
        )
    
    if legend:
        handles = [
            Line2D([], [], linestyle="", label=r"Local Moran's $I$ clusters:"),
        ] + [
            Patch(facecolor=color, edgecolor="black", linewidth=0.2, label=name)
            for name, color in colors.items()
        ]

        ax.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=len(handles),
                frameon=False,
            handlelength=1.0,      # size of colored square
            handleheight=1.0,
            handletextpad=0.4,
            columnspacing=1.0,
        )

    if title:
        ax.set_title(f"{title_prefix} SAE layer {num_layer}, feature {neuron_id}", loc="left", fontsize=12)
    ax.set_axis_off()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    return gdf, w, lisa



def plot_feature(df, z, neuron_id, save_path, figsize=(10, 10), title=True, num_layer=None, title_prefix="", fix_colorbar=False, min_colorbar=0, max_colorbar=1):
    scores = np.asarray(z[:, neuron_id].detach().cpu().numpy(), dtype=np.float64)

    work_df = df.copy()
    work_df["score"] = scores

    gdf = gpd.GeoDataFrame(
        work_df,
        geometry=gpd.points_from_xy(
            work_df["longitude"],
            work_df["latitude"]
        ),
        crs="EPSG:4326"
    )
    gdf = gdf.to_crs("EPSG:3857")

    fig, ax = plt.subplots(figsize=figsize)

    high = gdf[gdf["score"] > 0]
    low = gdf[gdf["score"] <= 0]    

    cmap = "Blues"

    if len(low):
        low.plot(
            ax=ax,
            color="whitesmoke",
            markersize=1,
            alpha=0.01,
            legend=False,
        )

    if len(high):
        high.plot(
            ax=ax,
            column="score",
            cmap=cmap,
            markersize=1,
            alpha=1,
            legend=False,
        )

    # Create a colorbar matching the score range
    if fix_colorbar:
        norm = plt.Normalize(vmin=min_colorbar, vmax=max_colorbar)
    else:
        norm = plt.Normalize(vmin=gdf["score"].min(), vmax=gdf["score"].max())

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=norm
    )
    sm.set_array([])

    plt.colorbar(
        sm,
        ax=ax,
        #label="Neuron activation",
        shrink=0.3
    )

    if title:
        ax.set_title(f"{title_prefix}SAE layer {num_layer}, feature {neuron_id}", loc="left", fontsize=12)
    ax.set_axis_off()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def print_feature_topcities(df, z, feature_id, top_k=5,):
    scores = z[:, feature_id]
    topk = torch.topk(scores, k=top_k)
    print(f"    === Feature {feature_id} ===")
    for i, (idx, score) in enumerate(zip(topk.indices, topk.values)): 
        idx = idx.item()
        print(f"top {i+1} -> {score.item():.4f} : {df.iloc[idx]['prompt']}; country_code : {df.iloc[idx]['country_code']}")


def main():    
    import argparse

    parser = argparse.ArgumentParser(description="Generate activations and Moran's I local clusters maps of an SAE feature.",
            usage="python generate_maps.py --df_path mlp_act_l17.pkl --model_path topk_256_sae_l17_dim16384.pt  ")
    parser.add_argument("--df_path",             type=str,   required=True,          help="Path to the pickled activation DataFrame.")
    parser.add_argument("--model_path",          type=str,   required=True,          help="Path to the model save state.")
    parser.add_argument("--col",                 type=str,   default="mean_pooling", help="Name of the column in df for activations")
    parser.add_argument("--output_dir",          type=str,   default="./",           help="Dir path to save the map(s).")
    parser.add_argument("--feature_id",          type=int,   required=True,          help="Feature id to plot maps of.")

    args = parser.parse_args()

    model = load_model(args.model_path)
    df = pd.read_pickle(args.df_path)

    layer = int(args.df_path.split('_')[-1][1:-4])
    print("Layer:",layer)

    activations_sums, z = inspect_neurons(model, df, args.col, mute=True)
    
    print(f"Top 10 features (most activations) : {torch.topk(torch.tensor(activations_sums), k=10)[1].tolist()}")

    output_moran = args.output_dir + f"map_moran_feature{args.feature_id}.png"
    output_activation =  args.output_dir + f"map_activations_feature{args.feature_id}.png"

    plot_local_moran_clusters(df, z, args.feature_id, num_layer=layer, save_path=output_moran, p_threshold=0.01)
    plot_feature(df, z, neuron_id=args.feature_id, save_path=output_activation, num_layer=layer)

    print(f"Saved figs at : \n   {output_moran}\n   {output_activation}")

if __name__ == "__main__":
    main()