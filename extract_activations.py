"""
Extract MLP activations from a Hugging Face causal language model for
GeoNames-based prompts.

The script loads prompts from a pickle dataset, captures MLP outputs at
selected transformer layers using forward hooks, computes mean-pooled
representations, and saves activations with geographic metadata.
Use --col_name asciiname to save activations of ASCII city names.

Uses CUDA when available, otherwise falls back to CPU.

Example:
    python extract_activations.py --input_path prompts_world.pkl --layers "10,17,31" --output_path activations/
"""

import argparse
import ast

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os

def parse_layers(layer_str):
    value = ast.literal_eval(layer_str)

    if isinstance(value, int):
        return [value]

    return list(value)


def extract_mlp_activations(
    model,
    tokenizer,
    prompt,
    layers,
    device,
):
    """
    Extract AFTER-MLP activations for selected layers.

    Returns:
        dict[layer_idx] -> tensor(seq_len, hidden_size)
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)

    activations = {}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, module_input, module_output):

            acts = module_output.detach()[0].float().cpu()

            activations[layer_idx] = acts

        return hook

    for layer_idx in layers:
        h = model.model.layers[layer_idx-1].mlp.register_forward_hook(
            make_hook(layer_idx)
        )
        hooks.append(h)

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    return activations


def main():
    parser = argparse.ArgumentParser(description="Extract MLP activations from a Hugging Face causal LM.",
                usage="python extract_activations.py --input_path prompts_world.pkl --layers \"10,17,31\" --output_path activations/")

    parser.add_argument("--input_path",   type=str, required=True,                       help="Path to input prompts pickle file.")
    parser.add_argument("--model_name",  type=str, default="HuggingFaceTB/SmolLM3-3B",  help="Hugging Face model name.")
    parser.add_argument("--output_path", type=str, default="./",                       help="Output path to save activation dataset as pickle file.")
    parser.add_argument("--layers",      type=str, required=True,                       help='Layers list, e.g. "10,15,28" or just one layer, e.g. "10"')
    parser.add_argument("--batch_size",  type=int, default=1)
    parser.add_argument("--col_name",    type=str, choices=["name", "asciiname"],       default="name", help="Name of the column to use for the prompts : name or asciiname")

    args = parser.parse_args()

    layers = parse_layers(args.layers)

    print(f"Loading dataframe: {args.input_path}")
    df = pd.read_pickle(args.input_path)

    required_cols = ["geoname_id", args.col_name, "lat", "lon", "country_code"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    print(f"Loading model: {args.model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16
    ).to(device)
    model.eval()


    rows = {layer: [] for layer in layers}
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        geonameid = row["geoname_id"]
        prompt = row[args.col_name]
        lat = row["lat"]
        lon = row["lon"]
        country_code = row["country_code"]
        try:
            acts_dict = extract_mlp_activations(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                layers=layers,
                device=device,
            )
            for layer_idx, acts in acts_dict.items():
                # acts: [seq_len, hidden_size]
                mean_pool = acts.mean(dim=0).numpy()
                rows[layer_idx].append({
                    "geonameid": geonameid,
                    "prompt": prompt,
                    "layer": layer_idx,
                    "activations": acts.numpy(),
                    "mean_pooling": mean_pool,
                    "latitude": lat,
                    "longitude": lon,
                    "country_code": country_code
                })
        except Exception as e:
            print(f"Error processing geonameid={geonameid}: {e}")

    output_dir = os.path.dirname(args.output_path)
    for layer_idx, layer_rows in rows.items():
        out_df = pd.DataFrame(layer_rows)
        out_path = os.path.join(output_dir, f"mlp_act_l{layer_idx}.pkl")
        print(f"Saving layer {layer_idx} to: {out_path}")
        out_df.to_pickle(out_path)

    print("Done.")


if __name__ == "__main__":
    main()