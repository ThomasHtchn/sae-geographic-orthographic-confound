# sae-geographic-orthographic-confound

## References
The following repository aim is to reproduce the work of De Sabbata et al. (2026), who used Sparse Autoencoders (SAE) to better understand how geographical information is represented by Large Language Models (LLMs) by discovering spatially coherent features.<br> 
Our work extends their experiments to a global scale while investigating the role and influence of orthographic and lexical patterns of place names in the creation of the SAE features.

### Original work

De Sabbata et al. (2026). *Geospatial Mechanistic Interpretability of Large Language Models.*<br>
https://arxiv.org/abs/2505.03368

### This work

Hitchon, T., Decoupes, R., & Guille, A. *Can Sparse Autoencoder-Based Interpretability Demonstrate Geographic Knowledge Beyond Orthographic Features?*

[Read the paper (PDF)](Revisiting_Geospatial_Mechanistic_Interpretability_in_Large_Language_Models__A_Generalization_Study.pdf)

## Code

### Set-up the conda environement
```
conda env create env.yaml -n xai_env || conda env update -f env.yaml
conda activate xai_env
```

### Creating prompt from GeoNames and extracting model's activations
+ Download and extract GeoNames data : 
```
wget -nc https://download.geonames.org/export/dump/allCountries.zip
python -c "import zipfile; zipfile.ZipFile('allCountries.zip').extractall('.')"
```
+ Run the cities (feature class P) extraction for population threshold (`min_pop`)
```
python create_prompts.py 
    --file allCountries.txt
    --min_pop 10000
    --output_path prompts.pkl
```
+ Extract activations from `HuggingFaceTB/SmolLM3-3B` Hugging face model for `layers`. Set `col_name` argument to `name` or `ascii` for corruption.
```
python extract_activations.py 
    --input_path prompts.pkl
    --layers 10,17,31
    --output_path ./
    --col_name [name, asciiname]
```

### Training, evaluating TopK Sparse Autoencoders
+ Train TopK SAE, set k with `top_k` and d_sae with `hidden_dim` (d_sae = d_in * `hidden_dim`) 
```
python topk_sae.py
    --intput_path mlp_act_l17.pkl
    --hidden_dim 8
    --top_k 256
    --epochs 500
```
+ Evaluation recap. To save feature density histogram, set `plot_feat` argument to `True`
```
python eval_sae.py
    --model_path topk_256_sae_l17_dim8.pt
    --df_path mlp_act_l17.pkl
    --k 256
    --N 8
    --plot_feat True
```

### Ploting feature maps
```
python generate_maps.py
    --model_path topk_256_sae_l17_dim8.pt
    --df_path mlp_act_l17.pkl
    --feature_id 924
```
<img src="paper_figures/map_moran_l17_feature924.png">
<img src="paper_figures/map_activation_l17_feature924.png">

### Additionnal experience : predictions on cities n-grams

```
python geonames_cities.py
```