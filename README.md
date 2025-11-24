# Protein-Glycan Interaction Prediction

## How to use
- run `conda env create -n glycanml python=3.10`
- `conda activate glycanml`
- `pip install -r requirements.txt`
- pretrain with `python pretrain.py`
- train and evaluate model with `python train.py --run-id example_run_id`

## Architecture
- pretrained monosaccharide-level Graphormer for a glycan encoder and ESM-C for the protein encoder
- Cross Attention on the glycan and protein tokens
- mean-pool embeddings
- MLP block
- regression output to predict binding affinity z-score