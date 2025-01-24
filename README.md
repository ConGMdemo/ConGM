

## Files
```
   .
    ├── dataset_apis                  # Code process datasets.
    │   ├── topology_dist              # Storing the distance of the shortest path (SPD) between vi and vj.
    │   ├── citeseer.py                # processing for citeseer dataset.
    │   ├── cora.py                    # processing for cora dataset. 
    │   ├── dblp.py                    # processing for dblp dataset.
    │   ├── pubmed.py                  # processing for pubmed dataset. 
    │   ├── cornell.py                 # processing for cornell dataset. 
    │   ├── wisconsin.py               # processing for wisconsin dataset. 
    │   ├── texas.py                   # processing for texas dataset.     
    │   └── ...                        # More datasets will be added.
    │
    ├── adversarial.py                # Code for unsupervised adversarial training.
    ├── augmentation.py               # Code for augmentation.
    ├── config.yaml                   # Configurations for our method.
    ├── eval_utils.py                 # The toolkits for evaluation.
    ├── eval.py                       # Code for evaluation.
    ├── global_var.py                 # Code for storing global variable.
    ├── model.py                      # Code for building up model.
    ├── train.py                      # Training process.
    ├── test_runs.py                  # Reproduce the results reported in our paper
    └── ...
```


## Setup
Our experiments are conducted on a platform with NVIDIA GeForce RTX 3090.

Recommand you to set up a Python virtual environment with the required dependencies as follows:
```
conda create -n congm python==3.9
conda activate congm
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.8.0+cu111.html
```
## Usage
** Taking Cora dataset for example, command for training model on Cora dataset**

CUDA_VISIBLE_DEVICES=0 python train.py --dataset=Cora --config=config.yaml

