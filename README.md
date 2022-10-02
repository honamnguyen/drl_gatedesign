# Deep Reinforcement Learning for Gate Design
## Installation guide
- (Highly recommended) Create a `conda` environment `envname`: 
```
conda create -n envname python==3.9
conda activate envname
```
- Run the followwing in to install the required packages
```
conda install -c weinbe58 quspin
pip install -r requirements.txt
pip install -e .
```
- Attach the conda environment to jupyter via `python -m ipykernel install --user --name=envname`
