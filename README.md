
# Relaqs
Relqs is a Python framework for doing reinforcement learning for quantum control and quantum gate compilation 


## Installation

To install Relaqs:

1. First install poetry with installation instructions [here](https://python-poetry.org/docs/)

2. Then execute the command: 

        poetry install

in the folder with the `pyproject.toml` file

Ray installation:

        pip install -U "ray[rllib]"
        pip install torch
        conda install tensorboardX
        pip install tensorflow-probability

Alternatively:
* Use requirements.txt
* If using a conda environment `conda env create -n <ENVNAME> --file requirements.yml`
