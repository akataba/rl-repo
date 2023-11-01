
# Relaqs
Relqs is a Python framework for doing reinforcement learning for quantum control and quantum gate compilation 


## Install relaqs
```
pip install -e .
```

### Required Packages:
The require packages should be installed automatically by pip.  Legacy install instructions are below:





One may install required packages in a number of ways. Either by manually installing, or by executing one of the commands below:

### Direct install:
        pip install -U "ray[rllib]"
        pip install torch
        conda install tensorboardX
        pip install tensorflow-probability
        pip install qutip

### Install with pip:
`pip install -r requirements.txt`

### Install in a conda environment (verified on M1 Mac):
If you are using a conda environment running `conda env create -n <ENVNAME> --file requirements.yml` will install packages using a combination of `pip` and `conda`. This is useful on M1 Macs where some packages aren't available through `pip`.
