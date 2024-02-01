
# Relaqs
Relaqs is a Python framework for doing reinforcement learning for quantum control and quantum gate compilation 


## Install relaqs
```
pip install .
```


### Install on an Arm (e.g. M1/M2) Mac
Some of the required packages only support x86 and therefore can't be installed with the default arm Python on Arm Macs.  There are two popular ways to get an x86 version of Python running on Arm Macs:
- Install conda and create a conda env, which should default to x86:  `conda create -n <ENVNAME>`.  You can then install the repository as shown above.  If that doesn't work, you can force it to be x86 using: `CONDA_SUBDIR=osx-64 conda create -n <ENVNAME>`.
- Launch a Rosetta terminal, intall x86 homebrew, and install python3 with the x86 homebrew.  If you create a virtualenv: `python3 -m venv </path/to/venv>` with the x86 python, using that virtualvenv should run x86 python without having to create a Rosetta terminal first.

A helpful StackOverflow post on this topic:
https://stackoverflow.com/questions/71691598/how-to-run-python-as-x86-with-rosetta2-on-arm-macos-machine

