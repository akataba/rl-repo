#!/bin/bash

conda activate relaqs
pip install setuptools wheel
pip install --upgrade /workdir/rl-repo
cd /workdir/rl-repo
