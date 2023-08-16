#!/bin/bash

if [ -d "/workdir/rl-repo/docker/venv" ]; then
    source /workdir/rl-repo/docker/venv/bin/activate
else
    python3 -m venv /workdir/rl-repo/docker/venv
    source /workdir/rl-repo/docker/venv/bin/activate
    pip install --upgrade pip
    pip install setuptools wheel
    pip install /workdir/rl-repo
fi
cd /workdir/rl-repo
