# This is necessary to allow pytest to correctly find the scripts folder
# Potentially less hacky option: make the scripts folder a module to be installed
import pytest
import sys
import os
sys.path.insert(1, os.path.abspath('./scripts/'))
