"""
Generate nominal trajectory using SCvx planner.

This script:
1. Loads configuration files
2. Creates unicycle model and workspace with obstacles
3. Runs SCvx trajectory optimization
4. Validates the resulting trajectory
5. Saves trajectory to disk

Usage:
    python scripts/01_generate_nominal_scvx.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import pickle
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ddfs.utils.config_loader import ExperimentConfig
