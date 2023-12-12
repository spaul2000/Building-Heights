"""Define constants to be used throughout the repository."""
from pathlib import Path
import os 
# Main paths
PROJECT_DIR = Path(os.path.expanduser('~/group/main'))


SANDBOX_PATH = PROJECT_DIR / "sandbox"
TB_PATH = PROJECT_DIR / "tb"

EST_DS = {
    'inital_test': (
        # Path to metadata csv
        '~/data/dual_bands/metadata.csv',
        # Number of input bands
        6,
    )
}
NUM_S1 = 3
NUM_S2 = 3

UNET_DECOD = [256, 128, 64, 32, 16]