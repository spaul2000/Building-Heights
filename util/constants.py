"""Define constants to be used throughout the repository."""
from pathlib import Path

# Main paths
PROJECT_DIR = Path('/home/Duke/group/minor')


SANDBOX_PATH = PROJECT_DIR / "sandbox"
TB_PATH = PROJECT_DIR / "tb"

EST_DS = {
    'inital_test': (
        # Path to metadata csv
        '/home/Duke/data/dual_bands/metadata.csv',
        # Number of input bands
        5
    )
}

NUM_S1 = 3
NUM_S2 = 3

UNET_DECOD = [256, 128, 64, 32, 16]