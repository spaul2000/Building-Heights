"""Define constants to be used throughout the repository."""
from pathlib import Path

# Main paths
PROJECT_DIR = Path('/home/Duke/group/main')


SANDBOX_PATH = PROJECT_DIR / "sandbox"
TB_PATH = PROJECT_DIR / "tb"

EST_DS = {
    'inital_test': (
        # Path to metadata csv
        '/home/Duke/data/full_tx_2000_mean_vv_vh_rgb/metadata.csv',
        # Number of input bands
        5
    )
}

UNET_DECOD = [256, 128, 64, 32, 16]