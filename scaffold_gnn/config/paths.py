from pathlib import Path
import os
DATA_ROOT = Path(os.environ.get("SCAFFOLD_DATA_ROOT", "./data"))
SCRATCH_ROOT = Path(os.environ.get("SCAFFOLD_CACHE_ROOT", "./results/cache"))
