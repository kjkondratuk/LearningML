import sys
from pathlib import Path

_module_dir = str(Path(__file__).resolve().parent.parent)

for _mod_name in list(sys.modules.keys()):
    if _mod_name in ("exercises", "mini_project", "pipeline", "model",
                      "train", "utils", "mathlib", "ml_library",
                      "transformer", "config", "experiment", "analyze",
                      "data_prep", "reproduce"):
        del sys.modules[_mod_name]

if _module_dir in sys.path:
    sys.path.remove(_module_dir)
sys.path.insert(0, _module_dir)
