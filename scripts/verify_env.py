"""
Environment verification script.

Run after Phase 0 setup to confirm everything is correctly installed.
Usage: python scripts/verify_env.py
"""

import sys

MIN_PYTHON = (3, 10)
if sys.version_info < MIN_PYTHON:
    print(f"[FAIL] Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required. Got {sys.version}")
    sys.exit(1)
print(f"[OK]   Python {sys.version.split()[0]}")

required = [
    ("torch",      "PyTorch"),
    ("torchvision","TorchVision"),
    ("cv2",        "OpenCV"),
    ("numpy",      "NumPy"),
    ("omegaconf",  "OmegaConf"),
    ("rich",       "Rich"),
    ("tqdm",       "tqdm"),
    ("requests",   "Requests"),
    ("PIL",        "Pillow"),
    ("yaml",       "PyYAML"),
    ("pytest",     "pytest"),
]

all_ok = True
for module, name in required:
    try:
        mod = __import__(module)
        print(f"[OK]   {name} {getattr(mod, '__version__', 'unknown')}")
    except ImportError:
        print(f"[FAIL] {name} — not installed")
        all_ok = False

try:
    import torch
    if torch.cuda.is_available():
        print(f"[OK]   CUDA {torch.version.cuda} — {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] CUDA not available — CPU mode")
except Exception as e:
    print(f"[WARN] Could not check CUDA: {e}")

try:
    import surveillance
    print("[OK]   surveillance package importable")
except ImportError:
    print("[FAIL] surveillance package — run: pip install -e .")
    all_ok = False

try:
    from surveillance.core.config import load_config
    cfg = load_config()
    assert cfg.project.name == "home-surveillance"
    print("[OK]   Config system")
except Exception as e:
    print(f"[FAIL] Config system: {e}")
    all_ok = False

try:
    from surveillance.core.logger import get_logger
    get_logger("verify_env").info("Logging system OK")
    print("[OK]   Logging system")
except Exception as e:
    print(f"[FAIL] Logging system: {e}")
    all_ok = False

from surveillance.core.constants import CONFIGS_DIR, WEIGHTS_DIR, DATA_DIR, OUTPUTS_DIR, LOGS_DIR
for name, path in [
    ("configs/", CONFIGS_DIR), ("weights/", WEIGHTS_DIR),
    ("data/",    DATA_DIR),    ("outputs/", OUTPUTS_DIR),
    ("outputs/logs/", LOGS_DIR),
]:
    if path.exists():
        print(f"[OK]   {name}")
    else:
        print(f"[FAIL] Missing directory: {path}")
        all_ok = False

print()
if all_ok:
    print("=" * 50)
    print("  Phase 0 environment ready.")
    print("=" * 50)
else:
    print("=" * 50)
    print("  Failures detected — fix before proceeding.")
    print("=" * 50)
    sys.exit(1)
