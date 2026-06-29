# Home Surveillance System

Research-grade intelligent home surveillance system.

## Pipeline

Camera → YOLOv11s → BoostTrack++ → SCRFD → Face Quality →
EfficientNetV2-B0 → MFR-GAN → AdaFace → FAISS → Firebase/Twilio

## Setup

```bash
conda create -n surveillance python=3.10.14 -y
conda activate surveillance
pip install -e .
python scripts/verify_env.py
```

## Project Structure
configs/        YAML configuration files

src/            Python package (surveillance)

scripts/        CLI entrypoints

weights/        Model weights (gitignored)

data/           Datasets (gitignored)

outputs/        Inference outputs (gitignored)

tests/          Unit and integration tests
