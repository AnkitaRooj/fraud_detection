# Fraud Detection — Model Trainer

Lightweight Streamlit app to train and compare Isolation Forest (unsupervised) and LightGBM (supervised) models for fraud detection on the PaySim dataset.
![Image](https://github.com/user-attachments/assets/d708af3d-7784-4d59-af33-d62c015c5919)
- Live app entry: [app.py](app.py) — contains the Streamlit UI and orchestration.
- Installable dependencies: [requirements.txt](requirements.txt)
- Raw dataset: [paySim/PaySim_log.csv](paySim/PaySim_log.csv)
- Processed features expected by the app: `notebooks/processeddataset/final_feature_paySim.csv`
- ML artifacts and runs: `mlruns/` and `mlartifacts/` (excluded by [.gitignore](.gitignore))

Quick links to main code symbols in the project:
- Data loader/cache: [`load_and_sample_data`](app.py)
- Trainer class: [`EfficientModelTrainer`](app.py)
  - Isolation Forest training: [`EfficientModelTrainer.train_isolation_forest`](app.py)
  - LightGBM training: [`EfficientModelTrainer.train_lightgbm`](app.py)
  - Evaluation logic: [`EfficientModelTrainer.evaluate_model`](app.py)
- App entrypoint: [`main`](app.py)

Getting started

1. Create and activate a virtual environment (recommended).
2. Install dependencies:
   ```sh

   pip install -r requirements.txt
