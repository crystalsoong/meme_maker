# trainers/hp_optuna_sweep.py
"""
Run a small Optuna hyperparameter sweep to tune lr and batch size.
This script calls the existing training script with different args via subprocess.
Requires optuna.
"""
import optuna
import subprocess
import json
from pathlib import Path

TRAIN_SCRIPT = "python trainers/train_mememaker.py"

def objective(trial):
    lr = trial.suggest_loguniform("learning_rate", 1e-6, 1e-3)
    batch = trial.suggest_categorical("batch_size", [4, 8, 16])
    epochs = 2
    cmd = f"{TRAIN_SCRIPT} --train_manifest data/processed/merged_manifest_train_small.json --val_manifest data/processed/merged_manifest_val_small.json --per_device_train_batch_size {batch} --per_device_eval_batch_size {max(1,batch//2)} --epochs {epochs} --learning_rate {lr} --fp16"
    print("Running:", cmd)
    res = subprocess.run(cmd, shell=True)
    # For simplicity, we expect trainer writes eval_loss to outputs/mememaker/trainer_state.json
    st = Path("outputs/mememaker/trainer_state.json")
    if st.exists():
        try:
            tr = json.load(st.open())
            # parse last eval loss if present
            metrics = tr.get("best_metric", None)
            if metrics is not None:
                return float(metrics)
        except Exception:
            pass
    # fallback: when trainer writes nothing, return a large loss to penalize
    return 1e6

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=6)
    print("Best params:", study.best_params)
