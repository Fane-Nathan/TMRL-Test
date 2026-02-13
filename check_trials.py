
import optuna
from tmrl.hyperparameter_sweep_optuna import STUDY_NAME, OUTPUT_DIR

try:
    storage_path = OUTPUT_DIR / "study_database.db"
    storage = f"sqlite:///{storage_path}"
    study = optuna.load_study(study_name=STUDY_NAME, storage=storage)

    print(f"Total Trials: {len(study.trials)}")
    for trial in study.trials:
        print(f"Trial {trial.number}: State={trial.state}, Value={trial.value if trial.value is not None else 'N/A'}, Duration={trial.duration}")
except Exception as e:
    print(f"Error reading study: {e}")
