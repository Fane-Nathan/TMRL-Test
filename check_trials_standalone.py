
import optuna
import os
from pathlib import Path

# Manually set paths based on project structure
STUDY_NAME = "DroQ_SAC_stability_optimization"
# Typically OUTPUT_DIR is in user's home directory under TmrlData
OUTPUT_DIR = Path.home() / "TmrlData" / "hyperparameter_results"

try:
    storage_path = OUTPUT_DIR / "study_database.db"
    print(f"Connecting to database: {storage_path}")
    storage = f"sqlite:///{storage_path}"
    study = optuna.load_study(study_name=STUDY_NAME, storage=storage)

    print(f"Total Trials: {len(study.trials)}")
    for trial in study.trials:
        # Convert trial states to string
        state_str = str(trial.state).split('.')[-1]
        
        # Check if value is None
        val = f"{trial.value:.4f}" if trial.value is not None else "N/A"
        
        # Duration can be calculated if end_date is set, or current if running
        duration = "Running"
        if trial.datetime_complete:
             delta = trial.datetime_complete - trial.datetime_start
             duration = str(delta).split('.')[0] # Remove microseconds
        elif trial.datetime_start:
             import datetime
             delta = datetime.datetime.now() - trial.datetime_start
             duration = f"Running ({str(delta).split('.')[0]})"

        print(f"Trial {trial.number}: State={state_str}, Value={val}, Duration={duration}")
        
        if trial.state == optuna.trial.TrialState.PRUNED:
            # Optuna usually stores *failed* trials with None, but prune reason might be intermediate values
            pass

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error checking study: {e}")
