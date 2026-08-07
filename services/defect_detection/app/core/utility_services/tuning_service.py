import gc
import logging

import optuna
import torch

from services.defect_detection.app.core.utility_services.training_service import TrainAutoEncoder
from utils.system_utils import setup_logger


class TuneAutoEncoder:
    """
    Optuna hyperparameter search for the autoencoder, run fully in memory.

    Each trial runs a short training via TrainAutoEncoder with artifact saving
    turned off, so no weights, params.json, TensorBoard log or any other
    persistent file is produced. Only the returned result dict survives:
    the best five parameters and the best validation loss.
    """

    def __init__(self, config: dict):
        """
        Prepare the search space and the fixed training fields for one tuning run.

        The five tuned hyperparameters (learning_rate, latent_space_dimension,
        step_size, gamma, batch_size) are searched within the settings carried by
        config. Every other training field is fixed for the whole run and reused
        unchanged in each trial.

        Args:
            config: Effective tuning config resolved by the API (network/dataset,
                n_trials, epochs_per_trial, the search-space settings for the five
                parameters, and the fixed training fields taken from
                training_config.json).
        """
        setup_logger()

        self.network_type = config.get("network_type")
        self.dataset_type = config.get("dataset_type")
        self.n_trials = config.get("n_trials")
        self.epochs_per_trial = config.get("epochs_per_trial")

        # Search-space bounds for the four range-sampled hyperparameters.
        self.learning_rate_min = config.get("learning_rate_min")
        self.learning_rate_max = config.get("learning_rate_max")
        self.latent_space_dimension_min = config.get("latent_space_dimension_min")
        self.latent_space_dimension_max = config.get("latent_space_dimension_max")
        self.step_size_min = config.get("step_size_min")
        self.step_size_max = config.get("step_size_max")
        self.gamma_min = config.get("gamma_min")
        self.gamma_max = config.get("gamma_max")

        # batch_size is chosen from a discrete list (e.g. [32, 64, 128]), not a
        # continuous range, so it never lands on values like 73 or 117.
        self.batch_size_values = config.get("batch_size_values")

        # Fixed (non-tuned) training fields, taken from training_config.json and
        # reused unchanged for every trial.
        self.base_train_cfg = {
            "network_type": self.network_type,
            "dataset_type": self.dataset_type,
            "validation_split": config.get("validation_split"),
            "grayscale": config.get("grayscale"),
            "early_stopping": config.get("early_stopping"),
            "seed": config.get("seed"),
        }

        self._progress_callback = None
        self._total_steps = self.n_trials * self.epochs_per_trial

    def _report(self, current: int, phase: str) -> None:
        """
        Forward one overall-progress update to the outer progress callback.

        Args:
            current: Completed epochs across all trials so far.
            phase: Short label naming the trial currently running.

        Returns:
            None
        """
        if self._progress_callback is not None:
            self._progress_callback(current, self._total_steps, phase)

    def _epoch_progress(self, trial_number: int):
        """
        Build the per-epoch progress callback handed to one trial's training.

        The trial reports epochs within itself; this maps them onto the overall
        tuning progress (all trials × epochs_per_trial) and labels the phase with
        the current trial number.

        Args:
            trial_number: Zero-based index of the running trial.

        Returns:
            Callable(current, total, phase): One overall PROGRESS update per epoch.
        """
        def report(current: int, total: int, phase: str) -> None:
            overall = trial_number * self.epochs_per_trial + current
            self._report(overall, f"trial {trial_number + 1}/{self.n_trials}")

        return report

    def _on_trial_complete(self, study, frozen_trial) -> None:
        """
        Snap overall progress to the trial boundary after each finished trial.

        A trial may early-stop before epochs_per_trial, so this keeps the bar
        monotonic and lets it reach 100% once the last trial completes.

        Args:
            study: The running Optuna study (unused; required by the callback API).
            frozen_trial: The trial that just finished.

        Returns:
            None
        """
        completed = frozen_trial.number + 1
        self._report(completed * self.epochs_per_trial, f"trial {completed}/{self.n_trials}")

    def _objective(self, trial) -> float:
        """
        Run one trial: sample the five hyperparameters and train briefly.

        The trainer, its optimizer and dataloaders are released in a finally
        block so their (GPU) memory is freed before the next trial starts.

        Args:
            trial: The Optuna trial that samples the hyperparameters.

        Returns:
            float: The best validation loss reached in this trial (minimized).
        """
        trial_cfg = {
            **self.base_train_cfg,
            "epochs": self.epochs_per_trial,
            "learning_rate": trial.suggest_float(
                "learning_rate", self.learning_rate_min, self.learning_rate_max, log=True
            ),
            "latent_space_dimension": trial.suggest_int(
                "latent_space_dimension",
                self.latent_space_dimension_min,
                self.latent_space_dimension_max,
            ),
            "step_size": trial.suggest_int(
                "step_size", self.step_size_min, self.step_size_max
            ),
            "gamma": trial.suggest_float(
                "gamma", self.gamma_min, self.gamma_max
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size", self.batch_size_values
            ),
            # step_size and gamma only take effect when the LR scheduler runs.
            "decrease_learning_rate": True,
            # A tuning trial writes nothing to disk: no weights, params.json,
            # TensorBoard log or visualizations.
            "vis_during_training": False,
            "save_artifacts": False,
        }

        trainer = TrainAutoEncoder(trial_cfg)
        try:
            result = trainer.fit(
                progress_callback=self._epoch_progress(trial.number)
            )
            best_valid_loss = result["best_valid_loss"]
        finally:
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return best_valid_loss

    def run(self, progress_callback=None) -> dict:
        """
        Run the whole Optuna study in memory and return the best parameters.

        Args:
            progress_callback: Optional callable(current, total, phase) forwarded
                as overall tuning progress (epochs across all trials). None
                disables progress reporting.

        Returns:
            dict: {"status", "best_params", "best_valid_loss"} — JSON-compatible,
            with best_params holding exactly the five tuned hyperparameters.
        """
        self._progress_callback = progress_callback

        logging.info(
            f"Starting Optuna tuning: {self.n_trials} trials, "
            f"{self.epochs_per_trial} epochs each, network {self.network_type}, "
            f"dataset {self.dataset_type}"
        )

        self._report(0, f"trial 1/{self.n_trials}")

        study = optuna.create_study(direction="minimize")
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            callbacks=[self._on_trial_complete],
        )

        best_params = {
            "learning_rate": study.best_params["learning_rate"],
            "latent_space_dimension": study.best_params["latent_space_dimension"],
            "step_size": study.best_params["step_size"],
            "gamma": study.best_params["gamma"],
            "batch_size": study.best_params["batch_size"],
        }

        logging.info(
            f"Tuning done. Best valid loss {study.best_value:.5f}, params {best_params}"
        )

        return {
            "status": "DONE",
            "best_params": best_params,
            "best_valid_loss": float(study.best_value),
        }