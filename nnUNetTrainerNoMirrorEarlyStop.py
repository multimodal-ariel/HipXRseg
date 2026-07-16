"""Early stopping + NO mirroring (critical for L/R anatomical labels)."""
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerNoMirrorEarlyStop(nnUNetTrainer):
    """
    Hip bones are side-specific (Left/Right Innominate/Femur).
    Default nnU-Net MirrorTransform flips L↔R without swapping labels,
    which teaches the network that left/right are interchangeable.
    We disable mirroring here; use offline hflip+label-swap instead.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 200
        self.early_stop_patience = 30
        self._epochs_since_improvement = 0

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, _mirror = (
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        # Disable all mirroring
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        self.print_to_log_file(
            "Mirroring DISABLED (L/R anatomical labels). "
            "Use offline horizontal-flip + label-swap augmentation instead."
        )
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def on_epoch_end(self):
        prev_best = self._best_ema
        super().on_epoch_end()
        if self._best_ema is not None and (prev_best is None or self._best_ema > prev_best + 1e-6):
            self._epochs_since_improvement = 0
        else:
            self._epochs_since_improvement += 1
        if self._epochs_since_improvement >= self.early_stop_patience:
            self.print_to_log_file(
                f"Early stopping at epoch {self.current_epoch} "
                f"(best EMA={self._best_ema:.4f}, patience={self.early_stop_patience})"
            )
            self.num_epochs = self.current_epoch
