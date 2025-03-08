import sys

from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm

from config import Config
from model import LightBert
from loader import load_data
import lightning as L


mdl = LightBert(Config)


class CustomProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for validation."""
        # The train progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.trainer.state.fn != "validate"
        return Tqdm(
            desc=self.train_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=False,
            file=sys.stdout,
            smoothing=0,
            bar_format=self.BAR_FORMAT,
        )

    def init_train_tqdm(self) -> Tqdm:
        return Tqdm(
            desc=self.train_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=False,
            file=sys.stdout,
            smoothing=0,
            bar_format=self.BAR_FORMAT,
        )


progress_bar = CustomProgressBar(
    leave=True,
)

trainer = L.Trainer(
    max_epochs=60,
    callbacks=[progress_bar],
)

trainer.fit(
    model=mdl,
    train_dataloaders=load_data(Config["train_data_path"], Config),
    val_dataloaders=load_data(Config["valid_data_path"], Config)
)