from pytorch_lightning import Trainer
from data import WhisperDataModule
from model import WhisperModule
import os

from pytorch_lightning.loggers import WandbLogger


# os.environ["WANDB_API_KEY"] = "3275fc392e01519d0c4b70bd8494a49baa6be928"
# os.environ["WANDB_MODE"] = "dryrun"
def main():
    dm = WhisperDataModule()
    model = WhisperModule(pretrained_model_name_or_path="openai/whisper-large-v2", lora=True)

    # wandb_logger = WandbLogger(project="whisper")

    trainer = Trainer(accelerator="gpu",\
            precision=16,\
            max_epochs=5,\
            # logger=wandb_logger,\
            default_root_dir="temp_large-v2")

    trainer.fit(model, dm)


if __name__ == '__main__':

    main()
