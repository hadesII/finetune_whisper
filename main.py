from pytorch_lightning import Trainer
from data import WhisperDataModule
from model import WhisperModule
import hydra
from config import Config,init_cfg
import os

from pytorch_lightning.loggers import WandbLogger


# os.environ["WANDB_API_KEY"] = "3275fc392e01519d0c4b70bd8494a49baa6be928"
# os.environ["WANDB_MODE"] = "dryrun"

@hydra.main(config_path="config", config_name="config")
def main(cfg:Config):
    dm = WhisperDataModule(cfg.data)
    model = WhisperModule(cfg.model,pretrained_model_name_or_path="openai/whisper-large-v2", lora=True, load_in_8bit=True, device_map="auto")

    # wandb_logger = WandbLogger(project="whisper")

    trainer = Trainer(accelerator="gpu",\
            precision=16,\
            max_epochs=5,\
            # logger=wandb_logger,\
            default_root_dir="temp_large-v2",\
            enable_checkpointing=False)

    trainer.fit(model, dm)


if __name__ == '__main__':


    from hydra._internal.utils import get_args
    cfg_name = get_args().config_name or "config"
    print(cfg_name)

    init_cfg(cfg_name)
    main()

