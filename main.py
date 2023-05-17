from pytorch_lightning import Trainer
from data import WhisperDataModule
from model import WhisperModule

def main():
    dm = WhisperDataModule()
    model = WhisperModule(pretrained_model_name_or_path="openai/whisper-large-v2", lora=True)

    trainer = Trainer(accelerator="gpu", devices=8,\
            max_epochs=5,\
            default_root_dir="temp_large-v2")

    trainer.fit(model, dm)


if __name__ == '__main__':

    main()
