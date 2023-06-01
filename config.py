from dataclasses import dataclass,field
from typing import Optional
from hydra.core.config_store import ConfigStore

from omegaconf import II

@dataclass
class Model:
    model_name: str = field(default="large-v2")
    learning_rate: float = field(default=0.00001)
    weight_decay: float = field(default=0.01)
    adam_epsilon: float = field(default=1e-8)
    warmup_steps: int = field(default=1000)
    batch_size: int = II("data.batch_size")
    gradient_accumulation_steps: int = II("trainer.gradient_accumulation_steps")
    num_train_epochs: int = II("trainer.max_epoch")
    frozen_encoder: bool = field(default=True)
    frozen_decoder: bool = field(default=False)
    checkpoint_root: str = II("trainer.root_dir")


@dataclass
class Data:
    data_path: str = field(default="/data1/dtm/corpus/dialect")
    batch_size: int = field(default=2)
    num_workers: int = field(default=5)
    sample_rate: int = field(default=16000)
    lang_pairs: tuple[str] = field(default=("cantonese",))
    prompt: Optional[dict] = field(default=None)

@dataclass
class Trainer:
    gradient_accumulation_steps: int = field(default=2)
    max_epoch: int = field(default=10)
    root_dir: str = field(default="/data1/outputs/lora")

@dataclass
class Config:
    model: Model = Model()
    data: Data = Data()
    trainer: Trainer = Trainer()

def init_cfg(cfg_name:Config):

    cs = ConfigStore.instance()
    cs.store(f"{cfg_name}",node=Config)
    cs.store("model",node=Model)
    cs.store("data",node=Data)
    cs.store("trainer",node=Trainer)
