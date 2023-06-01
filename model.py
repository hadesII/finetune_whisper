from lightning_transformers.core.seq2seq.model import Seq2SeqTransformer
from typing import Any
import evaluate
from peft import LoraConfig, get_peft_model
from transformers import WhisperForConditionalGeneration

# from transformers import WhisperTokenizer
import torch
import whisper
import os
from config import Model



class WhisperModule(Seq2SeqTransformer):

    def __init__(self, *args: Any,cfg:Model, downstream_model_type=WhisperForConditionalGeneration, lora=False, **kwargs: Any):
        super().__init__(*args,downstream_model_type=downstream_model_type, **kwargs)
        # self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v2", language="chinese", task="transcribe")
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, task="transcribe")
        self.cfg = cfg
        if lora == True:
            config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05,
                                bias="none")

            self.model = get_peft_model(self.model, config)


    def compute_generate_metrics(self, batch, prefix):
        labels = batch["labels"].long()
        output = self.model(**batch)
        out, loss = output["logits"], output["loss"]

        out[out == -100] = self.tokenizer.eot
        labels[labels == -100] = self.tokenizer.eot

        o_list, l_list = [], []
        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1)
            o_list.append(self.tokenizer.decode_sk(o, skip_special_tokens=True))
            l_list.append(self.tokenizer.decode_sk(l, skip_special_tokens=True))
        cer = self.metrics_cer.compute(references=l_list, predictions=o_list)
        wer = self.metrics_wer.compute(references=l_list, predictions=o_list)

        self.log("val/loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("val/cer", cer, on_step=True, prog_bar=True, logger=True)
        self.log("val/wer", wer, on_step=True, prog_bar=True, logger=True)


    def configure_metrics(self, stage: str):
        self.metrics_wer = evaluate.load('metrics/wer')
        self.metrics_cer = evaluate.load('metrics/cer')

    def on_validation_epoch_end(self) -> None:

        checkpoint_folder = os.path.join(self.cfg.checkpoint_root, f"checkpoints-{self.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        self.model.save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
