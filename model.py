from lightning_transformers.core.seq2seq.model import Seq2SeqTransformer
from typing import Any
import evaluate
from peft import LoraConfig, get_peft_model
from transformers import WhisperForConditionalGeneration

from transformers import WhisperTokenizer



class WhisperModule(Seq2SeqTransformer):

    def __init__(self, *args: Any, downstream_model_type=WhisperForConditionalGeneration, lora=False, **kwargs: Any):
        super().__init__(*args,downstream_model_type=downstream_model_type, **kwargs)
        self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v2", language="chinese", task="transcribe")
        if lora == True:
            config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05,
                                bias="none")

            self.model = get_peft_model(self.model, config)


    def compute_generate_metrics(self, batch, prefix):
        tgt_lns = self.tokenize_labels(batch["labels"])
        pred_lns = self.model.generate(inputs=batch["input_features"], language="chinese", task="transcribe")
        # wrap targets in list as score expects a list of potential references
        result = 100 * self.wer.compute(predictions=pred_lns, references=tgt_lns)
        self.log(f"{prefix}_wer", result, on_step=False, on_epoch=True, prog_bar=True)

    def configure_metrics(self, stage: str):
        self.wer = evaluate.load("wer")
