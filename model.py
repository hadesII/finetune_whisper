from lightning_transformers.core.seq2seq.model import Seq2SeqTransformer
from typing import Any, Optional
import torch
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


    # def compute_generate_metrics(self, batch, prefix):
    #     label_ids = batch["labels"]
    #     label_ids[label_ids == -100] = self.tokenizer.pad_token_id
    #     tgt_lns = self.tokenize_labels(label_ids)
    #     pred_lns = self.super().model.generate(inputs=batch["input_features"], language="chinese", task="transcribe")
    #     pred_lns = self.tokenizer.batch_decode(pred_lns, skip_special_tokens=True)
    #     # wrap targets in list as score expects a list of potential references
    #     result = 100 * self.wer.compute(predictions=pred_lns, references=tgt_lns)
    #     self.log(f"{prefix}_wer", result, on_step=False, on_epoch=True, prog_bar=True)
    #
    def configure_metrics(self, stage: str):
        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")

    def common_step(self, prefix: str, batch: Any) -> torch.Tensor:
        outputs = self.model(**batch)
        import pdb; pdb.set_trace()
        loss, logits = outputs[:2]
        import pdb; pdb.set_trace()
        if self.should_compute_generate_metrics:
            self.compute_generate_metrics(batch, prefix)
        return loss

    def compute_generate_metrics(self, batch, prefix):
        # input_ids = batch["input_features"]
        labels = batch["labels"].long()
        # # dec_input_ids = batch["dec_input_ids"].long()
        #
        # audio_features = self.model.encoder(input_ids)
        # out = self.model.decoder(encoder_hidden_states=audio_features)
        out = self.model(batch["input_features"])

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

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

        return {
            "cer": cer,
            "wer": wer,
            "loss": loss
        }
