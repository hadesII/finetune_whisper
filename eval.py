from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import gc
from peft import PeftModel, PeftConfig
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainer
from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperProcessor
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, DatasetDict
import os
import json
import torchaudio

language= "chinese"
language_abbr = "zh"
task = "transcribe"
peft_model_id = "temp_large-v2/lightning_logs/version_34/checkpoints/epoch\=3-step\=632908.ckpt"
peft_config = PeftConfig.from_pretrained(peft_model_id)

print(peft_config.base_model_name_or_path)


model = WhisperForConditionalGeneration.from_pretrained(
	peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
)
model = PeftModel.from_pretrained(model, peft_model_id)

processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
feature_extractor = WhisperFeatureExtractor.from_pretrained(peft_config.base_model_name_or_path)
tokenizer = WhisperTokenizer.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)


data_dir = 'data/zh_yue'
eval_audio_transcript_pair_list = []
dev_list = os.path.join(data_dir, 'dev.list')

with open(dev_list) as fr:
    for line in fr:
        line = line.strip()
        data = json.loads(line)
        audio_id = data['key']
        audio_path = data['wav']
        text = data['txt']
        eval_audio_transcript_pair_list.append((audio_id, str(audio_path), text))
print(len(eval_audio_transcript_pair_list))

def load_from_local_path(wav_path, sample_rate = 16000, num_channel = 1):
    waveform, sr = torchaudio.backend.sox_io_backend.load(wav_path)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
    assert num_channel == 1
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform, sample_rate

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, audio_info_list, sample_rate) -> None:
        super().__init__()
        self.audio_info_list = audio_info_list
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.audio_info_list)

    def __getitem__(self, id):
        audio_id, audio_path, text = self.audio_info_list[id]
        waveform, samplerate = load_from_local_path(audio_path)
        waveform = waveform.tolist()
        mel = processor.feature_extractor(waveform, sampling_rate=self.sample_rate).input_features[0]
        labels =  processor.tokenizer(text).input_ids

        return  {
            "input_features": mel,
            "labels": labels,
        }

dev_dataset = SpeechDataset(eval_audio_transcript_pair_list, 16000)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

import evaluate

metric = evaluate.load("wer")


eval_dataloader = DataLoader(dev_dataset, batch_size=8, collate_fn=data_collator)

model.eval()
for step, batch in enumerate(tqdm(eval_dataloader)):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            generated_tokens = (
                    model.generate(
                            input_features=batch["input_features"].to("cuda"),
                            decoder_input_ids=batch["labels"][:, :4].to("cuda"),
                            max_new_tokens=255,
                    ).cpu().numpy()
            )
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            metric.add_batch(predictions=decoded_preds, references=decoded_labels)
	
    del generated_tokens, labels, batch
    gc.collect()
	
wer = 100 * metric.compute()
print(f"{wer=}")
