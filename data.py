import pytorch_lightning as pl
from utils import load_from_local_path, dataset_merge,load_wave
from transformers import WhisperProcessor
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import whisper

import os

class WhisperDataModule(pl.LightningDataModule):

    def __init__(self, batch_size = 1, num_workers = 2, sample_rate = 16000):
        super().__init__()
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", language="chinese", task="transcribe")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate

    def setup(self, stage: str,**kwargs) -> None:
        data_dir = 'data/zh_yue'
        train_list = os.path.join(data_dir, 'train.list.tokenizer')
        dev_list = os.path.join(data_dir, 'dev.list.tokenizer')
        train_audio_transcript_pair_list, eval_audio_transcript_pair_list = dataset_merge(train_list=train_list,dev_list=dev_list)
        self.train_dataset = SpeechDataset(train_audio_transcript_pair_list, self.sample_rate, processor=self.processor)
        self.valid_dataset = SpeechDataset(eval_audio_transcript_pair_list, self.sample_rate, processor=self.processor)

    def train_dataloader(self) :
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=self.whisperdatacollatorwhithadding
                          )
    def val_dataloader(self) :
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=self.whisperdatacollatorwhithadding
                          )


    def whisperdatacollatorwhithadding(self,features):

        input_ids, labels, dec_input_ids = [], [], []
        for f in features:
            if len(f) != 0:
                input_ids.append(f["input_ids"])
                labels.append(f["labels"])
                dec_input_ids.append(f["dec_input_ids"])

        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths + dec_input_ids_length)

        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in
                  zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in
                         zip(dec_input_ids, dec_input_ids_length)]  # 50257 is eot token id

        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["input_ids"] = input_ids

        return batch

        # # split inputs and labels since they have to be of different lengths and need different padding methods
        # # first treat the audio inputs by simply returning torch tensors
        # input_features = [{"input_features": feature["input_features"]} for feature in features]
        # batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        #
        #
        # # get the tokenized label sequences
        # label_features = [{"input_ids": feature["labels"]} for feature in features]
        # # pad the labels to max length
        # labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        #
        # # replace padding with -100 to ignore loss correctly
        # labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        #
        # # if bos token is appended in previous tokenization step,
        # # cut bos token here as it's append later anyways
        # if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
        #     labels = labels[:, 1:]
        #
        # batch["labels"] = labels
        #
        #
        # return batch


# class SpeechDataset(Dataset):
#     def __init__(self, audio_info_list, sample_rate, processor) -> None:
#         super().__init__()
#         self.audio_info_list = audio_info_list
#         self.sample_rate = sample_rate
#         self.processor = processor
#
#     def __len__(self):
#         return len(self.audio_info_list)
#
#     def __getitem__(self, id):
#         audio_id, audio_path, text = self.audio_info_list[id]
#
#         waveform, samplerate = load_from_local_path(audio_path)
#         waveform = waveform.tolist()
#
#         mel = self.processor.feature_extractor(waveform, sampling_rate=self.sample_rate).input_features[0]
#         labels = self.processor.tokenizer(text).input_ids
#
#         return {
#             "input_features": mel,
#             "labels": labels,
#         }


class SpeechDataset(Dataset):
    def __init__(self, audio_info_list, sample_rate) -> None:
        super().__init__()

        self.audio_info_list = audio_info_list
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.audio_info_list)

    def __getitem__(self, id):
        audio_id, audio_path, text, ids = self.audio_info_list[id]

        try:
            audio = load_wave(audio_path, sample_rate=self.sample_rate)
        except:
            return {}
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio)

        text = ids[:-1]
        labels = ids[1:]

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": text
        }
