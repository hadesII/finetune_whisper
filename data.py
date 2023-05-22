import pytorch_lightning as pl
# from utils import load_from_local_path, dataset_merge
from transformers import WhisperProcessor
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, Audio

import os

class WhisperDataModule(pl.LightningDataModule):

    def __init__(self, batch_size = 8, num_workers = 2, sample_rate = 16000):
        super().__init__()
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", language="chinese", task="transcribe")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate

    def setup(self, stage: str,**kwargs) -> None:
        data_dir = '/home/yangwei/dtm/corpus/data/zh_yue'
        train_list = os.path.join(data_dir, 'train.list')
        dev_list = os.path.join(data_dir, 'dev.list')
        audio_dataset = load_dataset('json', data_files={'train': train_list, 'test': dev_list})

        audio_dataset = audio_dataset.cast_column("audio", Audio(sampling_rate=16000))
        # if 'duration' in audio_dataset['train'].features.keys():
        #     is_audio_in_length = get_audio_length_processor(args.min_audio_len, args.max_audio_len)
        #     audio_dataset["train"] = audio_dataset["train"].filter(is_audio_in_length, input_columns=["duration"])
        #     audio_dataset["test"] = audio_dataset["test"].filter(is_audio_in_length, input_columns=["duration"])
        #     print(f"过滤后训练数据：{audio_dataset['train'].num_rows}，测试数据：{audio_dataset['test'].num_rows}")
        # train_audio_transcript_pair_list, eval_audio_transcript_pair_list = dataset_merge(train_list=train_list,dev_list=dev_list)
        self.train_dataset, self.valid_dataset = audio_dataset["train"] , audio_dataset["test"]
        # self.train_dataset = SpeechDataset(train_audio_transcript_pair_list, self.sample_rate, processor=self.processor)
        # self.valid_dataset = SpeechDataset(eval_audio_transcript_pair_list, self.sample_rate, processor=self.processor)

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

        batch["labels"] = labels.cuda()


        return batch


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
