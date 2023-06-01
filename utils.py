import torchaudio
import torch
import json
import random
import torchaudio.transforms as at
import os
from pathlib import Path





def load_from_local_path(wav_path, sample_rate = 16000, num_channel = 1):
    waveform, sr = torchaudio.backend.sox_io_backend.load(wav_path)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
    assert num_channel == 1
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform, sample_rate

# def dataset_merge(train_list,dev_list):
#     train_audio_transcript_pair_list = []
#     eval_audio_transcript_pair_list = []
#     with open(train_list) as fr:
#         for line in fr:
#             line = line.strip()
#             data = json.loads(line)
#             audio_id = data['key']
#             audio_path = data['wav']
#             text = data['txt']
#             train_audio_transcript_pair_list.append((audio_id, str(audio_path), text))
#
#     with open(dev_list) as fr:
#         for line in fr:
#             line = line.strip()
#             data = json.loads(line)
#             audio_id = data['key']
#             audio_path = data['wav']
#             text = data['txt']
#             eval_audio_transcript_pair_list.append((audio_id, str(audio_path), text))
#
#
#     random.shuffle(train_audio_transcript_pair_list)
#     random.shuffle(eval_audio_transcript_pair_list)
#
#     return train_audio_transcript_pair_list, eval_audio_transcript_pair_list


def dataset_merge(data_path,lang_pair=("cantonese",)):
    train_audio_transcript_pair_list = []
    eval_audio_transcript_pair_list = []
    for v in lang_pair:
        train_list = data_path /  v / "txt" / 'train.list.tokenizer'
        with open(train_list) as fr:
            for line in fr:
                line = line.strip()
                data = json.loads(line)
                audio_id = data['key']
                audio_path = data['wav']
                if not Path(audio_path).is_file():
                    import pdb; pdb.set_trace()
                text= data['txt']
                ids = data['ids']
                train_audio_transcript_pair_list.append((audio_id, str(audio_path), text, ids))
        dev_list = data_path / v / "txt" / 'dev.list.tokenizer'
        with open(dev_list) as fr:
            for line in fr:
                line = line.strip()
                data = json.loads(line)
                audio_id = data['key']
                audio_path = data['wav']
                if not Path(audio_path).is_file():
                    import pdb; pdb.set_trace()
                text= data['txt']
                ids = data['ids']
                eval_audio_transcript_pair_list.append((audio_id, str(audio_path), text, ids))

    random.shuffle(train_audio_transcript_pair_list)
    random.shuffle(eval_audio_transcript_pair_list)

    return train_audio_transcript_pair_list, eval_audio_transcript_pair_list

def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform
