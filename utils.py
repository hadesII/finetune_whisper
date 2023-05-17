import torchaudio
import torch
import os
import json
import random





def load_from_local_path(wav_path, sample_rate = 16000, num_channel = 1):
    waveform, sr = torchaudio.backend.sox_io_backend.load(wav_path)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
    assert num_channel == 1
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform, sample_rate

def dataset_merge(**kwargs):
    train_audio_transcript_pair_list = []
    eval_audio_transcript_pair_list = []
    for k, v in kwargs.items():
        train_list = os.path.join(v, 'train.list.tokenizer')
        with open(train_list) as fr:
            for line in fr:
                line = line.strip()
                data = json.loads(line)
                audio_id = data['key']
                audio_path = data['wav']
                text= data['txt']
                ids = data['ids']
                train_audio_transcript_pair_list.append((audio_id, str(audio_path), text, ids))
        dev_list = os.path.join(v, 'dev.list.tokenizer')
        with open(dev_list) as fr:
            for line in fr:
                line = line.strip()
                data = json.loads(line)
                audio_id = data['key']
                audio_path = data['wav']
                text= data['txt']
                ids = data['ids']
                eval_audio_transcript_pair_list.append((audio_id, str(audio_path), text, ids))

    random.shuffle(train_audio_transcript_pair_list)
    random.shuffle(eval_audio_transcript_pair_list)

    return train_audio_transcript_pair_list, eval_audio_transcript_pair_list
