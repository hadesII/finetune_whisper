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

def dataset_merge(train_list,dev_list):
    train_audio_transcript_pair_list = []
    eval_audio_transcript_pair_list = []
    with open(train_list) as fr:
        for line in fr:
            line = line.strip()
            data = json.loads(line)
            audio_id = data['key']
            audio_path = data['wav']
            text = data['orig_txt']
            train_audio_transcript_pair_list.append((audio_id, str(audio_path), text))

    with open(dev_list) as fr:
        for line in fr:
            line = line.strip()
            data = json.loads(line)
            audio_id = data['key']
            audio_path = data['wav']
            text = data['orig_txt']
            eval_audio_transcript_pair_list.append((audio_id, str(audio_path), text))


    random.shuffle(train_audio_transcript_pair_list)
    random.shuffle(eval_audio_transcript_pair_list)

    return train_audio_transcript_pair_list, eval_audio_transcript_pair_list
def get_audio_length_processor(min_audio_length, max_audio_length):
    def is_audio_in_length_range(duration):
        return min_audio_length < duration < max_audio_length

    return is_audio_in_length_range


def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)
