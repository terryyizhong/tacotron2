import os
import json
import time

import torch
import numpy as np
from scipy.io.wavfile import write

from hparams import create_hparams
from text import text_to_sequence
from train import load_model
from model import Tacotron2
from mel2samp import files_to_list, MAX_WAV_VALUE
from waveglow.denoiser import Denoiser

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

hparams = create_hparams("mask_padding=True")

model = load_model(hparams)
checkpoint_path = 'outdir/checkpoint_378000'
#model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(checkpoint_path ,map_location='cpu')['state_dict'].items()})
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval()#.half()

is_fp16 = True
waveglow_path = 'waveglow/waveglow_1084000'
sigma = 0.6
denoiser_strength = 0.0
sampling_rate = 22050
waveglow = torch.load(waveglow_path)['model']
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow.cuda().eval()
if is_fp16:
    from apex import amp
    waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")

if denoiser_strength > 0:
    denoiser = Denoiser(waveglow).cuda()

print("Init finished")

texts = []
with open('text_list20.txt', 'r') as f:
    for line in f:
        texts.append(line.strip())

start = time.time()
sum_taco = 0
length = 0
for text in texts:
    start1 = time.time()
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    mel_out = mel_outputs_postnet.squeeze()
    print('taco2 time: ', time.time() - start1)
    sum_taco += time.time()-start1

    mel = torch.autograd.Variable(mel_out.cuda())
    mel = torch.unsqueeze(mel, 0)
    mel = mel.half() if is_fp16 else mel
    start2 = time.time()
    with torch.no_grad():
        audio = waveglow.infer(mel, sigma=sigma)
        if denoiser_strength > 0:
            audio = denoiser(audio, denoiser_strength)
        audio = audio * MAX_WAV_VALUE
    print('glow time: ', time.time() - start2)
    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio = audio.astype('int16')
    audio_path = os.path.join(
        './', "{}_synthesis.wav".format("flask-test"))
    write(audio_path, sampling_rate, audio)
    length += (audio.shape[0])
print('total time: ', time.time() - start)
print('total taco time: ', sum_taco)
print('total glow time: ', time.time() - start - sum_taco)
print('total audio time: ', length / 22050.0)

