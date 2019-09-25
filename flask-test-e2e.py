import os
import json
import time

import torch
import numpy as np
from scipy.io.wavfile import write
from flask import Flask, request, Response, send_file, make_response, send_from_directory

from hparams import create_hparams
from text import text_to_sequence
from train_cpu import load_model
from model import Tacotron2
from mel2samp import files_to_list, MAX_WAV_VALUE
from waveglow.denoiser import Denoiser

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
app = Flask(__name__)

@app.route('/api/test', methods=['POST'])
def test():
    r = request.json
    r_json = json.loads(r)
    text = r_json['data']
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    mel_out = mel_outputs_postnet.squeeze()
    
    mel = torch.autograd.Variable(mel_out.cuda())
    mel = torch.unsqueeze(mel, 0)
    mel = mel.half() if is_fp16 else mel
    with torch.no_grad():
        audio = waveglow.infer(mel, sigma=sigma)
        if denoiser_strength > 0:
            audio = denoiser(audio, denoiser_strength)
        audio = audio * MAX_WAV_VALUE
    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio = audio.astype('int16')
    audio_path = os.path.join(
        './', "{}_synthesis.wav".format("flask-test"))
    write(audio_path, sampling_rate, audio)
    #print(audio_path)
    #with open(audio_path, 'rb') as f:
    #    fp = f.read()
    return make_response(send_file(audio_path))



    #response = {'message': 'Data type:{},Shape:{}'.format(type(audio), audio.shape)}
    #response_pickled = json.dumps(response)
    #return Response(response=response_pickled, status=200, mimetype="application/json")


# start flask app
hparams = create_hparams("mask_padding=True")

model = load_model(hparams)
checkpoint_path = 'outdir/checkpoint_378000'
model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(checkpoint_path ,map_location='cpu')['state_dict'].items()})
_ = model.eval()#.half()

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

app.run(host="0.0.0.0", port=5000)


