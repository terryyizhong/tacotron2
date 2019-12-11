# -*- coding:utf-8 -*-
import os
import json
import time
from random import shuffle

import torch
import uuid
import numpy as np
from scipy.io.wavfile import write
from flask import Flask, request, Response, send_file, make_response, send_from_directory, jsonify

from hparams import create_hparams
from text import text_to_sequence
from train_cpu import load_model
from model import Tacotron2
from mel2samp import files_to_list, MAX_WAV_VALUE
from waveglow.denoiser import Denoiser


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# chinese2pinyin
def chs_pinyin(text):
    pys = pinyin(text, style=Style.TONE3)
    results = []
    sentence = []
    for i in range(len(pys)):
        if pys[i][0][0] == "，" or pys[i][0][0] == "、" or pys[i][0][0] == '·':
            pys[i][0] = ','
        elif pys[i][0][0] == '。' or pys[i][0][0] == "…":
            pys[i][0] = '.'
        elif pys[i][0][0] == '―' or pys[i][0][0] == "――" or pys[i][0][0] == '—' or pys[i][0][0] == '——':
            pys[i][0] = ','
        elif pys[i][0][0] == "；":
            pys[i][0] = ';'
        elif pys[i][0][0] == "：":
            pys[i][0] = ':'
        elif pys[i][0][0] == "？":
            pys[i][0] = '?'
        elif pys[i][0][0] == "！":
            pys[i][0] = '!'
        elif pys[i][0][0] == "《" or pys[i][0][0] == '》' or pys[i][0][0] == '（' or pys[i][0][0] == '）':
            continue
        elif pys[i][0][0] == '“' or pys[i][0][0] == '”' or pys[i][0][0] == '‘' or pys[i][0][0] == '’' or pys[i][0][
            0] == '＂':
            continue
        elif pys[i][0][0] == '(' or pys[i][0][0] == ')' or pys[i][0][0] == '"' or pys[i][0][0] == '\'':
            continue
        elif pys[i][0][0] == ' ' or pys[i][0][0] == '/' or pys[i][0][0] == '<' or pys[i][0][0] == '>' or pys[i][0][
            0] == '「' or pys[i][0][0] == '」':
            continue

        if pys[i][0][-1] in "qwertyuiopasdfghjklzxcvbnm":
            pys[i][0] = pys[i][0] + '5'
        sentence.append(pys[i][0])

        if pys[i][0] in ",.;?!:":
            results.append(' '.join(sentence))
            sentence = []

    if len(sentence) > 0:
        sent = ' '.join(sentence)
        if sent[-1] not in '.?!':
            sent = sent + ' .'
        results.append(sent)

    return results


# start flask app
app = Flask(__name__)

@app.route('/api/test', methods=['POST'])
def test():
    #resp = {"success": False}

    # preprocess
    r = request.json
    print('r:', r)
    r_json = r
    print('rjson:', r_json)
    data = r_json['data']
    print(data)
    # inference
    sequence = np.array(text_to_sequence(data, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
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
    print('infer done')

    #resp['audio_array'] = list(audio)
    #resp["success"] = True
    #print(resp)
    #return jsonify(resp)
    
    audio_path = os.path.join('./tmp/server/', "{}.wav".format(uuid.uuid4()))
    write(audio_path, sampling_rate, audio)
    return make_response(send_file(audio_path))

    #response_pickled = json.dumps(resp)
    #return Response(response=response_pickled, status=200, mimetype="application/json")


#if __name__ == "__main__":

# init model1
hparams = create_hparams("mask_padding=True")
model = load_model(hparams)
checkpoint_path = 'english-woman-vivid-17/checkpoint_378000'
# cpu init
#model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(checkpoint_path ,map_location='cpu')['state_dict'].items()})
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval()#.half()

# init model2
is_fp16 = True
waveglow_path = 'english-woman-vivid-17/waveglow_1084000'
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

#app.run(host="0.0.0.0", port=5000)


