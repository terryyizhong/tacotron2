# -*- coding:utf-8 -*-
import os
import json
import time
from random import shuffle

import torch
import uuid
import numpy as np
from scipy.io.wavfile import write
from pypinyin import pinyin, load_single_dict, Style
from flask import Flask, request, Response, send_file, make_response, send_from_directory, jsonify

from hparams import create_hparams
from text import text_to_sequence, an_to_cn
from train import load_model
from model import Tacotron2
from mel2samp import files_to_list, MAX_WAV_VALUE
#from waveglow.denoiser import Denoiser

MODEL_DIR = './TTS_models/'

MODEL_LIST={1: os.path.join(MODEL_DIR, 'english-woman-mature-LJspeech'),
            2: os.path.join(MODEL_DIR, 'english-woman-vivid-17'),
            3: os.path.join(MODEL_DIR, 'english-VOA-Bryan-male'), 
            4: os.path.join(MODEL_DIR, 'mandarin-woman-biaobei'), 
            5: os.path.join(MODEL_DIR, 'mandarin-17male700'), 
            6: os.path.join(MODEL_DIR, 'tuobi-ketang-name'), 
            }

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
    rjson = request.json
    print('rjson:', rjson)
    text = rjson['text']
    model_idx = rjson['model']
    language = rjson['language']

    taco2 = taco2s[model_idx-1]
    waveglow = waveglows[model_idx-1]
    cleaner = 'english_cleaners'

    if language == 'chinese':
        text = an_to_cn(text)
        text = ' '.join(chs_pinyin(text)).strip()
        cleaner = 'chinese_cleaners'

    print(text)
    # inference
    sequence = np.array(text_to_sequence(text, [cleaner]))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    mel_outputs, mel_outputs_postnet, _, alignments = taco2.inference(sequence)
    mel_out = mel_outputs_postnet.squeeze()
    mel = torch.autograd.Variable(mel_out.cuda())
    mel = torch.unsqueeze(mel, 0)
    mel = mel.half() if is_fp16 else mel
    with torch.no_grad():
        audio = waveglow.infer(mel, sigma=sigma)
        audio = audio * MAX_WAV_VALUE
    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio = audio.astype('int16')
    print('infer done')

    audio_path = os.path.join('./tmp/server/', "{}.wav".format(uuid.uuid4()))
    write(audio_path, sampling_rate, audio)

    return make_response(send_file(audio_path))

    #response_pickled = json.dumps(resp)
    #return Response(response=response_pickled, status=200, mimetype="application/json")


#if __name__ == "__main__":

# init model1 tacotron2
hparams = create_hparams("mask_padding=True")
hparams_chs = create_hparams("chs_symbols_diff=5")
taco2s = []
for i in range(6):
    model = load_model(hparams) if i < 3 else load_model(hparams_chs)
    # cpu init
    #model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(checkpoint_path ,map_location='cpu')['state_dict'].items()})
    model.load_state_dict(torch.load(os.path.join(MODEL_LIST[i+1], 'taco2_model'))['state_dict'])
    model.cuda().eval()
    taco2s.append(model)

# init model2 waveglow
is_fp16 = True
sigma = 0.6
denoiser_strength = 0.0
sampling_rate = 22050
if is_fp16:
    from apex import amp

waveglows = []
for i in range(6):
    waveglow = torch.load(os.path.join(MODEL_LIST[i+1], 'waveglow_model'))['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval()
    if is_fp16:
        waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")
    waveglows.append(waveglow)

print('INIT DONE')
#app.run(host="0.0.0.0", port=5000)


