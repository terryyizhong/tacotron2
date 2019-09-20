import sys
import numpy as np
import torch
import os
import argparse

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
#from stft import STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from scipy.io.wavfile import write
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def plot_data(data, num, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', 
                       interpolation='none')
    plt.savefig(plot_dir + str(num) + '.png')


def infer(checkpoint_path, griffin_iters, texts):
    hparams = create_hparams()

    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval()#.half()

    for num, text in enumerate(texts):
        sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

        torch.save(mel_outputs_postnet.squeeze(),mel_dir + "/{}.pt".format(num))
        plot_data((mel_outputs.float().data.cpu().numpy()[0], mel_outputs_postnet.float().data.cpu().numpy()[0], alignments.float().data.cpu().numpy()[0].T), num)

        taco_stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length, sampling_rate=hparams.sampling_rate)

        mel_decompress = taco_stft.spectral_de_normalize(mel_outputs_postnet)
        mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
        spec_from_mel_scaling = 1000
        spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis)
        spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
        spec_from_mel = spec_from_mel * spec_from_mel_scaling

        audio = griffin_lim(torch.autograd.Variable(spec_from_mel[:, :, :-1]) ** 1.2, taco_stft.stft_fn, griffin_iters)

        audio = audio.squeeze()
        audio = audio.cpu().numpy()
        audio_path = os.path.join('samples', "{}_synthesis.wav".format(num))
        write(audio_path, hparams.sampling_rate, audio)
        print(audio_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--text', type=str,help='text to infer')
    parser.add_argument('-s', '--steps', type=int,help='griffin lim iters', default=80)
    parser.add_argument('-c', '--checkpoint', type=str,help='checkpoint path')
    parser.add_argument('-o', '--out_filename', type=str, help='output filename', default='sample')
    args = parser.parse_args()

    step = args.checkpoint.split('/')[-1]
    res_dir = './result_' + step
    mel_dir = res_dir + '/mel_out/'
    plot_dir = res_dir + '/plot_out/'
    wav_dir = res_dir + '/wav_out/'
    if not os.path.exists(mel_dir):
        os.makedirs(mel_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)

    texts = []
    with open(args.text, 'r') as f:
        for line in f:
            texts.append(line.strip())

    infer(args.checkpoint, args.steps, texts)
