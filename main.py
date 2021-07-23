import librosa
import numpy as np
import yaml
import signal
import copy
import torch
from Griffin_Lim import griffin_lim
from scipy.io.wavfile import write
import os
args = None
with open("config.yaml","r") as yaml_file:
    args = yaml.load((yaml_file.read()))
gl = griffin_lim(args)
g = os.walk("testaudio")
for p,d,f_List in g:
    for f_name in f_List:
        mel,spect = gl.get_spectrograms(p+'/'+f_name)
        r1= gl.melspectrogram2wav(mel)
        r2 = gl.spectrogram2wav(spect)
        #生成数据存储位置
        write('melspect_gl/'+f_name, 22050, r1)
        write('spect_gl/'+f_name, 22050, r2)