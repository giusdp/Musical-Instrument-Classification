import os

import matplotlib

matplotlib.use('agg')

from matplotlib import cm
from tqdm import tqdm
import pylab
import librosa
from librosa import display
import numpy as np

WAV_DIR = 'bassoon/' # Cartella contenente i wav file
IMG_DIR = 'bassoon/' # Cartella dove vengono salvati gli spettrogrammi
wav_files = os.listdir(WAV_DIR)

for f in tqdm(wav_files):
    try:
        # Leggi i wav file
        y, sr = librosa.load(WAV_DIR+f)
        
        # Filtro pre emphasis (come dal paper)
        pre_emphasis = 0.97
        y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
        
        # Fai lo spettrogramma, valori presi seguendo il paper
        M = librosa.feature.melspectrogram(y, sr, 
                                           fmax = sr/2,
                                           n_fft=2048, 
                                           hop_length=512, 
                                           n_mels = 96,
                                           power = 2)

        log_power = librosa.power_to_db(M, ref=np.max)
        
        # Crea l'immagine con matplotlib e salvala
        pylab.figure(figsize=(3,3))
        pylab.axis('off') 
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
        librosa.display.specshow(log_power, cmap=cm.jet)
        pylab.savefig(IMG_DIR + f[:-4]+'.png', bbox_inches=None, pad_inches=0)
        pylab.close()

    except Exception as e:
        print(f, e)
        pass