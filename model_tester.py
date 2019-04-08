import os
import random

import numpy as np
import tensorflow as tf
from PIL import Image

# carica il modello da testare
model = tf.keras.models.load_model('saved_models/transfer_learning_epoch_10_0.9844.h5')
categories = ['BASSOON', 'FLUTE', 'PIANO', 'TROMBONE']
test_dir = 'test/'  # cartella contenente i file da dare in input


def prepare(filepath):
    img_array = []
    im = Image.open(filepath)
    im = im.resize((300, 300), Image.ANTIALIAS)
    img_array.append(np.array(im))
    img_array = np.array(img_array) / 255.0  # Normalize RGB
    return img_array

mappa = {categories[0]: 0, categories[1]: 0, categories[2]: 0, categories[3]: 0}

for instrument in categories:
    d = test_dir + instrument.lower() + '/'
    for i in range(20):
        file = random.choice(os.listdir(d))
        prediction = model.predict([prepare(d + str(file))])
        pred = prediction[0].tolist()
        m = max(pred)
        c = categories[pred.index(m)]
        mappa[c] = mappa[c] + 1

    print('STRUMENTO DA CLASSIFICARE: ' + instrument)
    print('-------------------------------------------------')
    print([[k, v] for k, v in mappa.items()])
    print('-------------------------------------------------')
    print()
    mappa = {categories[0]: 0, categories[1]: 0, categories[2]: 0, categories[3]: 0}
