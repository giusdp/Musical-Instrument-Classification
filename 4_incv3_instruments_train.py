import os

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import optimizers
from tensorflow.contrib.keras import callbacks
from tensorflow.contrib.keras import regularizers
from tensorflow.contrib.keras import models


IMG_DIR = 'spectrograms_dataset/'
IMG_SAMPLE_DIR = 'spectrogram_samples/'
IMG_HEIGHT = 300  # 216
IMG_WIDTH = 300  # 216
NUM_CLASSES = 4
NUM_EPOCHS = 10
BATCH_SIZE = 32
L2_LAMBDA = 0.001

# Prendi i 4 file rappresentanti le classi
sample_files = ['Bassoon.png',
                'Flute.png',
                'Piano.png',
                'Trombone.png'
                ]

label_dict = {'Bassoon': 0,
              'Flute': 1,
              'Piano': 2,
              'Trombone': 3,
              }

one_hot = OneHotEncoder(n_values=NUM_CLASSES)

all_files = os.listdir(IMG_DIR)

# Prende le classi dei file dal nome
label_array = []
for file_ in all_files:
    vals = file_[:-4].split('_')
    label_array.append(label_dict[vals[1]])

cl_weight = compute_class_weight(class_weight='balanced',
                                 classes=np.unique(label_array),
                                 y=label_array)

# Split del dataset
train_files, test_files, train_labels, test_labels = train_test_split(all_files,
                                                                      label_array,
                                                                      random_state=10,
                                                                      test_size=0.1
                                                                      )

# Ulteriore split per la validation
val_files, test_files, val_labels, test_labels = train_test_split(test_files, test_labels,
                                                                  random_state=10,
                                                                  test_size=0.5
                                                                  )

f, axarr = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(20, 10))
coordinates = [(0, 0), (0, 1), (0, 2), (0, 3),
               (1, 0), (1, 1), (1, 2)]

for i, file_ in enumerate(sample_files):
    im = Image.open(IMG_SAMPLE_DIR + file_)
    # im = im.resize((IMG_WIDTH, IMG_HEIGHT), resample = Image.ANTIALIAS)
    axarr[coordinates[i]].imshow(np.asarray(im))
    axarr[coordinates[i]].axis('off')
    axarr[coordinates[i]].set_title(file_[:-4], fontsize=18)

plt.savefig('classes_samples.png') #salva l'immagine con gli spettrogrammi di esempio
# sys.exit()

# scarica ed utilizza i pesi da imagenet per vgg16, questa è la base convoluzionale
conv_base = tf.contrib.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet',
                                                input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)  # 3 per i canali RGB
                                                )
conv_base.summary()
# stampa info sulla base conv

# RETE
# Layer finali per la vgg
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())

model.add(layers.Dense(512, name='dense_1', kernel_regularizer=regularizers.l2(L2_LAMBDA)))
model.add(layers.Activation(activation='relu', name='activation_1'))

model.add(layers.Dense(NUM_CLASSES, activation='softmax', name='dense_output'))
model.summary()

conv_base.trainable = False
model.summary()


def load_batch(file_list):
    img_array = []
    idx_array = []
    label_array = []

    for file_ in file_list:
        im = Image.open(IMG_DIR + file_)
        im = im.resize((IMG_WIDTH, IMG_HEIGHT), Image.ANTIALIAS)
        img_array.append(np.array(im))

        vals = file_[:-4].split('_')
        idx_array.append(vals[0])
        label_array.append([label_dict[vals[1]]])

    label_array = one_hot.fit_transform(label_array).toarray()
    img_array = np.array(img_array) / 255.0  # Normalizza RGB

    return img_array, np.array(label_array), np.array(idx_array)


def batch_generator(files, BATCH_SIZE):
    L = len(files)

    while True:

        batch_start = 0
        batch_end = BATCH_SIZE

        while batch_start < L:
            limit = min(batch_end, L)
            file_list = files[batch_start: limit]
            batch_img_array, batch_label_array, batch_idx_array = load_batch(file_list)

            yield (batch_img_array, batch_label_array)

            batch_start += BATCH_SIZE
            batch_end += BATCH_SIZE


# Ottimizzatore per la fase di training
optimizer = optimizers.Adam(lr=1e-5)

loss = 'categorical_crossentropy'

metrics = ['categorical_accuracy']

# Salva un modello dopo ogni epoch, in base alla rete può pesare molto
filepath = "saved_models/transfer_learning_epoch_{epoch:02d}_{val_categorical_accuracy:.4f}.h5"
checkpoint = callbacks.ModelCheckpoint(filepath,
                                       monitor='val_categorical_accuracy',
                                       verbose=0,
                                       save_best_only=False)
callbacks_list = [checkpoint]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
STEPS_PER_EPOCH = len(train_files) // BATCH_SIZE
VAL_STEPS = len(val_files) // BATCH_SIZE


# Per non iniziare da 0 puoi caricare in model un modello salvato, ma il conto delle epoch parte da 0 di nuovo
# model = tf.keras.models.load_model('saved_models/transfer_learning_epoch_03_0.9185.h5')
# INIZIA TRAINING
history = model.fit_generator(generator=batch_generator(train_files, BATCH_SIZE),
                              epochs=NUM_EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              class_weight=cl_weight,
                              validation_data=batch_generator(val_files, BATCH_SIZE),
                              validation_steps=VAL_STEPS,
                              callbacks=callbacks_list,
                              )

# Da qui in poi si fa la predizione per creare la confusion matrix
# Carica un modello salvato, magari saltanto la fase di training
model = models.load_model(filepath='saved_models/transfer_learning_epoch_03_0.9185.h5')

TEST_STEPS = len(test_files) // BATCH_SIZE

# Fai tutte le predizioni
pred_probs = model.predict_generator(generator=batch_generator(test_files, BATCH_SIZE),
                                     steps=TEST_STEPS)
pred = np.argmax(pred_probs, axis=-1)

from sklearn.metrics import confusion_matrix, accuracy_score
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')


plot_confusion_matrix(confusion_matrix(y_true=test_labels[:len(pred)], y_pred=pred),
                      classes=label_dict.keys())

# print('Test Set F-score =  {0:.2f}'.format(f1_score(y_true=test_labels[:len(pred)], y_pred=pred, average='macro')))
print('Test Set Accuracy =  {0:.2f}'.format(accuracy_score(y_true=test_labels[:len(pred)], y_pred=pred)))
