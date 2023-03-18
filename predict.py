# imports

import keras.api._v2.keras as keras
import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras import optimizers, models
import numpy as np
import librosa as lb
import os
import matplotlib.pyplot as plt
import librosa.display
from keras.utils import img_to_array, load_img
from sklearn.preprocessing import normalize
import math

# opening and store file in a variable

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()

# use Keras model_from_json to make a loaded model

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("model.h5")
print("Loaded Model from disk")

# compile and evaluate loaded model

adam = optimizers.Adam(lr=1e-4)

loaded_model.compile(loss='sparse_categorical_crossentropy',optimizer=adam,metrics=['accuracy'])


SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
num_segments = 5
hop_length = 512
n_fft = 2048
n_mfcc = 13
samples_ps = int(SAMPLES_PER_TRACK/num_segments) # ps = per segment
expected_vects_ps = math.ceil(samples_ps/hop_length)

#predict from a file path
def predict(wav_file):
    # extract mfcc
    signal,sr = librosa.load(wav_file,sr=SAMPLE_RATE)
    for s in range(num_segments):
        start_sample = samples_ps * s
        finish_sample = start_sample + samples_ps

        mfcc = lb.feature.mfcc(signal[start_sample:finish_sample],
                                sr = sr,
                                n_fft = n_fft,
                                n_mfcc = n_mfcc,
                                hop_length = hop_length)

        #print (mfcc.shape)
        mfcc = mfcc.T
        #print (mfcc.shape)
    # we need a 4d array to feed to the model for prediction: (# samples, # time steps, # coefficients, # channels)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    mfcc = np.resize(mfcc, (1, 130, 13, 1))
    #print(mfcc.shape)
    #print (mfcc[0])

    # get the predicted label
    prediction = loaded_model.predict(mfcc)
    predicted_index = np.argmax(prediction, axis=1)
    #print("The predicted label is: ", predicted_index[0])
    genres = ['pop','metal','disco','blues','reggae','classical','rock','hiphop','country','jazz']
    for x in range(9):
        if x == predicted_index[0]:
            print(genres[x])
            return genres[x]
    return "Error"

def plot_mfcc(fp):
    signal,sr = librosa.load(fp,sr=SAMPLE_RATE)
    n_fft = 2048
    hop_length = 1024
    mel = librosa.feature.mfcc(signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
    mel = librosa.power_to_db(mel, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title("MFCC")
    #plt.savefig("/Users/tyler/Desktop/music-genre-classification-cnn-master/pngNew/mel_spectrogram.png")
    plt.show()
    #plt.savefig('C:Users/Kenne/Desktop/MusicWebsite/static/upload.png')


#(pop, metal, disco, blues, reggae, classical, rock, hiphop, country, jazz)
#predict("static/blues.00001.wav")
#predict("static/disco.00004.wav")
#predict("static/countryTest.wav")
#predict("static/jars.wav")


#plot_mfcc("static/pass.wav")

#predict("static/boogie.wav")
#plot_mfcc("static/boogie.wav")
