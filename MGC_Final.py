import os
import librosa
import librosa.display
import math
import json 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import json
import keras.api._v2.keras as keras
from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D, BatchNormalization
from keras.models import Sequential
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn import metrics
from IPython.display import clear_output
import seaborn as sns
from keras import backend as K



dataset_path = r"/Users/tyler/Desktop/music-genre-classification-cnn-master/data/genres_original"
json_path = r"/Users/tyler/Desktop/music-genre-classification-cnn-master/output/data.json"
# set sample rate and duration of audio files to be processed
SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
labels = [f.name for f in os.scandir(dataset_path) if f.is_dir()]


fp = "/Users/tyler/Desktop/music-genre-classification-cnn-master/data/genres_original/blues/blues.00000.wav"
fp2 = "/Users/tyler/Desktop/music-genre-classification-cnn-master/data/genres_original/classical/classical.00000.wav"


# plot mfcc of audio file to visualize the data we are working with
def plot_mfcc(fp):
    signal,sr = librosa.load(fp,sr=SAMPLE_RATE)
    n_fft = 2048
    hop_length = 1024
    mfcc = librosa.feature.mfcc(signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
    mfcc = librosa.power_to_db(mfcc, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, sr=sr, hop_length=hop_length, x_axis="time", cmap="Spectral")
    plt.colorbar(format="%+2.0f dB")
    plt.title("MFCC")
    #plt.savefig("/Users/tyler/Desktop/music-genre-classification-cnn-master/pngNew/mel_spectrogram.png")
    plt.show()

plot_mfcc(fp)
plot_mfcc(fp2)



def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048,
             hop_length=512, num_segments=5):
    # Data storage dictionary
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": [],
    }
    samples_ps = int(SAMPLES_PER_TRACK/num_segments) # ps = per segment
    expected_vects_ps = math.ceil(samples_ps/hop_length)
    
    # loop through all the genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # ensuring not at root
        if dirpath is not dataset_path:
            # save the semantic label
            dirpath_comp = dirpath.split("/")
            semantic_label = dirpath_comp[-1]
            data["mapping"].append(semantic_label)
            print(f"Processing: {semantic_label}")
            
            # for each audio file in genre dir extract mfcc and store data
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                signal,sr = librosa.load(file_path,sr=SAMPLE_RATE)
                for s in range(num_segments):
                    start_sample = samples_ps * s
                    finish_sample = start_sample + samples_ps

                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                sr = sr,
                                                n_fft = n_fft,
                                                n_mfcc = n_mfcc,
                                                hop_length = hop_length)

                    mfcc = mfcc.T

                    # store mfcc for segment if it has expected length
                    if len(mfcc)==expected_vects_ps:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print(f"{file_path}, segment: {s+1}")

    # save data
    with open(json_path,"w") as f:
        json.dump(data,f,indent=4)


# save mfcc to json file
save_mfcc(dataset_path,json_path,num_segments=10)
clear_output()


# load data from json file
def load_data(dataset_path):
    with open(dataset_path,"r") as f:
        data = json.load(f)
    
    # convert lists to numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])    
    
    return inputs,targets

# load data from json file
inputs,targets = load_data(r"/Users/tyler/Desktop/music-genre-classification-cnn-master/output/data.json")

# split data into train and test sets
input_train, input_test, target_train, target_test = train_test_split(inputs, targets, test_size=0.3)
print(input_train.shape, target_train.shape)

# adam is a type of optimizer that is used to update the weights of the neural network
adam = optimizers.Adam(lr=1e-4)


# plot history of training process
def plot_history(hist):
    plt.figure(figsize=(20,15))
    fig, axs = plt.subplots(2)
    fig.tight_layout(pad=3.0)
    # accuracy subplot
    axs[0].plot(hist.history["accuracy"], label="train accuracy")
    axs[0].plot(hist.history["val_accuracy"], label="test accuracy")    
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy")
    
    # error subplot
    axs[1].plot(hist.history["loss"], label="train error")
    axs[1].plot(hist.history["val_loss"], label="test error")    
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error")
    
    plt.savefig("/Users/tyler/Desktop/music-genre-classification-cnn-master/pngNew/accuracy_chart.png")
    plt.show()


# prepare the dataset to be used by the CNN model
def prepare_dataset(test_size, validation_size):
    X,y = load_data(r"/Users/tyler/Desktop/music-genre-classification-cnn-master/output/data.json")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = validation_size)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_val, X_test, y_train, y_val, y_test


X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(0.25, 0.2)
input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3])
print(input_shape)


# build CNN
model = Sequential()
# 1st conv layer
model.add(Conv2D(64, (3, 3), activation = "relu", input_shape = input_shape))
model.add(MaxPool2D((3, 3), strides=(2, 2), padding="same"))
model.add(BatchNormalization())

# 2nd conv layer
model.add(Conv2D(32, (3, 3), activation = "relu"))
model.add(MaxPool2D((3, 3), strides=(2, 2), padding="same"))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# 3rd conv layer
model.add(Conv2D(32, (2, 2), activation = "relu"))
model.add(MaxPool2D((2, 2), strides=(2, 2), padding="same"))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# 4th conv layer
model.add(Conv2D(16, (1, 1), activation = "relu"))
model.add(MaxPool2D((1, 1), strides=(2, 2), padding="same"))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# flatten output and feed it into dense layer
model.add(Flatten())
model.add(Dense(64, activation="relu"))
# add dropout layer to prevent overfitting
model.add(Dropout(0.3))
# output layer with 10 nodes (one for each class)
model.add(Dense(10, activation="softmax"))

model.summary()

# compile CNN model with adam optimizer and sparse categorical cross entropy loss function
model.compile(optimizer=adam,
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# train CNN model with train data and test data for validation and 200 epochs (iterations)
hist = model.fit(X_train, y_train,
                 validation_data = (X_val, y_val),
                 epochs = 200)

plot_history(hist)

# evaluate model on test set and print accuracy score
test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test accuracy: {test_accuracy}")


model.evaluate(X_test, y_test, verbose=1)

# Heatmap filtered 
y_true = tf.argmax(y_test, axis=-1)
y_pred = tf.argmax(model.predict(X_test), axis=-1)

# Create a confusion matrix using seaborn
confusion_mtx = tf.math.confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(confusion_mtx, xticklabels=labels, yticklabels=labels, annot=True, fmt='g'
            , ax=ax, cmap="viridis", vmax=300)
ax.set_xlabel('Prediction')
ax.set_ylabel('Label')
ax.set_title('Confusion Matrix')
plt.savefig("/Users/tyler/Desktop/music-genre-classification-cnn-master/pngNew/cnn64_confmatrix.png")
plt.show()



# Print precision, recall, f1-score, support
print (metrics.classification_report(y_test, y_pred, target_names=labels))


# Saving the model for Future Inferences
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")


# predict function for a single sample
def predict(model, X, y):
    X = X[np.newaxis,...]
    prediction = model.predict(X)
    predicted_index = np.argmax(prediction, axis=1)
    print(f"Expected index: {y}, Predicted index: {predicted_index}")


predict(model, X_test[10], y_test[10])




