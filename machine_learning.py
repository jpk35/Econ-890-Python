"""
24 July 2023
Econ 890 session V: machine learning

required packages:
--pandas
--numpy
--keras
--matplotlib
--tensorflow
"""

import os
import pandas as pd
import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")   # set backend
import matplotlib.pyplot as plt
import random

debug = True

########################################################################################################################
########################################################################################################################
# EXAMPLE USED THROUGHOUT: Predicting authorship of text (paragraphs). Paragraphs in the text data are sourced from the
# following (roughly) Jazz Age novels:
# --The Great Gatsby (F. Scott Fitzgerald)
# --The Sun Also Rises (Ernest Hemingway)
# --The Heart is a Lonely Hunter (Carson McCullers)
# -- Look Homeward, Angel (Thomas Wolfe)
# -- You Can't Go Home Again (Thomas Wolfe)

# TASK: Binary Classification. Predict whether each paragraph was authored by Thomas Wolfe (1) or not (0).
########################################################################################################################
########################################################################################################################


########################################################################################################################
# Prepare text data to be read into a machine learning model
########################################################################################################################

# import data
root = r'C:\Users\jpkad\Dropbox\RA work 2022\DEAL\course_iv'
df = pd.read_csv(os.path.join(root, 'text_data_train.csv'))
df['text'] = df['text'].astype(str)

# separate into texts (paragraphs) and labels (authorship)
texts = list(df['text'])
labels = list(df['author_is_TW'])

# parameters
#############################################
# cut off paragraphs after 250 words
maxlen = 250

# train on 5000 samples
training_samples = 5000

# validate on 2000 samples
validation_samples = 2000

# consider only the top 10k words in the dataset
max_words = 10000

# embedding dimension
embedding_dim = 100

# tokenize the text and convert to numerical sequences
########################################################

# tokenize using keras built-in tokenizer
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)

# convert to sequences
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index  # save index mapping numbers to words
print('Found %s unique tokens.' % len(word_index))

# "pad" sequences so that each sequence has the same length
data = pad_sequences(sequences, maxlen=maxlen)

# convert labels (list) to a numpy array
labels = np.asarray(labels)

# examine shape of our data and labels
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
#############################################################

# first shuffle the data, because you're starting with data in which samples are ordered
# (all positive first, then all negative)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# training data
x_train = data[:training_samples]
y_train = labels[:training_samples]

# validation data
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

########################################################################################################################
# Now we're ready to train our first model!
########################################################################################################################

# MODEL DEFINITION
#########################################
# ML models are networks of layers. Each layer transforms the data. In each round of training, the model tries
# different weights for each parameter of each transformation, trying to learn useful representations of the data
# for the given task.

# Key inputs are:
# --how many layers to use
# --how many "hidden units" to use in each layer
# --what types of layers to use

# initialize a sequential model
model = Sequential()
# add an embedding layer (this means the model will "learn" an embedding from the input data; you can also use
# a pre-trained embedding model such as GloVe (https://nlp.stanford.edu/projects/glove/))
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
# flatten the 3D tensor of embeddings into a 2D tensor of shape (samples, maxlen*embedding_dim)
model.add(Flatten())
# include a Dense ("fully connected") layer with relu ("rectified linear unit") activation, one of the simplest building
# blocks of deep neural networks
model.add(Dense(32, activation='relu'))
# activation for final layer should be "sigmoid" for binary classification tasks (specifying output function so that
# most values are close to either 0 or 1)
model.add(Dense(1, activation='sigmoid'))
model.summary()

# training and evaluation
#########################################
# choices: optimizer (what algorithm you want the model to use for optimization [typically a version of gradient
# descent]), loss (the loss function the model will use to guide its updating), and metrics (what metrics should the
# model track, e.g., accuracy or "acc" for classification tasks)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc', 'Precision', 'Recall'])

# call model.fit to train the model; we can set the number of epochs (rounds of training, basically), the batch size
# (how much of the data to use in each training "step"), and validation data (not directly used in training, so this
# gives us a sense of when the model starts overfitting)

# capture the results from each training epoch in a variable we'll call "history"
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))

# Plot the results of the training
############################################
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='training acc')
plt.plot(epochs, val_acc, 'b', label='validation acc')
plt.title('training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('training and validation loss')
plt.legend()

########################################################################################################################
# Examine model performance on hold-out data
########################################################################################################################

# read in holdout data
df_holdout = pd.read_csv(os.path.join(root, 'text_data_holdout.csv'))
texts_holdout = list(df_holdout['text'])
labels_holdout = list(df_holdout['author_is_TW'])

# convert to sequences
holdout_sequences = tokenizer.texts_to_sequences(texts_holdout)

# "pad" sequences so that each sequence has the same length
holdout_data = pad_sequences(holdout_sequences, maxlen=maxlen)

# convert labels (list) to a numpy array
holdout_labels = np.asarray(labels_holdout)

# shuffle the data
indices = np.arange(holdout_data.shape[0])
np.random.shuffle(indices)
x_holdout = holdout_data[indices]
y_holdout = holdout_labels[indices]

# Show predictions for a random sample of holdout data
#######################################################
for i in range(5):
    indx = random.randint(0, len(x_holdout))
    print('text:', tokenizer.sequences_to_texts([x_holdout[indx]]))
    print('predicted label:', round(model.predict(np.array([x_holdout[indx]]))[0][0], 2))
    print('actual label:', y_holdout[indx])

# Evaluate model on the holdout data
#######################################################
holdout_results = model.evaluate(x_holdout, y_holdout)
print('loss on hold-out data:', holdout_results[0])
print('accuracy on hold-out data:', holdout_results[1])

# Now let's see if ChatGPT can trick the model
# Let's see how it labels text that was generated using the prompt "write a paragraph in the style of
# early 20th century author Thomas Wolfe"
#######################################################
# read in ChatGPT text
df_gpt = pd.read_csv(os.path.join(root, 'chatgpt_text.csv'))
texts_gpt = list(df_gpt['text'])
labels_gpt = list(df_gpt['author_is_TW'])

# convert to sequences
gpt_sequences = tokenizer.texts_to_sequences(texts_gpt)

# "pad" sequences so that each sequence has the same length
gpt_data = pad_sequences(gpt_sequences, maxlen=maxlen)

# convert labels (list) to a numpy array
gpt_labels = np.asarray(labels_gpt)

# shuffle the data
indices = np.arange(gpt_data.shape[0])
np.random.shuffle(indices)
x_gpt = gpt_data[indices]
y_gpt = gpt_labels[indices]

# Show predictions for ChatGPT texts
#######################################################
for indx in range(len(texts_gpt)):
    print('ChatGPT text:', tokenizer.sequences_to_texts([x_gpt[indx]]))
    print('predicted label:', round(model.predict(np.array([x_gpt[indx]]))[0][0], 2))
    print('actual label:', y_gpt[indx])

########################################################################################################################
# Now let's get a little fancier
# We'll add some bells and whistles, including using a CNN rather than a simpler linear model
########################################################################################################################

# re-import data
df = pd.read_csv(os.path.join(root, 'text_data_train.csv'))
df['text'] = df['text'].astype(str)

# separate into texts (paragraphs) and labels (authorship)
texts = list(df['text'])
labels = list(df['author_is_TW'])

# cut off paragraphs after 300 words
maxlen = 300

# consider the top 20k words in the dataset
max_words = 20000

# set higher embedding dimension (150)
embedding_dim = 150


# # tokenize the text and convert to numerical sequences
# ########################################################
# tokenize using keras built-in tokenizer, but this time don't filter out punctuation (potentially informative)

# helper function to add spaces before common punctuation, so that these symbols will be read as their own tokens
def separate_punctuation(txts):
    for t in range(len(txts)):
        txts[t] = txts[t].replace(".", " .")
        txts[t] = txts[t].replace("!", " !")
        txts[t] = txts[t].replace("?", " ?")
        txts[t] = txts[t].replace(":", " :")
        txts[t] = txts[t].replace(";", " ;")
        txts[t] = txts[t].replace(",", " ,")
        txts[t] = txts[t].replace("(", "( ")
        txts[t] = txts[t].replace(")", " )")
        txts[t] = txts[t].replace('"', ' " ')
    return txts


texts = separate_punctuation(texts)

tokenizer = Tokenizer(num_words=max_words,
                      filters='#$%*+-/<=>@[\\]^_`{|}~\t\n')
# tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)

# convert to sequences
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index  # save index mapping numbers to words
print('Found %s unique tokens.' % len(word_index))

# "pad" sequences so that each sequence has the same length
data = pad_sequences(sequences, maxlen=maxlen)

# convert labels (list) to a numpy array
labels = np.asarray(labels)

# split the data into a training set and a validation set
#############################################################

# first shuffle the data, because you're starting with data in which samples are ordered
# (all positive first, then all negative)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# training data
x_train = data[:training_samples]
y_train = labels[:training_samples]

# validation data
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

# MODEL DEFINITION
#########################################
from keras.layers import LSTM
from keras.layers import Dropout

# initialize a sequential model
model = Sequential()
# add an embedding layer
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
# add dropout layers between each subsequent layer (to help slow overfitting by introducing random noise)
model.add(Dropout(0.15))
# use an LSTM ("Long Short-Term Memory) layer rather than a Dense layer; this allows the model to have memory, turning
# it into an "RNN" (Recurrent Neural Network)... Another common layer type for this purpose is the "GRU" (Gated
# Recurrent Unit)
model.add(LSTM(32))
model.add(Dropout(0.15))
model.add(Dense(1, activation='sigmoid'))
if debug:
    model.summary()

# training and evaluation
#########################################

# compile model (now using Keras' Adam optimizer, widely considered the gold standard)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss='binary_crossentropy',
              metrics=['acc', 'Precision', 'Recall'])

# add an early stopping rule for training
# this will end training early if the chosen monitor (here, validation accuracy) stops improving
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc',
                                                  verbose=1,
                                                  patience=5,
                                                  mode='max',
                                                  restore_best_weights=True)

# train the model
history = model.fit(x_train, y_train,
                    epochs=25,
                    batch_size=32,
                    callbacks=[early_stopping],
                    validation_data=(x_val, y_val))

# Plot the results of the training
############################################
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='training acc')
plt.plot(epochs, val_acc, 'b', label='validation acc')
plt.title('training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('training and validation loss')
plt.legend()

plt.close()

########################################################################################################################
# Examine model performance on hold-out data
########################################################################################################################

# read in holdout data
df_holdout = pd.read_csv(os.path.join(root, 'text_data_holdout.csv'))
texts_holdout = list(df_holdout['text'])
labels_holdout = list(df_holdout['author_is_TW'])

# separate punctuation
texts_holdout = separate_punctuation(texts_holdout)

# convert to sequences
holdout_sequences = tokenizer.texts_to_sequences(texts_holdout)

# "pad" sequences so that each sequence has the same length
holdout_data = pad_sequences(holdout_sequences, maxlen=maxlen)

# convert labels (list) to a numpy array
holdout_labels = np.asarray(labels_holdout)

# shuffle the data
indices = np.arange(holdout_data.shape[0])
np.random.shuffle(indices)
x_holdout = holdout_data[indices]
y_holdout = holdout_labels[indices]

# Show predictions for a random sample of holdout data
#######################################################
for i in range(5):
    indx = random.randint(0, len(x_holdout))
    print('text:', tokenizer.sequences_to_texts([x_holdout[indx]]))
    print('predicted label:', round(model.predict(np.array([x_holdout[indx]]))[0][0], 2))
    print('actual label:', y_holdout[indx])

# Evaluate model on the holdout data
#######################################################
holdout_results = model.evaluate(x_holdout, y_holdout)
print('loss on hold-out data:', holdout_results[0])
print('accuracy on hold-out data:', holdout_results[1])

########################################################################################################################
# Let's retry the ChatGPT writing and see if this more
# sophisticated model performs better
########################################################################################################################
# read in ChatGPT text
df_gpt = pd.read_csv(os.path.join(root, 'chatgpt_text.csv'))
texts_gpt = list(df_gpt['text'])
labels_gpt = list(df_gpt['author_is_TW'])

# separate punctuation
texts_gpt = separate_punctuation(texts_gpt)

# convert to sequences
gpt_sequences = tokenizer.texts_to_sequences(texts_gpt)

# "pad" sequences so that each sequence has the same length
gpt_data = pad_sequences(gpt_sequences, maxlen=maxlen)

# convert labels (list) to a numpy array
gpt_labels = np.asarray(labels_gpt)

# shuffle the data
indices = np.arange(gpt_data.shape[0])
np.random.shuffle(indices)
x_gpt = gpt_data[indices]
y_gpt = gpt_labels[indices]

# Show predictions for ChatGPT texts
#######################################################
for indx in range(len(texts_gpt)):
    print('ChatGPT text:', tokenizer.sequences_to_texts([x_gpt[indx]]))
    print('predicted label:', round(model.predict(np.array([x_gpt[indx]]))[0][0], 2))
    print('actual label:', y_gpt[indx])