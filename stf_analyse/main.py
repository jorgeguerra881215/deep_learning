import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Embedding, LSTM

from sklearn import metrics
from sklearn.model_selection import train_test_split
from random import shuffle

def build_lstm(input_shape):
    model = Sequential()
    model.add(Embedding(20000, 128))

    model.add(LSTM(128, return_sequences=False))
    # Add dropout if overfitting
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_model(model, x_train, y_train, batch_size, checkpointer, epochs=20):
    print("Training model...")

    # FIT THE MODEL
    model.fit(x_train, y_train,
               batch_size = batch_size,
               epochs = epochs,
               verbose = 1,
               callbacks = [checkpointer],
               shuffle = True,
               validation_split=0.3
              )

def evaluate_model(model, x_test, y_test):
    score = model.evaluate(x_test, y_test,
                            batch_size=32
                           )
    return score

def test_model(model, x_test, y_test):
    model.predict(x_test, batch_size=32)
    test_preds = model.predict_classes(x_test, len(x_test), verbose=1)
    print ("Testing Dataset)\n", metrics.confusion_matrix(y_test, test_preds))




if __name__ == '__main__':
    data = pd.read_csv('data/data_all_result.txt', sep = ' ')
    data = data[['State', 'Label']]
    data = data[data['State'].str.len() > 3]
    data = data.replace(['Normal','Botnet'], [0,1])

    # How much attack are
    f, ax = plt.subplots(1, 2, figsize=(18, 8))
    data['Label'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
    ax[0].set_title('Attack')
    ax[0].set_ylabel('')
    sns.countplot('Label', data=data, ax=ax[1])
    ax[1].set_title('Attack')
    plt.show()

    #print('DONE')