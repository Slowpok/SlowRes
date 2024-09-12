import pathlib
import pandas as pd
import pickle
import numpy as np
import keras
import tensorflow as tf

from tensorflow.keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras import utils
from sklearn.utils.class_weight import compute_class_weight
from configurations.parameters import read_configuration
from keras.callbacks import CSVLogger


def get_model_binary_resnet(unique_words, size_token):

    main_input = tf.keras.layers.Input(shape=size_token, name='main_input')
    x = tf.keras.layers.Embedding(unique_words, 32, input_length=size_token)(main_input)
    x = tf.keras.layers.Conv1D(125, 3, activation='relu')(x)
    x = tf.keras.layers.Conv1D(250, 3, activation='relu')(x)
    output_1 = tf.keras.layers.MaxPooling1D(3)(x)

    x = tf.keras.layers.Conv1D(250, 3, padding='same', activation='relu')(output_1)
    x = tf.keras.layers.Conv1D(250, 3, padding='same', activation='relu')(x)
    output_2 = tf.keras.layers.add([x, output_1])

    x = tf.keras.layers.Conv1D(250, 3, padding='same', activation='relu')(output_2)
    x = tf.keras.layers.Conv1D(250, 3, padding='same', activation='relu')(x)
    output_3 = tf.keras.layers.add([x, output_2])

    x = tf.keras.layers.Conv1D(250, 3, padding='same', activation='relu')(output_3)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

    model_result = tf.keras.Model(main_input, outputs, name='resnet')

    model_result.compile(optimizer='adam',
                      loss=tf.keras.losses.binary_focal_crossentropy,
                      metrics=[tf.keras.metrics.Accuracy(),
                               ])
    model_result.summary()

    return model_result


def train(element_id, parameters):

    train_data = pd.read_csv(parameters['full_data_name'],
                             header=None,
                             names=['title', 'class'])

    data_text = train_data['title']

    y_train = utils.to_categorical(train_data['class'] - 1, parameters['nb_classes'])

    tokenizer = Tokenizer(num_words=parameters['num_words'])

    tokenizer.fit_on_texts(data_text)

    with open(get_name_tokinizer_file(parameters), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    sequences = tokenizer.texts_to_sequences(data_text)

    x_train = pad_sequences(sequences, maxlen=parameters['max_len'])

    df = pd.DataFrame(range(1, parameters['nb_classes']), columns=['class'])
    df = df._append(pd.DataFrame(train_data['class'], columns=['class']), ignore_index=True)

    class_weights = compute_class_weight('balanced',
                                         classes=np.unique(df['class']),
                                         y=df['class'])
    class_weight = dict(enumerate(class_weights))

    model_cnn = Sequential()
    model_cnn.add(Embedding(parameters['num_words'], 32, input_length=parameters['max_len']))
    model_cnn.add(Conv1D(250, 5, padding='valid', activation='relu'))
    model_cnn.add(GlobalMaxPooling1D())
    model_cnn.add(Dense(128, activation='relu'))
    model_cnn.add(Dense(parameters['nb_classes'], activation='softmax'))

    model_cnn.compile(optimizer='adam',
                      loss=keras.losses.categorical_focal_crossentropy,
                      metrics=[keras.metrics.Accuracy(),
                               keras.metrics.CategoricalAccuracy(),
                               keras.metrics.AUC(),
                               keras.metrics.F1Score(average='weighted'),
                               keras.metrics.Recall(),
                               keras.metrics.Precision(),
                               keras.metrics.SpecificityAtSensitivity(sensitivity=0.5),
                               keras.metrics.TruePositives(thresholds=0.5),
                               keras.metrics.TrueNegatives(thresholds=0.5),
                               keras.metrics.FalsePositives(thresholds=0.5),
                               keras.metrics.FalseNegatives(thresholds=0.5),
                               ])

    csv_logger = CSVLogger(parameters['csv_log'])
    model_cnn_save_path = parameters['model_save_path']
    checkpoint_callback_cnn = ModelCheckpoint(model_cnn_save_path,
                                              monitor='val_accuracy',
                                              save_best_only=True,
                                              verbose=1)

    history_cnn = model_cnn.fit(x_train,
                                y_train,
                                epochs=parameters['epoch_count'],
                                batch_size=parameters['batch_size'],
                                #validation_split=0,
                                validation_split=parameters['validation_split'],
                                callbacks=[checkpoint_callback_cnn, csv_logger],
                                class_weight=class_weight,
                                verbose=0,)
                                #validation_data=(x_test, y_test))

