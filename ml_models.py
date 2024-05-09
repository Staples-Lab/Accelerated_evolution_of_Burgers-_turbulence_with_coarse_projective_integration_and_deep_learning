import keras
import tensorflow as tf
# from tensorflow import keras
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input, BatchNormalization, \
    multiply, concatenate, Flatten, Activation, dot, Dropout, Masking
from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
from keras.layers import Input, LSTM, Dense, Lambda, Concatenate, Dot, Activation

class LearningRateLogger(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        if hasattr(lr, "value"):
            lr = lr.value()
        if hasattr(lr, "numpy"):
            lr = lr.numpy()
        print(f"\nEpoch {epoch + 1}: Current learning rate is {lr:.6f}")

lr_logger = LearningRateLogger()

reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_lr=1e-6)

def combined_mse_mae(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    mae = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    return mse + mae

def enc_dec_model():

    # Define hyperparameters
    n_units = 128  # Increased number of units in LSTM layers
    n_output_features = 8193
    dropout = 0.1

    l1_strength = 1e-7 #1e-6
    l2_strength = 1e-6 #1e-5

    # Define encoder model
    encoder_inputs = Input(shape=(None, X_input_train.shape[2]))
    encoder_lstm1 = LSTM(n_units, dropout=dropout, return_sequences=True,
                        kernel_regularizer=regularizers.l1_l2(l1=l1_strength, l2=l2_strength))(encoder_inputs)
    # encoder_lstm2 = LSTM(n_units, dropout=dropout, return_sequences=True,
    #                      kernel_regularizer=regularizers.l1_l2(l1=l1_strength, l2=l2_strength))(encoder_lstm1)
    # encoder_lstm3 = LSTM(n_units, dropout=dropout, return_sequences=True,
    #                      kernel_regularizer=regularizers.l1_l2(l1=l1_strength, l2=l2_strength))(encoder_lstm2)
    encoder_lstm4 = LSTM(n_units, dropout=dropout, return_sequences=True, return_state=True,
                        kernel_regularizer=regularizers.l1_l2(l1=l1_strength, l2=l2_strength))(encoder_lstm1)
    encoder_outputs, state_h, state_c = encoder_lstm4
    encoder_states = [state_h, state_c]

    # Define decoder model
    decoder_inputs = Input(shape=(None, Y_input_train.shape[2]))
    decoder_lstm1 = LSTM(n_units, dropout=dropout, return_sequences=True,
                        kernel_regularizer=regularizers.l1_l2(l1=l1_strength, l2=l2_strength))(decoder_inputs, initial_state=encoder_states)
    # decoder_lstm2 = LSTM(n_units, dropout=dropout, return_sequences=True,
    #                      kernel_regularizer=regularizers.l1_l2(l1=l1_strength, l2=l2_strength))(decoder_lstm1)
    # decoder_lstm3 = LSTM(n_units, dropout=dropout, return_sequences=True,
    #                      kernel_regularizer=regularizers.l1_l2(l1=l1_strength, l2=l2_strength))(decoder_lstm2)
    decoder_lstm4 = LSTM(n_units, dropout=dropout, return_sequences=True, return_state=True,
                        kernel_regularizer=regularizers.l1_l2(l1=l1_strength, l2=l2_strength))(decoder_lstm1)
    decoder_outputs, _, _ = decoder_lstm4

    # Add attention mechanism
    attention = Dot(axes=[2, 2])([decoder_outputs, encoder_outputs])
    attention = Activation('softmax')(attention)
    context = Dot(axes=[2, 1])([attention, encoder_outputs])
    decoder_combined_context = Concatenate(axis=-1)([context, decoder_outputs])

    # Add additional Dense layers before the output layer
    dense1 = TimeDistributed(Dense(4096, activation='relu'))(decoder_combined_context)
    dense2 = TimeDistributed(Dense(2048, activation='relu'))(dense1)
    dense3 = TimeDistributed(Dense(1024, activation='relu'))(dense2)
    dense4 = TimeDistributed(Dense(512, activation='relu'))(dense3)

    # Add output Dense layer with regularization
    output = TimeDistributed(Dense(n_output_features, activation='linear',
                                kernel_regularizer=regularizers.l1_l2(l1=l1_strength, l2=l2_strength)))(dense3)

    # Define the model
    model = Model([encoder_inputs, decoder_inputs], output)

    model.compile(optimizer='adam', loss=combined_mse_mae, metrics=['mse', 'mae'])

    X_train, X_val, Y_train_input, Y_val_input, Y_train_output, Y_val_output = train_test_split(
        X_input_train, Y_input_train, Y_output, test_size=0.2, shuffle=True)

    # Train the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=20)
    history = model.fit([X_train, Y_train_input], Y_train_output, epochs=150, batch_size=32, validation_data=([X_val, Y_val_input], Y_val_output), callbacks=[reduce_lr_on_plateau, lr_logger, es])

    return model, history