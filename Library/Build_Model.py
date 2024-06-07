###############################################################################
# This library provide utilities for buiding, training, evaluating, saving
# and loading models. The actual model is passed through the parameter
# 'model_type'. The library makes use of Keras, tensorfow and sklearn
# The provided models are:
# - ANN_dense: a simple Dense neural network
# - AMN_QP: a trainable QP solver using Gradient Descent
# - AMN_LP: a trainable LP solver of primal and dual LP from Y. Yang et al.
#           Mathematics and Computers in Simulation, 101 (2014) 103–112
# - AMN_Wt: a trainable RNN cell where V is updated with a weight matrix
# - MM_QP and MM_LP: non-trainable mechanistic model based on linear program
#   and gradient descent to compute all fluxes V when target objectives
#   are provided
# - RC: make use of trained AMNs (cf. previous module)
#   in reseroir computing (RC). The reservoir (non-trainable AMN)
#   is squized between two standard ANNs. The purpose of the prior ANN is to
#   transform problem features into nutrients added to media.
#   The post-ANN reads reservoir output (user predefined specific
#   reaction rates) and produce a readout to best match training set values.
#   Note that the two ANNs are trained but not the reservoir (AMN).
# Authors: Jean-loup Faulon, jfaulon@gmail.com and Bastien Mollet (LP model)
###############################################################################


from silence_tensorflow import silence_tensorflow
silence_tensorflow() # Tensorflow generates WARNINGS because of GPU unused, silence it

from Library.Build_Dataset import *
import keras.backend as K
import tensorflow as tf
print("Using tf version ", tf. __version__)
import keras
print("Using keras version ", keras. __version__)
tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'

# print(tf.config.list_physical_devices('GPU'))

# tf.debugging.set_log_device_placement(False)
# tf.device('/GPU:0')

from keras import initializers
from keras.models import Sequential
from keras.models import load_model
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input, Dense, LSTM, Dropout, Flatten, Activation
from keras.layers import Lambda, Reshape, multiply
from keras.layers import concatenate, add, subtract, dot
# from keras.wrappers.scikit_learn import KerasRegressor
from scikeras.wrappers import KerasRegressor
from keras.layers import Activation
# from keras.utils.generic_utils import get_custom_objects
# from keras.utils.generic_utils import CustomObjectScope
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.utils import CustomObjectScope
from keras.callbacks import EarlyStopping, CSVLogger

from sklearn import linear_model
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split

from keras_core import ops

###############################################################################
# WandB class for tracking
###############################################################################
import wandb
print("Using W&B version ", wandb.__version__)
from wandb.keras import WandbMetricsLogger, WandbEvalCallback, WandbCallback, WandbModelCheckpoint
wandb.login()

# # Implement your model prediction visualization callback
# class WandbClfEvalCallback(WandbEvalCallback):
#     def __init__(
#         self, validation_data, data_table_columns, pred_table_columns, num_samples=5
#     ):
#         super().__init__(data_table_columns, pred_table_columns)
#
#         self.x = validation_data[0]
#         self.y = validation_data[1]
#
#     def add_ground_truth(self, logs=None):
#         for idx, (image, label) in enumerate(zip(self.x, self.y)):
#             self.data_table.add_data(idx, wandb.Image(image), label)
#
#     def add_model_predictions(self, epoch, logs=None):
#         preds = self.model.predict(self.x, verbose=0)
#         preds = tf.argmax(preds, axis=-1)
#
#         table_idxs = self.data_table_ref.get_index()
#
#         for idx in table_idxs:
#             pred = preds[idx]
#             self.pred_table.add_data(
#                 epoch,
#                 self.data_table_ref.data[idx][0],
#                 self.data_table_ref.data[idx][1],
#                 self.data_table_ref.data[idx][2],
#                 pred,
#             )

###############################################################################
# Custom Loss function
###############################################################################
from tensorflow.keras.losses import Loss

NBR_CONSTRAINT = 3 # The number of contraints of the mechanistic models

def generate_loss_model(parameter, Vin, Vbf):
    # Create input layer
    batch_size = parameter.batch_size
    num_rxns   = len(parameter.reactions)

    yTrue = Input(shape=(batch_size, num_rxns), name="yTrue")
    yPred = Input(shape=(batch_size, num_rxns), name="yPred")
    Vin_in = Input(shape=Vin.shape, name="Vin")
    Vbf_in = Input(shape=Vbf.shape, name="Vbf")
    S_in = Input(shape=parameter.S.shape, name="S")
    Pin_in = Input(shape=parameter.Pin.shape, name="Pin")
    Pout_in = Input(shape=parameter.Pout.shape, name="Pout")

    # Apply custom operation using Lambda layer
    out_shape = (None, )
    # , output_shape=out_shape
    out = tf.keras.layers.Lambda(compute_loss_lambda) \
                    ((yTrue, yPred, Vbf_in, Vin_in, S_in, Pin_in, Pout_in))
    # Create model
    loss_model = tf.keras.models.Model(
                    inputs=[yTrue, yPred, Vbf_in, Vin_in, S_in, Pin_in, Pout_in],
                    outputs=out)

    return loss_model

# Custom loss function using model prediction
class CustomLoss(Loss):
    def __init__(self, loss_model, parameter):
        super().__init__()
        self.loss_model = loss_model
        self.parameter = parameter
        self.Vin  = None
        self.Vbf  = None
        self.end  = len(self.parameter.reactions)
        self.input_dim = self.parameter.input_dim

        self.S    = tf.expand_dims(
                        tf.convert_to_tensor(np.float32(parameter.S)),
                        axis=0)
        self.Pin  = tf.expand_dims(
                        tf.convert_to_tensor(np.float32(parameter.Pin)),
                        axis=0)
        self.Pout = tf.expand_dims(
                        tf.convert_to_tensor(np.float32(parameter.Pout)),
                        axis=0)

        # print("Vin ", self.Vin.shape, " Vbf ", self.Vbf.shape)
        print("S ", self.S.shape, " Pin ", self.Pin.shape)
        print("Pout ", self.Pout.shape)
        loss_model.summary()


    def call(self, y_true, y_pred):
        d0, d1 = \
            y_true.shape[0], y_true.shape[1]
        if d1 > self.input_dim:
            self.Vin = CROP(1, 0, self.input_dim)(y_true)
            self.Vbf = CROP(1, self.input_dim, d1-3)(y_true)
        else:
            self.Vin = tf.constant(1000, shape=(y_true.shape[0], self.input_dim))
            self.Vbf = tf.constant(1000, shape=(y_true.shape[0], self.end))



        print('before y_true ', y_true.shape)
        print('before y_pred ', y_pred.shape)
        prediction_index = self.end + NBR_CONSTRAINT
        y_true, y_pred = y_true[:,:self.end], y_pred[:,prediction_index:]
        print('after y_true ', y_true.shape)
        print('after y_pred ', y_pred.shape)
        # print(abc)


        self.Vin = tf.expand_dims(self.Vin, axis=0)
        self.Vbf = tf.expand_dims(self.Vbf, axis=0)
        y_pred   = tf.expand_dims(y_pred, axis=0)
        y_true   = tf.expand_dims(y_true, axis=0)
        y_true   = tf.convert_to_tensor(y_true)

        loss = compute_loss_lambda(y_true, y_pred, self.Vbf, self.Vin, self.S, self.Pin, self.Pout)

        print("loss type ", type(loss))
        return loss

def compute_loss_ops(y_true, y_pred, Vbf, Vin, S, Pin, Pout):
    # y_true, y_pred, Vbf, Vin, S, Pin, Pout = vals

    y_pred = ops.squeeze(y_pred)
    Pout = ops.squeeze(Pout)
    Vbf = ops.squeeze(Vbf)
    S = ops.squeeze(S)
    Vin = ops.squeeze(Vin)
    Pin = ops.squeeze(Pin)
    print(type(y_true))

    y_true = ops.convert_to_tensor(y_true)
    y_pred = ops.convert_to_tensor(y_pred)
    Pout   = ops.convert_to_tensor(Pout)
    Vbf    = ops.convert_to_tensor(Vbf)
    S      = ops.convert_to_tensor(S)
    Vin    = ops.convert_to_tensor(Vin)
    Pin    = ops.convert_to_tensor(Pin)

    num_rxns = y_true.shape[2]
    num_mets = S.shape[1]
    num_ins  = Pin.shape[1]
    print("y_true ", y_true.shape, " y_pred ", y_pred.shape)
    print("Vin ", Vin.shape, " Vbf ", Vbf.shape)
    print("S ", S.shape, " Pin ", Pin.shape)
    print("Pout ", Pout.shape)

    # print(abc)


    ## Loss_Vout_constraint
    PoutV = ops.matmul(y_pred, ops.transpose(Pout)) - Vbf
    # PoutV = ops.subtract(PoutV, Vbf)
    PoutV = tf.keras.activations.relu(PoutV)
    PoutV = ops.norm(PoutV, axis=1, keepdims=True) / tf.constant(num_rxns, dtype=tf.float32)

    # PoutV = ops.square(PoutV)
    # print("type L1 --> ", type(L1))
    # print(abc)
    # return L1

    ## Loss_SV
    SV = ops.matmul(y_pred, tf.transpose(S))
    SV = ops.norm(SV, axis=1, keepdims=True) / tf.constant(num_mets, dtype=tf.float32)

    ## Loss_Vin
    PinV = ops.matmul(y_pred, tf.transpose(Pin)) - Vin
    # tf.cast(tf.multiply(Vin, parameter.scaler), tf.float32)
    PinV = tf.keras.activations.relu(PinV)
    PinV = ops.norm(PinV, axis=1, keepdims=True) / tf.constant(num_ins, dtype=tf.float32)

    ## Loss_Vpos
    Vpos = tf.keras.activations.relu(-y_pred)
    Vpos = ops.norm(Vpos, axis=1, keepdims=True) / tf.constant(num_rxns, dtype=tf.float32) # rescaled


    L1 = PoutV
    L2 = SV
    L3 = PinV
    L4 = Vpos

    # square sum of L1, L2, L3, L4, L5
    L1 = ops.square(PoutV)
    L2 = ops.square(SV)
    L3 = ops.square(PinV)
    L4 = ops.square(Vpos)
    # # L5 = tf.math.square(L5)
    # print(f"L1: {L1} -- L2: {L2} -- L3: {L3} -- L4: {L4}")

    # L = tf.math.reduce_sum(tf.concat([L1, L2, L3, L4, L5], axis=1), axis=1)
    L = tf.math.reduce_sum(tf.concat([L1, L2, L3, L4], axis=1), axis=1)
    # divide by 4
    num_const = 4.0
    L = L / tf.constant(num_const, dtype=tf.float32)

    print("Loss shape ", L.shape)

    return L


def compute_loss_lambda(value):
    y_true, y_pred, Vbf, Vin, S, Pin, Pout = value
    y_pred = tf.squeeze(y_pred)
    Pout = tf.squeeze(Pout)
    Vbf = tf.squeeze(Vbf)
    S = tf.squeeze(S)
    Vin = tf.squeeze(Vin)
    Pin = tf.squeeze(Pin)
    print(type(y_true))

    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    Pout   = tf.convert_to_tensor(Pout)
    Vbf    = tf.convert_to_tensor(Vbf)
    S      = tf.convert_to_tensor(S)
    Vin    = tf.convert_to_tensor(Vin)
    Pin    = tf.convert_to_tensor(Pin)

    num_rxns = y_true.shape[2]
    num_mets = S.shape[1]
    num_ins  = Pin.shape[1]
    print("y_true ", y_true.shape, " y_pred ", y_pred.shape)
    print("Vin ", Vin.shape, " Vbf ", Vbf.shape)
    print("S ", S.shape, " Pin ", Pin.shape)
    print("Pout ", Pout.shape)

    ## Loss_Vout_constraint
    PoutV    = tf.linalg.matmul(y_pred, tf.transpose(Pout), b_is_sparse=True)  - Vbf
    PoutV = tf.keras.activations.relu(PoutV)
    PoutV = tf.norm(PoutV, axis=1, keepdims=True) / tf.constant(num_rxns, dtype=tf.float32)

    ## Loss_SV
    SV = tf.linalg.matmul(y_pred, tf.transpose(S), b_is_sparse=True)
    SV = tf.norm(SV, axis=1, keepdims=True) / tf.constant(num_mets, dtype=tf.float32)
    # SV = tf.math.divide_no_nan(SV, tf.constant(num_mets, dtype=tf.float32)) # rescaled

    ## Loss_Vin
    PinV = tf.linalg.matmul(y_pred, tf.transpose(Pin), b_is_sparse=True) - Vin
    # tf.cast(tf.multiply(Vin, parameter.scaler), tf.float32)
    PinV = tf.keras.activations.relu(PinV)
    PinV = tf.norm(PinV, axis=1, keepdims=True) / tf.constant(num_ins, dtype=tf.float32)
    # PinV = tf.math.divide_no_nan(PinV, tf.constant(num_ins, dtype=tf.float32))

    ## Loss_Vpos
    Vpos = tf.keras.activations.relu(-y_pred)
    Vpos = tf.norm(Vpos, axis=1, keepdims=True) / tf.constant(num_rxns, dtype=tf.float32) # rescaled


    # square sum of L1, L2, L3, L4, L5
    L1 = tf.math.square(PoutV)
    L2 = tf.math.square(SV)
    L3 = tf.math.square(PinV)
    L4 = tf.math.square(Vpos)
    # # L5 = tf.math.square(L5)
    # print(f"L1: {L1} -- L2: {L2} -- L3: {L3} -- L4: {L4}")

    L = tf.math.reduce_sum(tf.concat([L1, L2, L3, L4], axis=1), axis=1)
    # divide by 4
    num_const = 4.0
    # num_const = 1.0
    L = tf.math.divide_no_nan(L, tf.constant(num_const, dtype=tf.float32))
    print(L.shape)

    return L


def compute_loss(y_true, y_pred, Vbf, Vin, S, Pin, Pout):
    print(type(y_true))

    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    Pout   = tf.convert_to_tensor(Pout)
    Vbf    = tf.convert_to_tensor(Vbf)
    S      = tf.convert_to_tensor(S)
    Vin    = tf.convert_to_tensor(Vin)
    Pin    = tf.convert_to_tensor(Pin)

    print("y_true ", y_true.shape, " y_pred ", y_pred.shape)
    print("Vin ", Vin.shape, " Vbf ", Vbf.shape)
    print("S ", S.shape, " Pin ", Pin.shape)
    print("Pout ", Pout.shape)

    num_rxns = y_true.shape[1]
    num_mets = S.shape[1]
    num_ins  = Pin.shape[1]

    ## Loss_Vout_constraint
    PoutV    = tf.linalg.matmul(y_pred, tf.transpose(Pout), b_is_sparse=True)  - Vbf
    PoutV = tf.keras.activations.relu(PoutV)
    PoutV = tf.norm(PoutV, axis=1, keepdims=True) / tf.constant(num_rxns, dtype=tf.float32)

    ## Loss_SV
    SV = tf.linalg.matmul(y_pred, tf.transpose(S), b_is_sparse=True)
    SV = tf.norm(SV, axis=1, keepdims=True) / tf.constant(num_mets, dtype=tf.float32)
    # SV = tf.math.divide_no_nan(SV, tf.constant(num_mets, dtype=tf.float32)) # rescaled

    ## Loss_Vin
    PinV = tf.linalg.matmul(y_pred, tf.transpose(Pin), b_is_sparse=True) - Vin
    # tf.cast(tf.multiply(Vin, parameter.scaler), tf.float32)
    PinV = tf.keras.activations.relu(PinV)
    PinV = tf.norm(PinV, axis=1, keepdims=True) / tf.constant(num_ins, dtype=tf.float32)
    # PinV = tf.math.divide_no_nan(PinV, tf.constant(num_ins, dtype=tf.float32))

    ## Loss_Vpos
    Vpos = tf.keras.activations.relu(-y_pred)
    Vpos = tf.norm(Vpos, axis=1, keepdims=True) / tf.constant(num_rxns, dtype=tf.float32) # rescaled


    # square sum of L1, L2, L3, L4, L5
    L1 = tf.math.square(PoutV)
    L2 = tf.math.square(SV)
    L3 = tf.math.square(PinV)
    L4 = tf.math.square(Vpos)
    # # L5 = tf.math.square(L5)
    # print(f"L1: {L1} -- L2: {L2} -- L3: {L3} -- L4: {L4}")

    L = tf.math.reduce_sum(tf.concat([L1, L2, L3, L4], axis=1), axis=1)
    # divide by 4
    num_const = 4.0
    # num_const = 1.0
    L = tf.math.divide_no_nan(L, tf.constant(num_const, dtype=tf.float32))
    print(L.shape)

    return L


###############################################################################
# Custom functions for training
###############################################################################

def sharp_sigmoid(x):
    # Custom activation function
    return K.sigmoid(10000.0 * x)
get_custom_objects().update({'sharp_sigmoid': Activation(sharp_sigmoid)})

# source: https://medium.com/@Bloomore/how-to-write-a-custom-loss-function-with-additional-arguments-in-keras-5f193929f7a0
def biochem_loss(parameter):

    def loss(y_true, y_pred):

        end  = len(parameter.reactions)
        input_dim = parameter.input_dim
        d0, d1 = \
            y_true.shape[0], y_true.shape[1]
        if d1 > input_dim:
            Vin = CROP(1, 0, input_dim)(y_true)
            Vbf = CROP(1, input_dim, d1-3)(y_true)
        else:
            Vin = tf.constant(1000, shape=(y_true.shape[0], input_dim))
            Vbf = tf.constant(1000, shape=(y_true.shape[0], end))


        S    = tf.convert_to_tensor(np.float32(parameter.S))
        Pin  = tf.convert_to_tensor(np.float32(parameter.Pin))
        Pout = tf.convert_to_tensor(np.float32(parameter.Pout))

        prediction_index = end + NBR_CONSTRAINT
        y_true, y_pred = y_true[:,:end], y_pred[:,prediction_index:]

        # print("Vin ", Vin.shape, " Vbf ", Vbf.shape)
        print("S ", S.shape, " Pin ", Pin.shape)
        print("Pout ", Pout.shape)

        loss = compute_loss(y_true, y_pred, Vbf, Vin, S, Pin, Pout)
        print("loss type ", type(loss))
        return loss
        # end = len(parameter.reactions)
        # y_true, y_pred = y_true[:,:end], y_pred[:,:end]
        # print('y_true ', y_true.shape)
        # print('y_pred ', y_pred.shape)
        #
        # Vin, Vbf = paramVin, paramVbf
        # # print("x ", x.shape, " V ", V.shape)
        # print("Vin ", Vin.shape, " Vbf ", Vbf.shape)
        # print("S ", parameter.S.shape, " Pin ", parameter.Pin.shape)
        # print("Pout ", parameter.Pout.shape, " mediumbound ", parameter.mediumbound)
        # S     = tf.convert_to_tensor(np.float32(parameter.S))
        # Pin   = tf.convert_to_tensor(np.float32(parameter.Pin))
        # Pout  = tf.convert_to_tensor(np.float32(parameter.Pout))


        #
        # # # inp1 = tf.keras.layers.Input(shape=(None, 2, end), name="inp1")
        # yTrue = Input(shape=y_true.shape, name="yTrue")
        # yPred = Input(shape=y_pred.shape, name="yPred")
        # Vin_in = Input(shape=Vin.shape, name="Vin")
        # Vbf_in = Input(shape=Vbf.shape, name="Vbf")
        # S_in = Input(shape=S.shape, name="S")
        # Pin_in = Input(shape=Pin.shape, name="Pin")
        # Pout_in = Input(shape=Pout.shape, name="Pout")
        # print('Vbf ', Vbf.shape)
        # #
        # # out = tf.keras.layers.Lambda(add_numpy)((yTrue, yPred, Vin, Vbf, parameter))
        # out = tf.keras.layers.Lambda(add_numpy)((yTrue, yPred, Vbf_in, Vin_in, S_in, Pin_in, Pout_in))
        # # out = tf.keras.layers.Lambda(add_numpy_temp)((yTrue, yPred, Vbf_in, Vin_in, S_in, Pin_in, Pout_in))
        # #
        # # mdl = tf.keras.Model(input=[yTrue, yPred], output=out)
        # mdl = tf.keras.models.Model(inputs=[yTrue, yPred, Vbf_in, Vin_in, S_in, Pin_in, Pout_in], outputs=out)
        # # mdl = tf.keras.models.Model(inputs=[inp1, inp2], outputs=out)
        # mdl.summary()
        # y_true = tf.expand_dims(y_true, axis=0)
        # y_pred = tf.expand_dims(y_pred, axis=0)
        # Vbf = tf.expand_dims(Vbf, axis=0)
        # Vin = tf.expand_dims(Vin, axis=0)
        # S = tf.expand_dims(S, axis=0)
        # Pout = tf.expand_dims(Pout, axis=0)
        # Pin = tf.expand_dims(Pin, axis=0)
        # L =  mdl([y_true, y_pred, Vbf, Vin, S, Pin, Pout]) #mdl([y_true, y_pred])
        # print(L.shape)
        # # print(bac)
        # # mdl(input=[y_true, y_pred], output=[out])
        # # return mdl([y_true, y_pred])
        # # return keras.losses.mean_squared_error(y_true[:,:end], y_pred[:,:end])
        # return L

    return loss

def my_mse(y_true, y_pred):
    # Custom loss function
    end = y_true.shape[1]
    print('--- my MSE:')
    print('y_true: ', y_true[:,:end].shape)
    print('y_pred: ', y_pred[:,:end].shape)
    print('--- --- ---')
    return keras.losses.mean_squared_error(y_true[:,:end], y_pred[:,:end])

def my_mae(y_true, y_pred):
    # Custom metric function
    end = y_true.shape[1]
    return keras.losses.mean_squared_error(y_true[:,:end], y_pred[:,:end])

def my_r2(y_true, y_pred):
    # Custom metric function
    end = y_true.shape[1]
    yt, yp = y_true[:,:end], y_pred[:,:end]
    SS =  K.sum(K.square( yt-yp ))
    ST = K.sum(K.square( yt - K.mean(yt) ) )
    return 1 - SS/(ST + K.epsilon())

def my_binary_crossentropy(y_true, y_pred):
    # Custom loss function
    end = y_true.shape[1]
    return keras.losses.binary_crossentropy(y_true[:,:end], y_pred[:,:end])

def my_acc(y_true, y_pred):
    # Custom metric function
    end = y_true.shape[1]
    return keras.metrics.binary_accuracy(y_true[:,:end], y_pred[:,:end])

def CROP(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call x = crop(2,5,10)(x) to slice the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)

###############################################################################
# Custom Loss functions to evaluate models and compute gradients
# Inputs:
# - V: the (predicted) flux vector
# - Pout: the matrix selecting in V measured outgoing fluxes
# - Vout: the measured outgoing fluxes
# - Pin: the matrix selecting in V measured incoming fluxes
# - Vin: the measured incoming fluxes
# - S: the stoichiometric matrix
# Outputs:
# - Loss and gradient
###############################################################################

def Loss_Vout(V, Pout, Vout, gradient=False):
    # Loss for the objective (match Vout)
    # Loss = ||Pout.V-Vout||
    # When Vout is empty just compute Pout.V
    # dLoss = ∂([Pout.V-Vout]^2)/∂V = Pout^T (Pout.V - Vout)
    # print("in Loss_Vout: ", V.shape, Pout.shape, Vout.shape)
    if not tf.is_tensor(Pout):
        Pout = tf.convert_to_tensor(np.float32(Pout))
    # print(Pout, Vout, V)
    Loss = tf.linalg.matmul(V, tf.transpose(Pout), b_is_sparse=True) - Vout
    Loss_norm = tf.norm(Loss, axis=1, keepdims=True)/Pout.shape[0] # rescaled
    if gradient:
        dLoss = tf.linalg.matmul(Loss, Pout, b_is_sparse=True) # derivate
        dLoss = dLoss / (Pout.shape[0] * Pout.shape[0])  # rescaling
        # dLoss = 2 * dLoss
    else:
        dLoss =  0 * V

    return Loss_norm, dLoss

def Loss_Vout_obj(V, parameter=None, gradient=False):
    # Loss for the objective (match Vout)
    # Loss = ||Pout.V-Vout||
    # When Vout is empty just compute Pout.V
    # dLoss = ∂([Pout.V-Vout]^2)/∂V = Pout^T (Pout.V - Vout)
    # print("in Loss_Vout: ", V.shape, Pout.shape, Vout.shape)
    Pout = tf.convert_to_tensor(np.float32(parameter.objPout))
    Vout = tf.convert_to_tensor(np.float32(parameter.objY))
    # print(Pout, Vout, V)

    # np.savetxt("temp_mult.csv", temp, delimiter=',')
    Loss = tf.linalg.matmul(V, tf.transpose(Pout), b_is_sparse=True) - Vout
    Loss_norm = tf.norm(Loss, axis=1, keepdims=True)/Pout.shape[0] # rescaled

    # np.savetxt("temp_Loss.csv", Loss, delimiter=',')
    # np.savetxt("temp_LossScaled.csv", Loss_norm, delimiter=',')
    # print(abc)
    if gradient:
        dLoss = tf.linalg.matmul(Loss, Pout, b_is_sparse=True) # derivate
        dLoss = dLoss / (Pout.shape[0] * Pout.shape[0])  # rescaling
        # dLoss = 2 * dLoss
    else:
        dLoss =  0 * V

    return Loss_norm, dLoss

def Loss_Vout_obj_constraint(V, parameter=None, gradient=False):
    # Loss for the objective (match Vout)
    # Loss = ||Pout.V-Vout||
    # When Vout is empty just compute Pout.V
    # dLoss = ∂([Pout.V-Vout]^2)/∂V = Pout^T (Pout.V - Vout)
    # print("in Loss_Vout: ", V.shape, Pout.shape, Vout.shape)
    Pout = tf.convert_to_tensor(np.float32(parameter.objPout))
    Vout = tf.convert_to_tensor(np.float32(parameter.objY))
    # print(Pout, Vout, V)

    # np.savetxt("temp_mult.csv", temp, delimiter=',')
    Loss = tf.linalg.matmul(V, tf.transpose(Pout), b_is_sparse=True) - Vout
    Loss_norm = tf.norm(Loss, axis=1, keepdims=True)/Pout.shape[0] # rescaled

    # np.savetxt("temp_Loss.csv", Loss, delimiter=',')
    # np.savetxt("temp_LossScaled.csv", Loss_norm, delimiter=',')
    # print(abc)
    if gradient:
        dLoss = tf.linalg.matmul(Loss, Pout, b_is_sparse=True) # derivate
        dLoss = dLoss / (Pout.shape[0] * Pout.shape[0])  # rescaling
        # dLoss = 2 * dLoss
    else:
        dLoss =  0 * V

    return Loss_norm, dLoss

def Loss_Vout_constraint(V, Pout, Vout, gradient=False):
    # Loss for the objective (acts like a boundary constraint)
    # Loss = ReLU(Pout . V - Vout)
    # When Vout is empty just compute Pout.V
    # dLoss = ∂(ReLU(Pout . V - Vout)^2/∂V
    # print("in Loss_Vout: ", V.shape, Pout.shape, Vout.shape)
    if not tf.is_tensor(Pout):
        Pout = tf.convert_to_tensor(np.float32(Pout))
    Loss = tf.linalg.matmul(V, tf.transpose(Pout), b_is_sparse=True) - Vout
    Loss = tf.keras.activations.relu(Loss)
    print(Loss.shape)
    Loss_norm = tf.norm(Loss, axis=1, keepdims=True)/Pout.shape[0] # rescaled
    if gradient:
        # dLoss = tf.linalg.matmul(Loss, Pout, b_is_sparse=True) # derivate
        # dLoss = dLoss / (Pout.shape[0] * Pout.shape[0])  # rescaling
        dLoss = tf.math.divide_no_nan(Loss, Loss) # derivate: Hadamard div.
        dLoss = tf.math.multiply(Loss, dLoss) # !!!
        dLoss = tf.linalg.matmul(dLoss, Pout, b_is_sparse=True) # resizing
        dLoss = dLoss / (Pout.shape[0] * Pout.shape[0])   # rescaling
        # dLoss = 2 * dLoss
    else:
        dLoss =  0 * V

    return Loss_norm, dLoss

def Loss_Vin(V, Pin, Vin, bound='UB', gradient=False):
    # Gradient for input boundary constraint
    # Loss = ReLU(Pin . V - Vin)
    # dLoss = ∂(ReLU(Pin . V - Vin)^2/∂V
    # Input: Cf. Gradient_Descent
    if not tf.is_tensor(Pin):
        Pin  = tf.convert_to_tensor(np.float32(Pin))

    Loss = tf.linalg.matmul(V, tf.transpose(Pin), b_is_sparse=True) - Vin
    # tf.cast(tf.multiply(Vin, parameter.scaler), tf.float32)
    Loss = tf.keras.activations.relu(Loss) if bound == 'UB' else Loss
    Loss_norm = tf.norm(Loss, axis=1, keepdims=True)/Pin.shape[0] # rescaled
    if gradient:
        dLoss = tf.math.divide_no_nan(Loss, Loss) # derivate: Hadamard div.
        dLoss = tf.math.multiply(Loss, dLoss) # !!!
        dLoss = tf.linalg.matmul(dLoss, Pin, b_is_sparse=True) # resizing
        dLoss = dLoss / (Pin.shape[0] * Pin.shape[0])   # rescaling
    else:
        dLoss =  0 * V
    return Loss_norm, dLoss

def Loss_SV(V, S, gradient=False):
    # Gradient for SV constraint
    # Loss = ||SV||
    # dLoss =  ∂([SV]^2)/∂V = S^T SV
    if not tf.is_tensor(S):
        S  = tf.convert_to_tensor(np.float32(S))
    Loss = tf.linalg.matmul(V, tf.transpose(S), b_is_sparse=True)
    Loss_norm = tf.norm(Loss, axis=1, keepdims=True)/S.shape[0] # rescaled
    if gradient:
        dLoss = tf.linalg.matmul(Loss, S, b_is_sparse=True) # derivate
        dLoss = dLoss / (S.shape[0]*S.shape[0])  # rescaling
        dLoss = dLoss / 2
    else:
        dLoss =  0 * V
    return Loss_norm, dLoss

def Loss_Vpos(V, Vlb, gradient=False):
    # Gradient for V ≥ 0 constraint
    # Loss = ReLU(-V)
    # dLoss = ∂(ReLU(-V)^2/∂V
    # print(Vlb.shape)
    if (Vlb is not None) and (Vlb.shape[0] is not None) and (Vlb.shape[0] >= 1) \
            and (Vlb.shape[1] == V.shape[1]):
        # print("lower bound loss")
        Loss = tf.keras.activations.relu(Vlb - V)
    else:
        Loss = tf.keras.activations.relu(-V)

    Loss_norm = tf.norm(Loss, axis=1, keepdims=True)/V.shape[1] # rescaled
    if gradient:
        dLoss = - tf.math.divide_no_nan(Loss, Loss) # derivate: Hadamard div.
        dLoss = tf.math.multiply(Loss, dLoss) # !!!
        dLoss = dLoss / (V.shape[1] * V.shape[1]) # rescaling
    else:
        dLoss =  0 * V
    return Loss_norm, dLoss

def Loss_constraint(V, Vin, Vlb, parameter, gradient=False):
    # mean squared sum L2+L3+L4
    L2, dL2 = Loss_SV(V, parameter.S, gradient=gradient)
    L3, dL3 = Loss_Vin(V, parameter.Pin, Vin,
                       parameter.mediumbound, gradient=gradient)
    L4, dL4 = Loss_Vpos(V, Vlb, gradient=gradient)
    # square sum of L2, L3, L4
    L2 = tf.math.square(L2)
    L3 = tf.math.square(L3)
    L4 = tf.math.square(L4)
    L = tf.math.reduce_sum(tf.concat([L2, L3, L4], axis=1), axis=1)
    # divide by 3
    L = tf.math.divide_no_nan(L, tf.constant(3.0, dtype=tf.float32))
    return L, dL2+dL3+dL4

def Loss_all5(V, Vin, Vout, Vlb, parameter, gradient=False, p_sv=1):
    # mean square sum of L1, L2, L3, L4, L5
    if Vout.shape[0] < 1: # No target provided = no Loss_Vout
        L, dL = Loss_constraint(V, Vin, Vlb, parameter, gradient=gradient)
        return L, dL
    # np.savetxt("temp_Vout.csv", Vout, delimiter=',')
    # np.savetxt("temp_Pout.csv", parameter.Pout, delimiter=',')
    # np.savetxt("temp_V.csv", V, delimiter=',')
    L1, dL1 = Loss_Vout(V, parameter.Pout, Vout, gradient=gradient)
    # L1, dL1 = Loss_Vout_constraint(V, parameter.Pout, Vout, gradient=gradient)
    # print("L1, dL1: ", L1, dL1)
    L2, dL2 = Loss_SV(V, parameter.S, gradient=gradient)
    # L2 = tf.multiply(L2, 10.0)
    # apply penalty
    dL2 = dL2 * p_sv
    L2 = L2 * p_sv
    # print("L2, dL2: ", L2, dL2)
    L3, dL3 = Loss_Vin(V, parameter.Pin, Vin,
                       parameter.mediumbound, gradient=gradient)
    # print("L3, dL3: ", L3, dL3)
    L4, dL4 = Loss_Vpos(V, Vlb, gradient=gradient)
    # print("L4, dL4: ", L4, dL4)
    L5, dL5 = Loss_Vout_obj(V, parameter=parameter, gradient=gradient)
    # print("L5, dL5: ", L5, dL5)

    # print(abc)

    # square sum of L1, L2, L3, L4, L5
    L1 = tf.math.square(L1)
    L2 = tf.math.square(L2)
    L3 = tf.math.square(L3)
    L4 = tf.math.square(L4)
    L5 = tf.math.square(L5)
    # print(f"L1: {L1} -- L2: {L2} -- L3: {L3} -- L4: {L4} -- L5: {L5}")

    L = tf.math.reduce_sum(tf.concat([L1, L2, L3, L4, L5], axis=1), axis=1)
    # divide by 4

    num_const = 5.0
    L = tf.math.divide_no_nan(L, tf.constant(num_const, dtype=tf.float32))
    return L, dL1+dL2+dL3+dL4+dL5

def Loss_all(V, Vin, Vout, Vlb, parameter, gradient=False, wt=False):
    # mean square sum of L1, L2, L3, L4, L5
    if (Vout is None) or ((Vout.shape[0] is not None) and (Vout.shape[0] < 1)): # No target provided = no Loss_Vout
        L, dL = Loss_constraint(V, Vin, Vlb, parameter, gradient=gradient)
        return L, dL
    # np.savetxt("temp_Vout.csv", Vout, delimiter=',')
    # np.savetxt("temp_Pout.csv", parameter.Pout, delimiter=',')
    # np.savetxt("temp_V.csv", V, delimiter=',')
    # L1, dL1 = Loss_Vout(V, parameter.Pout, Vout, gradient=gradient)
    L1, dL1 = Loss_Vout_constraint(V, parameter.Pout, Vout, gradient=gradient)
    # print("L1, dL1: ", L1, dL1)
    L2, dL2 = Loss_SV(V, parameter.S, gradient=gradient)
    # print("L2, dL2: ", L2, dL2)
    # apply penalty
    dL2 = dL2 * p_sv
    L2 = L2 * p_sv

    L3, dL3 = Loss_Vin(V, parameter.Pin, Vin,
                       parameter.mediumbound, gradient=gradient)
    # print("L3, dL3: ", L3, dL3)
    L4, dL4 = Loss_Vpos(V, Vlb, gradient=gradient)
    # L2 = tf.multiply(L2, 10.0)
    # print("L4, dL4: ", L4, dL4)
    # L5, dL5 = Loss_Vout_obj(V, parameter=parameter, gradient=gradient)
    # # print("L5, dL5: ", L5, dL5)

    # print(abc)

    # square sum of L1, L2, L3, L4, L5
    L1 = tf.math.square(L1)
    L2 = tf.math.square(L2)
    L3 = tf.math.square(L3)
    L4 = tf.math.square(L4)
    # L5 = tf.math.square(L5)
    # print(f"L1: {L1} -- L2: {L2} -- L3: {L3} -- L4: {L4}")

    # L = tf.math.reduce_sum(tf.concat([L1, L2, L3, L4, L5], axis=1), axis=1)
    L = tf.math.reduce_sum(tf.concat([L1, L2, L3, L4], axis=1), axis=1)
    # divide by 4
    num_const = 4.0
    L = tf.math.divide_no_nan(L, tf.constant(num_const, dtype=tf.float32))
    # if wt:
    #     L = np.mean(L.numpy())

    return L, dL1+dL2+dL3+dL4

def custom_ReLU(V, Vout, Pout):
    # Vpos = tf.zeros(V.shape, dtype=tf.dtypes.float32)
    V = tf.keras.activations.relu(V)
    Pout_i = np.identity(Pout.shape[0])
    if not tf.is_tensor(Pout):
        Pout = tf.convert_to_tensor(np.float32(Pout))
        Pout_i = tf.convert_to_tensor(np.float32(Pout_i))
    V1 = Vout - tf.linalg.matmul(V, tf.transpose(Pout))
    # print(V1.shape)
    V1 = Vout - tf.keras.activations.relu(V1)
    # print(V1.shape)
    V2 = Pout_i - Pout
    # print(V2.shape)
    V2 = tf.linalg.matmul(V, V2)
    # print(V2.shape)
    Vpos = V1 + V2
    return Vpos

###############################################################################
# Dense model
###############################################################################

def input_ANN_Dense(parameter, verbose=False):
    # Shape X and Y depending on the model used
    if parameter.scaler != 0: # Normalize X
        parameter.X, parameter.scaler = MaxScaler(parameter.X)
    if verbose:
        print('ANN Dense scaler', parameter.scaler)
    return parameter.X, parameter.Y

def Dense_layers(inputs, parameter, trainable=True, verbose=False):
    # Build a dense architecture with some hidden layers

    activation=parameter.activation
    n_hidden=parameter.n_hidden
    dropout=parameter.dropout
    hidden_dim=parameter.hidden_dim
    output_dim=parameter.output_dim
    hidden = inputs
    n_hidden = 0 if hidden_dim == 0 else n_hidden
    for i in range(n_hidden):
        hidden = Dense(hidden_dim,
                       kernel_initializer='random_normal',
                       bias_initializer='zeros',
                       activation='relu', trainable=trainable) (hidden)
        hidden = Dropout(dropout)(hidden)
    if verbose:
        print('Dense layer n_hidden, hidden_dim, output_dim, activation, trainable:', \
              n_hidden, hidden_dim, output_dim, activation, trainable)
    outputs = Dense(output_dim,
                    kernel_initializer='random_normal',
                    bias_initializer='zeros',
                    activation=activation, trainable=trainable) (hidden)
    return outputs

def ANN_Dense(parameter, trainable=True, verbose=False):
    # A standard Dense model with several layers

    input_dim, output_dim = parameter.input_dim, parameter.output_dim
    inputs = Input(shape=(input_dim,))
    outputs = Dense_layers(inputs, parameter,
                           trainable=trainable, verbose=verbose)
    model = keras.models.Model(inputs=[inputs], outputs=[outputs])
    loss = 'mse' if parameter.regression else 'binary_crossentropy'
    metrics = ['mae'] if parameter.regression else ['acc']
    model.compile(loss=loss, optimizer='adam', metrics=metrics)
    if verbose == 2: print(model.summary())
    print('nbr parameters:', model.count_params())
    parameter.model = model

    return parameter

###############################################################################
# AMN models (1)
# AMN_QP: a ANN_Dense trainable prior layer and a mechanistic layer
# making use of gradient descent
###############################################################################

def input_AMN(parameter, verbose=False):
    # Shape the IOs
    # IO: X and Y
    # For all
    # - add additional zero columns to Y
    #   the columns are used to minimize SV, Pin V ≤ Vin, V ≥ 0
    # For AMN_LP: add b_int or b_ext
    # For AMN_Wt repeat X timestep times

    np.savetxt("Result/param_X.tsv", parameter.X, delimiter='\t')
    np.savetxt("Result/param_Y.tsv", parameter.Y, delimiter='\t')
    # print(abc)

    X, Y = parameter.X, parameter.Y
    if parameter.scaler != 0: # Normalize X
        X, parameter.scaler = MaxScaler(X)
    if verbose: print('AMN scaler', parameter.scaler)
    y = np.zeros(Y.shape[0]).reshape(Y.shape[0],1)
    Y = np.concatenate((Y, y), axis=1) # SV constraint
    Y = np.concatenate((Y, y), axis=1) # Pin constraint
    Y = np.concatenate((Y, y), axis=1) # V ≥ 0 constraint
    if 'QP' in parameter.model_type:
        if verbose: print('QP input shape',X.shape,Y.shape)
    elif 'RC' in parameter.model_type:
        if verbose: print('RC input shape',X.shape,Y.shape)
    elif 'LP' in parameter.model_type:
        # we add b_int and b_ext
        x = parameter.b_int
        x = np.vstack([x]*X.shape[0]) if len(x.shape) == 1 else x
        b_int = np.copy(x)
        X = np.concatenate((X, x), axis=1)
        x = parameter.b_ext
        x = np.vstack([x]*X.shape[0]) if len(x.shape) == 1 else x
        b_ext = np.copy(x)
        X = np.concatenate((X, x), axis=1)
        if parameter.mediumbound == 'UB' or "MM" in parameter.model_type:
            # print("NOT switching b_int and b_ext")
            parameter.b_int, parameter.b_ext = b_int, b_ext
        else:
            # EB case, but does not apply to MM notebooks
            # print("switching b_int and b_ext")
            parameter.b_int, parameter.b_ext = b_ext, b_int
            # parameter.b_int, parameter.b_ext = b_int, b_ext
        if verbose: print('LP input shape', X.shape,Y.shape)
    elif 'Wt' in parameter.model_type:
        x = np.copy(X)
        num_batches = int(x.shape[0]/parameter.batch_size)
        X = np.zeros((parameter.batch_size * num_batches,
                  parameter.timestep, x.shape[1]))
        for i in range(x.shape[0]):
            for j in range(parameter.timestep):
                X[i][j] = x[i]
        if verbose: print('Wt input shape', X.shape, Y.shape)
    else:
        print(parameter.model_type)
        sys.exit('This AMN type does not have input')
    parameter.input_dim = parameter.X.shape[1]

    return X, Y

def output_AMN(V, Vin, V0, Vlb, parameter, verbose=False):
    # Get output for all AMN models
    # output = PoutV + constaints = [SV + PinV + Relu(_V)] + V
    # where S and Pout are the stoichiometric and measurement matrix

    Pout     = tf.convert_to_tensor(np.float32(parameter.Pout))
    PoutV    = tf.linalg.matmul(V, tf.transpose(Pout), b_is_sparse=True)
    SV, _    = Loss_SV(V, parameter.S) # SV const
    PinV, _  = Loss_Vin(V, parameter.Pin, Vin, parameter.mediumbound) # Pin const
    Vpos, _  = Loss_Vpos(V, Vlb) # V ≥ 0 const

    # Return outputs = PoutV + SV + PinV + Vpos + V
    if V0 == None:
        print("output_AMN --")
        print("part1 PoutV -- ", PoutV.shape)
        print("part1 SV -- ", SV.shape)
        print("part1 PinV -- ", PinV.shape)
        print("part1 V -- ", V.shape)
        outputs = concatenate([PoutV, SV, PinV, Vpos, V], axis=1)
    else:
        outputs = concatenate([PoutV, SV, PinV, Vpos, V, V0], axis=1)
    parameter.output_dim = outputs.shape[1]
    if verbose:
        print('AMN output shapes for PoutV, SV, PinV, Vpos, V, outputs', \
              PoutV.shape, SV.shape, PinV.shape, Vpos.shape,\
              V.shape, outputs.shape)

    return outputs

def Gradient_Descent(V, Vin, Vout, Vlb, parameter, mask, trainable=True, history=False, verbose=False):
    # Input:
    # S [m x n]: stoichiometric matrix
    # V [n]: the reaction flux vector
    # Pin [n_in x n]: the flux to medium projection matrix
    # Vin [p]: the medium intake flux vector
    # V_out [n_out]: the measured fluxes (can be empty)
    # mask [n]: used to uddate dL
    # history: to specify if loss is computed and recorded
    # Output: Loss and updated V

    # Not history here if trainable
    history = False if trainable else history

    name = parameter.trainingfile.split('UB_')[1]
    wandb.init(
        # Set the project where this run will be logged
        project="omic_amn",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=name,
        # Track hyperparameters and run metadata
        config={
        "learning_rate": parameter.learn_rate,
        "decay_rate": parameter.decay_rate,
        "architecture": "MM_QP",
        "dataset": "QPSI",
        "epochs": parameter.timestep,
        })

    # GD loop
    Loss_mean_history, Loss_std_history, diff = [], [], 0 * V
    for t in range(1, parameter.timestep+1):  # Update V with GD
        # Get Loss and gradient
        L, dL = Loss_all(V, Vin, Vout, Vlb, parameter, gradient=True, p_sv = 2)
        dL = tf.math.multiply(dL, mask) # Apply mask on dL
        # Update V with learn and decay rates
        diff = parameter.decay_rate * diff - parameter.learn_rate * dL
        V = V + diff
        # V = np.maximum.reduce([V,Vout])
        # V = tf.keras.activations.relu(V)
        V = custom_ReLU(V, Vout, parameter.Pout)
        # print(abc)
        # Compile Loss history
        if history:
            Loss_mean, Loss_std = np.mean(L), np.std(L)
            Loss_mean_history.append(Loss_mean)
            Loss_std_history.append(Loss_std)
            # if verbose and (np.log10(t) == int(np.log10(t)) \
            #                 or t/1.0e3 == int(t/1.0e3)):
            if verbose and (t/1.0e3 == int(t/1.0e3)):

                print('QP-Loss', t, Loss_mean, Loss_std, ' -> bio flux ', V[:, parameter.bio_id])
                # log metrics using wandb.log
                report_dict = {f'{parameter.treatments[i]}_bio':V[i, parameter.bio_id] \
                                    for i in range(len(parameter.treatments))}
                report_dict['epochs'] = t
                report_dict['loss']   = Loss_mean
                wandb.log(report_dict)

    wandb.finish()
    return V, Loss_mean_history, Loss_std_history

def get_V0(inputs, parameter, targets, lower_bounds, trainable, verbose=False):
    # Get initial vector V0 from input and target
    # Return V0, Vin, Vout, mask
    # When target is not provided this function compute
    # the initial vector V0 using Dense_Layers

    Pin = tf.convert_to_tensor(np.float32(parameter.Pin))
    if targets.shape[0] > 0: # Initialize AMN when targets provided
        # Vin = inputs, V0 = (Pin)^T Vin
        Vin = inputs # tf.cast(tf.multiply(inputs, parameter.scaler), tf.float32)
        V0 = tf.linalg.matmul(inputs, Pin, b_is_sparse=True)
    else: # Initialize AMN when targets not provided
        # Vin = inputs, V0 = Dense_layers(inputs)
        param = copy.copy(parameter)
        param.output_dim = parameter.S.shape[1]
        param.activation = 'relu'
        Vin = inputs # tf.cast(tf.multiply(inputs, parameter.scaler), tf.float32)
        V0 = Dense_layers(inputs, param,
                          trainable=trainable, verbose=verbose)

    # Get a mask for EB and UB where elements in Vin are not updated in V
    ones = np.ones(parameter.S.shape[1])
    ones = tf.convert_to_tensor(np.float32(ones))
    # mask = np.matmul(np.ones(Vin.shape[1]), Pin)
    mask = tf.linalg.matvec(Pin, tf.ones([Vin.shape[1]]), transpose_a=True)
    # element in Vin are at 0 in mask others are at 1
    mask = ones - mask

    # Vin projection in V: elements not in Vin are at 0
    VinV  = tf.linalg.matmul(Vin, Pin, b_is_sparse=True)
    if parameter.mediumbound == 'UB': # we must have V ≤ Vin
        # relu = 1 when VinV > V, 0 othervise
        relu = tf.keras.activations.relu(VinV-V0)
        relu = tf.math.divide_no_nan(relu, relu) # 0/1 tensor
        # VinV = V when V < Vin, VinV = Vin when V > Vin
        VinV = relu * V0 + (ones-relu) * VinV
    V0 = tf.math.multiply(V0, mask) + VinV
    Vout = tf.convert_to_tensor(np.float32(targets))
    Vlb = tf.convert_to_tensor(np.float32(lower_bounds))
    mask = ones if parameter.mediumbound == 'UB' else mask

    return V0, Vin, Vout, Vlb, mask

def QP_layers(inputs, parameter, targets = np.asarray([]).reshape(0,0), lower_bounds=np.asarray([]).reshape(0,0), trainable=True, history=False, verbose=False):
    # Build and return an architecture using GD
    # The function is used with and without targets
    # - With targets there is no training set and GD is run
    #   to optimize both the objective min([PV-Target]^2))
    #   and the constraints.
    # - Without target an initial vector V is calculated via training
    #   through a Dense layer, GD is only used
    #   to minimize the constrains
    # Inputs:
    # - input flux vector, targets (can be empty)
    # - flags to train, record Loss history
    # Outputs:
    # - ouput_AMN (see function, and Loss (mean and std)

    V0, Vin, Vout, Vlb, mask = get_V0(inputs, parameter, targets, lower_bounds, trainable, verbose=verbose)

    print("V0: ", V0.shape)
    print("Vin: ", Vin.shape)
    print("Vout: ", Vout.shape)
    print("Vlb: ", Vlb.shape)
    print("mask: ", mask.shape)
    # print(abc)

    V, Loss_mean, Loss_std = Gradient_Descent(V0, Vin, Vout, Vlb, parameter, mask, trainable=trainable, history=history, verbose=verbose)
    outputs = output_AMN(V, Vin, V0, Vlb, parameter, verbose=verbose)

    return outputs, Loss_mean, Loss_std

def AMN_QP(parameter, trainable=True, verbose=False):
    # Build and return an AMN with training
    # input : problem parameter
    # output: Trainable model
    # Loss history is not recorded (already done thru tf training)

    # Get dimensions and build model
    input_dim, output_dim = parameter.X.shape[1], parameter.output_dim
    inputs = Input(shape=(input_dim,))
    outputs, loss_h, loss_std_h = QP_layers(inputs, parameter,
                              trainable=trainable,
                              history=False,
                              verbose=verbose)
    # Compile
    model = keras.models.Model(inputs=[inputs], outputs=outputs)
    (loss, metrics) = (my_mse, [my_r2])
    model.compile(loss=loss, optimizer='adam', metrics=metrics)
    if verbose == 2: print(model.summary())
    print('nbr parameters:', model.count_params())
    parameter.model = model

    return parameter

###############################################################################
# AMN models (2)
# AMN_LP: a ANN_Dense trainable prior layer and a mechanistic layer
# making use of the "LP" method (Solving a linear program with a
# recurrent neural network)
# Code written by Bastien Mollet
###############################################################################

def LP(V, M, b_int, b_ext, parameter, verbose=False):
    # Inputs:
    # - b_int and b_ext are not the same depending on whether
    #   UB is true or not (more details after).
    # Outputs: dV, dM
    # Recurrent cell from Y. Yang et al.
    # Mathematics and Computers in Simulation, 101 (2014) 103–112

    # Format all matrices and vectors
    OBJECTIVE_SCALER = 100 # scaler for the objective function c
    S_int, S_ext, Q, P, Sb, c = \
    tf.convert_to_tensor(parameter.S_int), \
    tf.convert_to_tensor(parameter.S_ext), \
    tf.convert_to_tensor(parameter.Q), \
    tf.convert_to_tensor(parameter.P), \
    tf.convert_to_tensor(parameter.Sb), \
    tf.convert_to_tensor(OBJECTIVE_SCALER*parameter.c)

    # The main difference between EB and UB is on the precomputed matrixes
    # S_int, S_ext, Q, P, Sb and the b_int and b_ext vectors
    SV  = tf.linalg.matvec(S_ext, V, transpose_a=True) # R = (M + S_extV - b_ext)^+
    # print("M, SV, b_ext, b_int, S_int", M.shape, SV.shape, b_ext.shape, b_int.shape, S_int.shape)
    R  = tf.keras.activations.relu(M + SV - b_ext)
    dV1 = tf.linalg.matvec(S_ext, R) + c # dV1 = P(S_extR + c)
    dV1 = tf.linalg.matvec(P, dV1)
    dV2 = tf.linalg.matvec(S_int, V) # dV2 = Q(S_intV - b_int)
    dV2 = dV2 - b_int
    dV2 = tf.linalg.matvec(Q,dV2)
    dV =  dV1 - dV2 # dV
    dM = 0.5 * (R - M) # dM = 1/2 (R - M)
    return dV, dM

def get_M0(inputs, parameter, targets, trainable, verbose=False):
    # Get initial vectors M0 from inputs and target
    # M0 is the initial value of the dual variable of LP
    # Return M0
    # When targets is provided M0 = 0 vector
    # When target is not provided M0 is computed via training
    # of Dense_Layers

    if parameter.mediumbound == 'UB':
        M0_size = 2 * parameter.S.shape[0] + parameter.S.shape[1]
    else:
        M0_size = parameter.S.shape[1]
    if targets.shape[0] > 0: # for MM models
        # Initialize M0 = 0 when targets provided, for MM (solving)
        M0 = tf.zeros((targets.shape[0], M0_size), dtype=tf.float32)
    else: # AMN models
        # M0 = Dense_layers(inputs)
        param = copy.copy(parameter)
        param.output_dim = M0_size
        param.activation = 'linear' # M0 can be negative
        M0 = Dense_layers(inputs, param,
                          trainable=trainable, verbose=verbose)
    return M0

def LP_layers(inputs_bounds, parameter, targets = np.asarray([]).reshape(0,0), lower_bounds=np.asarray([]).reshape(0,0), trainable=True, history=False, verbose=False):
    # Build and return an architecture using LP
    # UB:
    # Here the dimension of M corresponds to the 3 constraints:
    # 1: S_ext*V < b_ext [meta_dim] Upper bounds
    #    that are set from the medium compounds.
    # 2: V > 0 [flux_dim] Fluxs are supposed positive as we had split them.
    # 3: S_ext*V > 0 [meta_dim] ensure
    #    that there is no metabolite leaking from the cell.
    #    Thus the dimension of m is [flux_dim + 2*metadim]
    # EB :
    # the only inequality is V > 0 (that's why S_ext = I [n.n])
    # Inputs:
    # - input flux vector + boudary fluxes, targets (can be empty)
    # - flags to train, record Loss history
    # Outputs:
    # - ouput_AMN (see function, and Loss (mean and std)

    # Not history here if trainable
    history = False if trainable else history

    # Initialize AMN with VO, M0, b_int, b_ext
    # print("inputs_bounds ", inputs_bounds.shape)
    inputs = CROP(1, 0, parameter.Pin.shape[0]) (inputs_bounds)
    # print("inputs ", inputs.shape)
    b_int = CROP(1, parameter.Pin.shape[0],
                    parameter.Pin.shape[0] \
                 + parameter.b_int.shape[1] ) (inputs_bounds)
    # print("b_int ", b_int.shape)
    b_ext = CROP(1, parameter.Pin.shape[0] \
                 + parameter.b_int.shape[1],
                    parameter.Pin.shape[0] \
                 + parameter.b_int.shape[1] \
                 + parameter.b_ext.shape[1] ) (inputs_bounds)
    # print("b_ext ", b_ext.shape)
    V0, Vin, Vout, Vlb, mask = get_V0(inputs, parameter, targets, lower_bounds, trainable, verbose=verbose)
    M0 = get_M0(inputs, parameter, targets, trainable, verbose=verbose)

    # LP loop
    V, M = V0, M0
    Loss_mean_history, Loss_std_history = [], []
    for t in range(1, parameter.timestep+1):
        # Get Loss and gradients
        L, _ = Loss_all(V, Vin, Vout, Vlb, parameter)
        dV, dM = LP(V, M, b_int, b_ext, parameter, verbose=verbose)
        dV = tf.math.multiply(dV, mask) # Apply mask on dV
        V  = V + parameter.learn_rate * dV
        M  = M + parameter.learn_rate * dM
        # Compile Loss history
        if history:
            Loss_mean, Loss_std = np.mean(L), np.std(L)
            Loss_mean_history.append(Loss_mean)
            Loss_std_history.append(Loss_std)
            if verbose and (np.log10(t) == int(np.log10(t)) \
                            or t/1.0e3 == int(t/1.0e3)):
                print('LP-Loss', t, Loss_mean, Loss_std)
    outputs = output_AMN(V, Vin, V0, Vlb, parameter, verbose=verbose)

    return outputs, Loss_mean_history, Loss_std_history

def AMN_LP(parameter, trainable=True, verbose=False):
    # Build and return an AMN with training
    # input : problem parameter
    # output: Trainable model
    # Loss history is not recorded (already done thru tf training)

    # Get dimensions and build model
    input_dim, output_dim = parameter.X.shape[1], parameter.output_dim
    inputs = Input(shape=(input_dim,))
    outputs, loss_h, loss_std_h = LP_layers(inputs, parameter,
                              trainable=trainable,
                              history=False,
                              verbose=verbose)
    # Compile
    model = keras.models.Model(inputs=[inputs], outputs=outputs)
    (loss, metrics) = (my_mse, [my_r2])
    model.compile(loss=loss, optimizer='adam', metrics=metrics)
    if verbose == 2: print(model.summary())
    print('nbr parameters:', model.count_params())
    parameter.model = model
    return parameter

###############################################################################
# AMN models (3)
# AMN_Wt: An RNN where input (the medium) and flux vector V are passed
# to the recurrent cell
# M = V2M . V
# V = Win x Vin + Wrec x M2V . M
# Win and Wrec are weight matrices learned during training
# A hidden layer can be added to Win (not Wrec)
# Warning: The model AMN_Wt works only with UB training sets
###############################################################################

class RNNCell(keras.layers.Layer): # RNN Cell, as a layer subclass.
    def __init__(self, parameter):
        meta_dim = parameter.S.shape[0]
        flux_dim = parameter.S.shape[1]
        medm_dim = parameter.Pin.shape[0]
        self.input_size = medm_dim
        self.state_size = flux_dim
        self.mediumbound = parameter.mediumbound
        self.hidden_dim = parameter.hidden_dim
        self.S  = tf.convert_to_tensor(np.float32(parameter.S))
        self.V2M = tf.convert_to_tensor(np.float32(parameter.V2M))
        self.Pin = tf.convert_to_tensor(np.float32(parameter.Pin))
        # Normalize M2V
        M2V = parameter.M2V
        for i in range(flux_dim):
            if np.count_nonzero(M2V[i]) > 0:
                M2V[i] = M2V[i] / np.count_nonzero(M2V[i])
        self.M2V  = tf.convert_to_tensor(np.float32(M2V))
        self.dropout = parameter.dropout
        super(RNNCell, self).__init__(trainable=True)

    def build(self, input_shape):
        meta_dim = self.S.shape[0]
        flux_dim = self.S.shape[1]
        medm_dim = self.input_size
        hidden_dim = self.hidden_dim
        print("meta_dim:   ", meta_dim)
        print("flux_dim:   ", flux_dim)
        print("medm_dim:   ", medm_dim)
        print("hidden_dim: ", hidden_dim)
        # weigths to compute V for both input (i) and recurrent cell (r)
        if self.mediumbound == 'UB': # no kernel_Vh and kernel_Vi for EB
            if hidden_dim > 0: # plug an hidden layer upstream of Winput
                self.wh_V = self.add_weight(shape=(medm_dim, hidden_dim),
                                        name='kernel_Vh')
                self.wi_V = self.add_weight(shape=(hidden_dim, medm_dim),
                                        name='kernel_Vi')
            else:
                self.wi_V = self.add_weight(shape=(medm_dim, medm_dim),
                                        name='kernel_Vi')
        self.wr_V = self.add_weight(shape=(flux_dim, meta_dim),
                                        name='kernel_Vr')
        self.bi_V  = self.add_weight(shape=(medm_dim,),
                                        initializer='random_normal',
                                        name='bias_Vi',
                                        trainable=True)
        self.br_V  = self.add_weight(shape=(flux_dim,),
                                        initializer='random_normal',
                                        name='bias_Vr',
                                        trainable=True)
        self.built = True

    def call(self, inputs, states):
        # At steady state we have
        # M = V2M V and V = (M2V x W) M + V0
        # Keep Vin only
        # inputs = CROP(1, 0, self.input_size)(inputs)

        V = states[0]
        # print("RNN states -> ", states[0])
        if self.mediumbound == 'UB':
            if self.hidden_dim > 0:
                VH = K.dot(inputs, self.wh_V)
                V0 = K.dot(VH, self.wi_V) + self.bi_V
            else:
                V0 = K.dot(inputs, self.wi_V) + self.bi_V
        else:
            V0 = inputs # EB case
        V0 = tf.linalg.matmul(V0, self.Pin, b_is_sparse=True)
        M = tf.linalg.matmul(V,tf.transpose(self.V2M),b_is_sparse=True)
        W = tf.math.multiply(self.M2V,self.wr_V)
        V = tf.linalg.matmul(M,tf.transpose(W),b_is_sparse=True)
        V = V + V0 + self.br_V
        # print("RNN call -> ", V.shape)
        # print("End RNN call")
        return V, [V]

    def get_config(self): # override tf.get_config to save RNN model
        # The code below does not work !! anyone to debug?
        config = super().get_config().copy()
        #config.update({'parameter': self.parameter.__dict__})
        return config

def split_inputs(inputs, parameter):
    d0, d1 = \
        inputs[:,0,:].shape[0], inputs[:,0,:].shape[1]
    if d1 > parameter.input_dim:
        Vin = CROP(1, 0, parameter.input_dim)(inputs[:,0,:])
        Vbf = CROP(1, parameter.input_dim, d1)(inputs[:,0,:])
        print('Vlb ', Vbf.shape)
        # print(abc)
    else:
        Vin = inputs[:,0,:]
        Vbf = inputs[:,0,:]

    return Vin, Vbf

def Wt_layers(inputs, parameter, trainable=True, verbose=False):
    # Build and return AMN layers using an RNN cell
    with CustomObjectScope({'RNNCell': RNNCell}):
        rnn = keras.layers.RNN(RNNCell(parameter))

    Vin, Vbf = split_inputs(inputs, parameter)

    rnn_input = CROP(2, 0, parameter.input_dim)(inputs)
    V = rnn(rnn_input)
    # Vin = inputs[:,0,:]
    print("inputs Wt_layers ", inputs.shape)
    print("inputs 0 Wt_layers ", inputs[:,0,:].shape)

    return output_AMN(V, Vin, None, None, parameter, verbose=verbose)

def AMN_Wt(parameter, verbose=False):
    # Build and return an AMN using an RNN cell
    # input : medium vector in parameter
    # output: experimental steaty state fluxes

    # Get dimensions and build model
    input_dim, output_dim  = parameter.X.shape[2], parameter.Y.shape[1]
    inputs = keras.Input((None, input_dim))
    outputs = Wt_layers(inputs, parameter)


    # Compile
    model = keras.models.Model(inputs, outputs)
    # (loss, metrics) = (my_mse, [my_r2])
    Vin, Vbf = split_inputs(inputs, parameter)

    # ## use lambda layer loss
    # loss_model = generate_loss_model(parameter, Vin, Vbf)
    # # Create custom loss instance
    # custom_loss = CustomLoss(loss_model, parameter)
    # loss, metrics = custom_loss, None

    # use custom loss function
    loss, metrics = biochem_loss(parameter), None

    # Compile the model with the custom loss function
    model.compile(loss=loss,  optimizer='adam', metrics=metrics)#, run_eagerly=True)
    print(model.summary())
    if verbose == 2: print(model.summary())
    print('nbr parameters:', model.count_params())
    parameter.model = model

    return parameter

###############################################################################
# Non-trainable Mechanistic Model (MM)
# using QP or LP
###############################################################################

def write_loss(f_name, param, mean_history, std_history):
    if f_name is None:
        return 0
    timesteps = np.array(range(1, param.timestep+1))
    losses = np.array(mean_history)
    stdevs = np.array(std_history)
    to_write = np.concatenate([timesteps.reshape((len(timesteps), 1)), \
        losses.reshape((len(losses), 1)), \
        stdevs.reshape((len(stdevs), 1))], axis=1)
    np.savetxt(f_name, to_write, delimiter=',')
    return 0

def write_targets(f_name, param, Ypred):
    if f_name is None:
        return 0
    true = np.array(param.Y)
    pred = np.array(Ypred)
    to_write = np.concatenate([true.reshape((len(true), 1)), \
        pred.reshape((len(pred), 1))], axis=1)
    np.savetxt(f_name, to_write, delimiter=',')
    return 0

def get_flux_output(param, output):
    # Just getting vector V from output
    # output : PoutV (=Ypred) + SV + PinV + Vpos + V + V0
    len_fluxes = param.S.shape[1]
    if output.shape[1] > (len_fluxes+NBR_CONSTRAINT+1): # case where we get V0
        V0 = CROP(1,param.Y.shape[1]+NBR_CONSTRAINT+len_fluxes,
                    param.Y.shape[1]+NBR_CONSTRAINT+len_fluxes*2) (output)
        Vf = CROP(1,param.Y.shape[1]+NBR_CONSTRAINT,
                    param.Y.shape[1]+NBR_CONSTRAINT+len_fluxes) (output)
    else: # case where we don't have V0 at the end of the output
        Vf = CROP(1,param.Y.shape[1]+NBR_CONSTRAINT,
                    param.Y.shape[1]+NBR_CONSTRAINT+len_fluxes) (output)
    return Vf

def MM_LP_QP(parameter, LP=True, loss_outfile=None, targets_outfile=None, history=True, verbose=False):
    # Solve LP or QP without training
    # inputs:
    # - problem parameter, history flag
    # output:
    # - Predicted all fluxes and stats = loss history

    # inputs must be in tf format
    param = copy.copy(parameter)
    if param.X.shape[1] < param.S.shape[1]:
        # when all X provided no need to tranform
        param.X, _ = input_AMN(param, verbose=False)
    inputs  = tf.convert_to_tensor(np.float32(param.X))
    targets = param.Y
    lower_bounds = param.LB

    # run LP or QP
    if LP:
        output, Loss_mean, Loss_std = LP_layers(inputs, param, targets=targets,
                    lower_bounds=lower_bounds, trainable=False, history=history,
                    verbose=verbose)
    else:
        output, Loss_mean, Loss_std = QP_layers(inputs, param, targets=targets,
                    lower_bounds=lower_bounds, trainable=False, history=history,
                    verbose=verbose)
    Ypred = CROP(1,0,param.Y.shape[1]) (output)
    Vf = get_flux_output(param, output)
    # compute R2 and write losses and targets
    # print(param.Y)
    # print(Ypred.numpy())
    r2 = r2_score(param.Y, Ypred.numpy(), multioutput='variance_weighted')
    write_loss(loss_outfile, parameter, Loss_mean, Loss_std)
    # write_targets(targets_outfile, parameter, Ypred)
    write_targets(None, parameter, Ypred)

    return Vf.numpy(), ReturnStats(r2, 0, Loss_mean[-1], Loss_std[-1],
                                   0, 0, 0, 0)

def MM_LP(parameter, loss_outfile=None, targets_outfile= None, history=True, verbose=False):
    # Solve LP without training
    return MM_LP_QP(parameter, LP=True,
                    loss_outfile=loss_outfile, targets_outfile=targets_outfile,
                    history=history, verbose=verbose)

def MM_QP(parameter, loss_outfile=None, targets_outfile= None, history=True, verbose=False):
    # Solve QP without training
    return MM_LP_QP(parameter, LP=False,
                    loss_outfile=loss_outfile, targets_outfile=targets_outfile,
                    history=history, verbose=verbose)

###############################################################################
# RC models
# This module is making use of trained AMNs (cf. previous module)
# in reseroir computing (RC). The reservoir (non-trainable AMN)
# is squized between two standard ANNs. The purpose of the prior ANN is to
# transform problem features into nutrients added to media.
# The post-ANN reads reservoir output (user predefined specific
# reaction rates) and produce a readout to best match training set values.
# Note that the two ANNs are trained but not the reservoir (AMN).
###############################################################################

def input_RC(parameter, verbose=False):
    # Shape X and Y for RC
    X, Y = parameter.X, parameter.Y
    if parameter.scaler != 0: # Normalize X
        X, parameter.scaler = MaxScaler(X)
    if verbose: print('AMN scaler', parameter.scaler)
    if verbose: print('RC input shape',X.shape,Y.shape)
    parameter.input_dim = parameter.X.shape[1]

    # non variable part is set to valmed
    if parameter.mediumbound == 'UB':
        for i in range(len(parameter.res.levmed)):
            if parameter.res.levmed[i] == 1:
                # Scale X with valmed
                X[:,i] = parameter.res.valmed[i]
    parameter.X, parameter.Y = X, Y
    return parameter.X, parameter.Y

def RC(parameter, verbose=False):
    # Build and return a Reservoir Computing model
    # The model is composed of
    # - A prior trainable network that generate
    #   an outpout = input of the reservoir
    # - The non-trainable reservoir (must have been created and saved)
    # - A post trainable network that takes as input the reservor output
    #   and produce problem's output
    # - A last layer that concatenate the prior trainable output
    #   and the post trainable output

    # Prior
    # If the mode of the reservoir is UB prior computes only
    # the variable part of the medium
    inputs, L  = Input(shape=(parameter.input_dim,)), 0
    if parameter.prior:
        if parameter.res.mediumbound == 'UB':
            L = inputs.shape[1] - parameter.prior.input_dim
            Res_inputs = CROP(1, 0, L) (inputs) # minmed
            Prior_inputs = CROP(1, L, inputs.shape[1]) (inputs) # varmed
        else: # L=0
            Prior_inputs = inputs
        Prior_outputs = Dense_layers(Prior_inputs, parameter.prior,
                                     trainable=True, verbose=verbose)
        if verbose:
            print('Prior inputs and outputs', Prior_inputs.shape, Prior_outputs.shape)
            print('Res inputs added to Prior_outputs', L)
    else:
        Prior_outputs = inputs

    # Reservoir
    # 1. Add to Res_inputs the fixed part of input
    # 2. Mask in Res_inputs all zero elements in inputs
    # 3. Call (non trainable) reservoir
    # 4. Get for forward passing the loss on constraints
    Res_inputs = concatenate([Res_inputs, Prior_outputs], axis=1) \
    if L > 0 else Prior_outputs
    # O/1 mask
    inputs_mask = tf.math.divide_no_nan(inputs, inputs)
    Res_inputs = tf.math.multiply(Res_inputs, inputs_mask)
    # Res inputs is scaled to fit training data
    if parameter.res.scaler != 0:
        Res_inputs = Res_inputs / parameter.res.scaler
    if verbose: print('Res inputs (final)', Res_inputs.shape)
    # Run reservoir
    Res_layers = QP_layers
    Res_outputs, _, _ = Res_layers(Res_inputs, parameter.res,
                                   trainable=False, verbose=verbose)
    if verbose: print('Res_outputs--------------------', Res_outputs.shape)
    # Get losses
    L = len(parameter.res.objective) # Objective length
    Post_inputs = CROP(1, 0, L) (Res_outputs) # Objective only
    SV =   CROP(1, L,   L+1) (Res_outputs)
    PinV = CROP(1, L+1, L+2) (Res_outputs)
    Vpos = CROP(1, L+2, L+3) (Res_outputs)
    V =    CROP(1, L+3, L+3+parameter.res.S.shape[1]) (Res_outputs)
    if verbose: print('SV, PinV, Vpos, V--------------',
                      SV.shape, PinV.shape, Vpos.shape, V.shape)
    # Post
    if parameter.post:
        if verbose: print('Post_inputs--------------------', Post_inputs.shape)
        Post_outputs = Dense_layers(Post_inputs, parameter.post, verbose=verbose)
    else:
        Post_outputs = Post_inputs

    # RC output
    outputs = concatenate([Post_outputs, SV, PinV, Vpos, V, Res_inputs], axis=1)

    # Compile optimizer parametized for few data
    model = keras.models.Model(inputs, outputs)
    (loss, metrics) = (my_mse, [my_r2]) if parameter.regression \
    else (my_binary_crossentropy, [my_acc])
    opt = tf.keras.optimizers.Adam(learning_rate=parameter.train_rate)
    model.compile(loss=loss,  optimizer=opt, metrics=metrics)
    if verbose == 2: print(model.summary())
    print('nbr parameters:', model.count_params())
    parameter.model = model

    # Set the weights on the non-trainable part of the network
    weights_model = model.get_weights()
    weights_reservoir = parameter.res.model.get_weights()
    if parameter.post:
        lenpost = 2 + parameter.post.n_hidden * 2 # 2 = weight + biais
    else:
        lenpost = 0
    N = len(weights_model) - len(weights_reservoir) - lenpost
    for i in range(len(weights_reservoir)):
        weights_model[N+i] = weights_reservoir[i]
    model.set_weights(weights_model)

    return parameter

###############################################################################
# Train and Evaluate all models
###############################################################################

class ReturnStats:
    def __init__(self, v1, v2, v3, v4, v5, v6, v7, v8):
        self.train_objective = (v1, v2)
        self.train_loss = (v3, v4)
        self.test_objective = (v5, v6)
        self.test_loss = (v7, v8)

def print_loss_evaluate(y_true, y_pred, Vin, Vlb, parameter):
    # Print all losses
    loss_out0, loss_outf = -1, -1
    loss_cst0, loss_cstf = -1, -1
    loss_all0, loss_allf = -1, -1
    end = len(parameter.reactions) # y_true.shape[1] - NBR_CONSTRAINT
    nV = parameter.S.shape[1]
    pred_index = len(parameter.reactions) + NBR_CONSTRAINT
    # print("print_loss_evaluate : y_predShape ", y_pred.shape)
    # print("print_loss_evaluate : y_trueShape ", y_true.shape)
    # print("print_loss_evaluate : y_true.shape[1] ", y_true.shape[1])
    # print("print_loss_evaluate : nV ", nV)

    # Vf = y_pred[:,y_true.shape[1]:y_true.shape[1]+nV]
    Vf = y_pred[:,pred_index:pred_index+nV]
    if y_true.shape[1] <= pred_index:
        Vout = y_true[:,:end]
    else:
        Vout = y_true[:,parameter.input_dim:end+parameter.input_dim]
        # print("print_loss_evaluate : from ", parameter.input_dim, " to ", end+parameter.input_dim)
        # print("print_loss_evaluate : Vout ", Vout.shape)

    if y_pred.shape[1] == y_true.shape[1]+nV+nV:
        V0 = y_pred[:,y_true.shape[1]+nV:y_true.shape[1]+nV+nV]
        loss_out0, _ = Loss_Vout(V0, parameter.Pout, Vout)
        loss_out0 = np.mean(loss_out0.numpy())
        loss_cst0, _ = Loss_constraint(V0, Vin, Vlb, parameter)
        loss_cst0 = np.mean(loss_cst0.numpy())
        loss_all0, _ = Loss_all(V0, Vin, Vout, Vlb, parameter)
        loss_all0 = np.mean(loss_all0.numpy())
    loss_outf, _ = Loss_Vout(Vf, parameter.Pout, Vout)
    loss_outf = np.mean(loss_outf.numpy())
    loss_cstf, _ = Loss_constraint(Vf, Vin, Vlb, parameter)
    loss_cstf = np.mean(loss_cstf.numpy())
    loss_allf, _ = Loss_all(Vf, Vin, Vout, Vlb, parameter)
    loss_allf = np.mean(loss_allf.numpy())
    print('Loss out on V0: ', loss_out0)
    print('Loss constraint on V0: ', loss_cst0)
    print('Loss all on V0: ', loss_all0)
    print('Loss out on Vf: ', loss_outf)
    print('Loss constraint on Vf: ', loss_cstf)
    print('Loss all on Vf: ',  loss_allf)
    if y_pred.shape[1] == y_true.shape[1]+nV+nV:
        d = np.linalg.norm(Vf - V0)
        print('Distance V0 to Vf %f: ' % (d))
    return

def get_loss_evaluate(x, y_true, y_pred, parameter, verbose=False):
    # Return loss on constraint for y_pred
    end = y_true.shape[1]
    pred_index = y_true.shape[1] - parameter.input_dim
    if 'AMN' in parameter.model_type:
        nV = parameter.S.shape[1]
        Vf = y_pred[:,pred_index:pred_index+nV]
        if 'AMN_LP' in parameter.model_type:
            # x = Vin + bounds is truncated
            Vin = x[:,0: parameter.Pin.shape[0]]
        elif 'AMN_Wt' in parameter.model_type:
            # The dimension (time) added to x with RNN is removed
            Vin  = x[:,0,0:parameter.Pin.shape[0]]
            Vlb = Vin.copy()
            if Vin.shape[1] > parameter.Pin.shape[0]:
                Vlb  = x[:,0,parameter.Pin.shape[0]:]
        else:
            Vin = x

        if verbose:
            print_loss_evaluate(y_true, y_pred, Vin, Vlb, parameter)
        loss, _ = Loss_constraint(Vf, Vin, Vlb, parameter)
        loss = np.mean(loss.numpy())
    else:
        loss = -1

    return loss

def evaluate_model(model, x, y_true, parameter, inputmodel= False, verbose=False):
    # Return y_pred, stats (R2/Acc) for objective
    # and error on constraints for regression and classification
    # if input model than x, y_true sent to input model

    if inputmodel:
        param = copy.copy(parameter)
        param.X, param.Y = x, y_true
        X, Y = model_input(param, verbose=verbose)
        param.X, param.Y = X, Y
    y_pred = model.predict(x) # whole y prediction

    # AMN models have NBR_CONSTRAINT constraints added to y_true
    end = y_true.shape[1] - NBR_CONSTRAINT \
    if 'AMN' in parameter.model_type else y_true.shape[1]
    print('----------', end)
    if parameter.regression:
        yt, yp = y_true[:,:end], y_pred[:,:end]
        if yt.shape[0] == 1: # LOO case
            rss, tss = (yp - yt) * (yp - yt), yt * yt
            if np.sum(tss) > 0:
                obj = 1 - np.sum(rss) / np.sum(tss)
            else:
                obj = 1 - np.sum(rss)
            print('LOO True, Pred, Q2 =', yt, yp, obj)
        else:
            obj = r2_score(yt, yp, multioutput='variance_weighted')
    else:
        end = y_true.shape[1]
        obj = keras.metrics.binary_accuracy(y_true[:,:end],
                                            y_pred[:,:end]).numpy()
        #print('evaluate acc \ny_true', y_true[:,:end], '\ny_pred', y_pred[:,:end])
        obj = np.count_nonzero(obj)/obj.shape[0]

    # compute stats on constraints
    loss = get_loss_evaluate(x, y_true, y_pred, parameter, verbose=verbose)
    stats  = ReturnStats(obj, 0, loss,  0, obj, 0, loss,  0)

    return y_pred, stats

def model_input(parameter, trainable=True, verbose=False):
    # return input for the appropriate model_type
    if   'ANN' in parameter.model_type:
        return input_ANN_Dense(parameter, verbose=verbose)
    elif 'RC' in parameter.model_type:
        return input_RC(parameter, verbose=verbose)
    elif 'AMN' in parameter.model_type:
        # parameter.batch_size = 10
        return input_AMN(parameter, verbose=verbose)
    elif 'MM' in parameter.model_type:
        return input_AMN(parameter, verbose=verbose)
    else:
        print(parameter.model_type)
        sys.exit('no input available')

def model_type(parameter, verbose=False):
    # create the appropriate model_type
    if verbose:
        print('-----------------------------------', parameter.model_type)
    if 'ANN_Dense' in parameter.model_type:
        return ANN_Dense(parameter, verbose=verbose)
    elif 'RC' in parameter.model_type:
        return RC(parameter, verbose=verbose)
    elif 'AMN_LP' in parameter.model_type:
        return AMN_LP(parameter, verbose=verbose)
    elif 'AMN_QP' in parameter.model_type:
        return AMN_QP(parameter, verbose=verbose)
    elif 'AMN_Wt' in parameter.model_type:
        return AMN_Wt(parameter, verbose=verbose)
    else:
        print(parameter.model_type)
        sys.exit('not a trainable model')

def train_model(parameter, Xtrain, Ytrain, Xtest, Ytest, verbose=False):
    # A standard function to create a model, fit, and test
    # with early stopping
    # Inptuts:
    # - all necessary parameter including
    #   parameter.model, the function used to create the model
    #   parameter.input_model, the function used to shape the model inputs
    #   parameter.X and parameter.Y, the dataset
    #   parameter.regression (boolean) if false classification
    # Outputs:
    # - Net: the trained network
    # - ytrain, ytest: y values for training and tets sets
    # - otrain, ltrain: objective and loss for trainig set
    # - otest, ltest: objective and loss for trainig set
    # - history: tf fit histrory
    # Must have verbose=2 to verbose the fit

    Niter = 1 # maximum number of attempts to fit

    # Create model fit and evaluate
    for kiter in range(Niter): # Looping until properly trained
        if 'AMN_Wt' in parameter.model_type:
            # we have to recreate the object model with AMN-Wt
            # parameter.batch_size = 10
            model = Neural_Model(trainingfile = parameter.trainingfile,
            objective= parameter.objective, model=parameter.model,
            model_type=parameter.model_type, scaler=parameter.scaler,
            input_dim=parameter.input_dim, output_dim=parameter.output_dim,
            n_hidden=parameter.n_hidden, hidden_dim=parameter.hidden_dim,
            activation=parameter.activation, timestep=parameter.timestep,
            learn_rate=parameter.learn_rate, decay_rate=parameter.decay_rate,
            regression=parameter.regression, epochs=parameter.epochs,
            train_rate=parameter.train_rate, dropout=parameter.dropout,
            batch_size=parameter.batch_size, niter=parameter.niter,
            xfold=parameter.xfold,  es=parameter.es, verbose=verbose)
            model.X, model.Y = Xtrain, Ytrain
        else:
            model = parameter
        Net = model_type(model, verbose=verbose)
        # early stopping
        es = EarlyStopping(monitor='val_loss', mode='min',
                           patience=10, verbose=verbose)
        callbacks = [es] if model.es else []
        # fit
        v = True if verbose == 2 else False
        epochs = 0.9 * model.epochs
        print('Xtrain -> ', Xtrain.shape)
        print('Ytrain -> ', Ytrain.shape)
        print('Xtest  -> ', Xtest.shape)
        print('Ytest  -> ', Ytest.shape)
        print('epochs -> ', model.epochs)
        print('batch  -> ', model.batch_size)
        # csv_logger = CSVLogger('log.csv', append=True, separator=';')
        # save_freq
        run = wandb.init(config={"learning_rate": parameter.learn_rate,
                            "decay_rate": parameter.decay_rate,
                            "architecture": "AMM_wt",
                            "dataset": "secMet",
                            "epochs": parameter.timestep,
                            "bs": parameter.batch_size
                            })
        os.environ["WANDB_SILENT"] = "true"

        history = Net.model.fit(Xtrain, Ytrain,
                                validation_data=(Xtest, Ytest),
                                epochs=model.epochs,
                                batch_size=model.batch_size,
                                callbacks=callbacks+[WandbMetricsLogger(),
                                                    WandbModelCheckpoint(filepath="models/"),]
                                                    # csv_logger,]
                                , verbose=2)
                                # , run_eagerly=True)
        # evaluate
        ytrain, stats = evaluate_model(Net.model, Xtrain, Ytrain,
                                       model, verbose=verbose)
        otrain, ltrain = stats.train_objective[0], stats.train_loss[0]
        if otrain > 0.5:
            break
        else:
            print('looping bad training iter=%d r2=%.4f' % (kiter, otrain))

    # Hopefullly fit is > 0.5 now evaluate test set
    ytest, stats  = evaluate_model(Net.model, Xtest,  Ytest,
                                   model, verbose=verbose)
    otest, ltest = stats.test_objective[0], stats.test_loss[0]

    print("train = %.2f test = %.2f loss-train = %6f loss-test = %.6f iter=%d" % \
          (otrain, otest, ltrain, ltest, kiter))

    # Close the W&B run
    run.finish()
    return Net, ytrain, ytest, otrain, ltrain, otest, ltest, history

def train_evaluate_model(parameter, verbose=False):
    # A standard function to create a model, fit, and Kflod cross validate
    # with early stopping
    # Kfold is performed for param.xfold test sets (if param.niter = 0)
    # otherwise only for niter test sets
    # Inptuts:
    # - all necessary parameter including
    #   parameter.model, the function used to create the model
    #   parameter.input_model, the function used to shape the model inputs
    #   parameter.X and parameter.Y, the dataset
    #   parameter.regression (boolean) if false classification
    # Outputs:
    # - the best model (highest Q2/Acc on kfold test sets)
    # - the values predicted for each fold (if param.niter = 0)
    #   or the whole set when (param.niter > 0)
    # - the mean R2/Acc on the test sets
    # - the mean constraint value on the test sets
    # Must have verbose=True to verbose the fit

    param = copy.copy(parameter)
    X, Y = model_input(param, verbose=verbose)
    param.X, param.Y = X, Y
    # print('****** ', X[0])

    # Train on all data
    if param.xfold < 2: # no cross-validation
        Net, ytrain, ytest, otrain, ltrain, otest, ltest, history = \
        train_model(param, X, Y, X, Y, verbose=verbose)
        # Return Stats
        stats = ReturnStats(otrain, 0, ltrain, 0, otest, 0, ltest, 0)

        temp_pred = Net.model.predict(X)
        np.savetxt("Result/final_PredEval.csv", temp_pred, delimiter=',')

        return Net, ytrain, stats, history

    # Cross-validation loop
    Otrain, Otest, Ltrain, Ltest, Omax, Netmax, Ypred = \
    [], [], [], [], -1.0e32, None, np.copy(Y)
    kfold = KFold(n_splits=param.xfold, shuffle=True)
    kiter = 0
    for train, test in kfold.split(X, Y):
        if verbose: print('------- train', X[train].shape, Y[train].shape)
        if verbose: print('------- test ', X[test].shape, Y[test].shape)
        # print()
        # print(CROP(2, 4, X[train].shape[2])(X[train])[0])
        # print(abc)
        # parameter.input_dim
        Net, ytrain, ytest, otrain, ltrain, otest, ltest, history = \
        train_model(param, X[train], Y[train], X[test], Y[test], verbose=verbose)
        # compile Objective (O) and Constraint (C) for train and test
        Otrain.append(otrain)
        Otest.append(otest)
        Ltrain.append(ltrain)
        Ltest.append(ltest)
        # in case y does not have the same shape than Y
        if Ypred.shape[1] != ytest.shape[1]:
            n, m = Y.shape[0], ytest.shape[1]
            Ypred = np.zeros(n*m).reshape(n,m)
        for i in range(len(test)):
            Ypred[test[i]] = ytest[i]
        # Get the best network
        (Omax, Netmax) = (otest, Net) if otest > Omax else (Omax, Netmax)
        kiter += 1
        if (param.niter > 0 and kiter >= param.niter) or kiter >= param.xfold:
                break

    # Prediction using best model on whole dataset
    Pred, _ = evaluate_model(Netmax.model, X, Y, param, verbose=verbose)
    Ypred = Pred if param.niter > 0 else Ypred

    # np.savetxt("Result/final_Ypred.csv", Ypred, delimiter=',')
    # np.savetxt("Result/final_PredEval.csv", Pred, delimiter=',')

    # Get Stats
    stats = ReturnStats(np.mean(Otrain), np.std(Otrain),
                        np.mean(Ltrain), np.std(Ltrain),
                        np.mean(Otest),  np.std(Otest),
                        np.mean(Ltest),  np.std(Ltest))

    return Netmax, Ypred, stats, history

class Neural_Model:
    # To save, load & print all kinds of models including reservoirs
    def __init__(self,
                 trainingfile=None, # training set parameter file
                 objective=None,
                 model=None, # the actual Keras model
                 model_type='', # the function called Dense, AMN, RC...
                 scaler=False, # X is not scaled by default
                 input_dim=0, output_dim=0, # model IO dimensions
                 n_hidden=0, hidden_dim=0, # default no hidden layer
                 activation='relu', # activation for last layer
                 timestep=0, learn_rate=1.0, decay_rate=0.9,# for GD in AMN
                 # for all trainable models adam default learning rate = 1e-3
                 regression=True,
                 epochs=0, train_rate=1e-3, dropout=0.25, batch_size=100,
                 niter=0, xfold=5, # Cross valisation LOO does not work
                 es=False, # early stopping
                 biomass_max=4.0,
                 verbose=False,
                ):
        # Create empty object
        if model_type == '':
            return
        # model architecture parameters
        self.trainingfile = trainingfile
        self.model = model
        self.model_type = model_type
        self.objective = [objective[1]] if objective[0] else []
        self.scaler = float(scaler) # From bool to float
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.activation = activation
        # LP or QP parameters
        self.timestep = timestep
        self.learn_rate = learn_rate
        self.decay_rate = decay_rate
        # Training parameters
        self.epochs = epochs
        self.regression = regression
        self.train_rate = train_rate
        self.dropout = dropout
        self.batch_size = batch_size
        self.niter = niter
        self.xfold = xfold
        self.es = es
        self.mediumbound = '' # initialization
        self.reactions = list()
        self.treatments = list()
        # Get additional parameters (matrices)
        self.get_parameter(biomass_max=biomass_max, verbose=verbose)
        self.bio_id = self.reactions.tolist().index(objective[1])

    def get_parameter(self, biomass_max=4.0, verbose=False):
        # load parameter file if provided
        if self.trainingfile == None:
            return
        if not os.path.isfile(self.trainingfile+'.npz'):
            print(self.trainingfile+'.npz')
            sys.exit('parameter file not found')
        parameter = TrainingSet()
        parameter.load(self.trainingfile)
        if not ('MM_' in self.model_type):
            if self.objective:
                print("Filtering here AMN!")
                parameter.filter_measure(measure=self.objective, verbose=verbose)
                print("measure: ", parameter.measure)
            self.Yall = parameter.Yall if self.objective else None
        self.mediumbound = parameter.mediumbound
        self.levmed = parameter.levmed
        self.valmed = parameter.valmed
        # matrices from parameter file
        self.S = parameter.S # Stoichiometric matrix
        self.Pin = parameter.Pin # Boundary matrix from reaction to medium
        self.Pout = parameter.Pout # Measure matrix from reactions to measures
        self.V2M = parameter.V2M # Reaction to metabolite matrix
        self.M2V = parameter.M2V # Metabolite to reaction matrix
        self.X = parameter.X # Training set X
        self.Y = parameter.Y # Training set Y
        self.LB = parameter.LB # Training set LB
        self.S_int = parameter.S_int
        self.S_ext = parameter.S_ext
        self.Q = parameter.Q
        self.P = parameter.P
        self.b_int = parameter.b_int
        self.b_ext = parameter.b_ext
        self.Sb = parameter.Sb
        self.c = parameter.c
        self.reactions = parameter.reactions
        self.treatments = parameter.treatments
               
        # Update input_dim and output_dim
        self.input_dim = self.input_dim if self.input_dim > 0 \
        else parameter.X.shape[1]
        if self.input_dim > len(self.reactions):
            self.input_dim = self.input_dim - len(self.reactions)
        self.output_dim = self.output_dim if self.output_dim > 0 \
        else parameter.Y.shape[1]

        self.objPout, self.objY = None, None
        if 'MM_' in self.model_type:
            print("filtering the MM")
            self.Yall = None
            if self.objective:
                # parameter.filter_measure(measure=self.objective, verbose=verbose)
                self.objPout, parameter.Y, self.objY, self.Yall = parameter.filter_measure_return(measure=parameter.objective, biomass_max=biomass_max, verbose=verbose)

    def save(self, filename, verbose=False):
        fileparam = filename + "_param.csv"
        print(fileparam)
        filemodel = filename + "_model.h5"
        s = str(self.trainingfile) + ","\
                    + str(self.model_type) + ","\
                    + str(self.objective) + ","\
                    + str(self.scaler) + ","\
                    + str(self.input_dim) + ","\
                    + str(self.output_dim) + ","\
                    + str(self.n_hidden) + ","\
                    + str(self.hidden_dim) + ","\
                    + str(self.activation) + ","\
                    + str(self.timestep) + ","\
                    + str(self.learn_rate) + ","\
                    + str(self.decay_rate) + ","\
                    + str(self.epochs) + ","\
                    + str(self.regression) + ","\
                    + str(self.train_rate) + ","\
                    + str(self.dropout) + ","\
                    + str(self.batch_size) + ","\
                    + str(self.niter) + ","\
                    + str(self.xfold) + ","\
                    + str(self.es)
        with open(fileparam, "w") as h:
            # print(s, file = h)
            h.write(s)
        if self.model_type == "AMN_LP":
            json = filename + "_model.json"
            self.model.save_weights(filemodel)
            model_json = self.model.to_json()
            with open(json, "w") as json_file:
                json_file.write(model_json)
            json_file.close()
        self.model.save(filemodel)

    def load(self, filename, verbose=False):
        fileparam = filename + "_param.csv"
        filemodel = filename + "_model.h5"
        if not os.path.isfile(fileparam):
            print(fileparam)
            sys.exit('parameter file not found')
        if not os.path.isfile(filemodel):
            print(filemodel)
            sys.exit('model file not found')
        # First read parameter file
        with open(fileparam, 'r') as h:
            for line in h:
                K = line.rstrip().split(',')
        # model architecture
        self.trainingfile =  str(K[0])
        self.model_type =  str(K[1])
        self.objective =  str(K[2])
        self.scaler =  float(K[3])
        self.input_dim =  int(K[4])
        self.output_dim = int(K[5])
        self.n_hidden = int(K[6])
        self.hidden_dim = int(K[7])
        self.activation = str(K[8])
        # GD parameters
        self.timestep = int(K[9])
        self.learn_rate = float(K[10])
        self.decay_rate = float(K[11])
        # Training parameters
        self.epochs = int(K[12])
        self.regression = True if K[13] == 'True' else False
        self.train_rate = float(K[14])
        self.dropout = float(K[15])
        self.batch_size = int(K[16])
        self.niter = int(K[17])
        self.xfold = int(K[18])
        self.es = True if K[19] == 'True' else False
        # Make objective a list
        self.objective = self.objective.replace('[', '')
        self.objective = self.objective.replace(']', '')
        self.objective = self.objective.replace('\'', '')
        self.objective = self.objective.replace("\"", "")
        self.objective = self.objective.split(',')
        # Get additional parameters (matrices)
        self.get_parameter(verbose=verbose)
        # Then load model
        if self.model_type == 'AMN_Wt':
            self.model = load_model(filemodel,
                                    custom_objects={'RNNCell':RNNCell,
                                                    'parameter':Neural_Model
                                                    , 'custom_loss': CustomLoss
                                                    # , 'loss_model': generate_loss_model(parameter, Vin, Vbf)
                                                    , 'loss': biochem_loss(parameter)},
                                    compile=False)
        elif self.model_type == 'AMN_LP':
            # print("bugs here")
            from keras.models import model_from_json
            json = filename + "_model.json"
            json_file = open(json, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.load_weights(filemodel)
            self.model = model
            # print("loaded")
        else:
            self.model = load_model(filemodel, compile=False)

    def printout(self,filename=''):
        if filename != '':
            sys.stdout = open(filename, 'a')
        print('training file:', self.trainingfile)
        print('model type:', self.model_type)
        print('model scaler:', self.scaler)
        print('model input dim:', self.input_dim)
        print('model output dim:', self.output_dim)
        print('model medium bound:', self.mediumbound)
        print('timestep:', self.timestep)
        if self.trainingfile:
            if os.path.isfile(self.trainingfile+'.npz'):
                print('training set size', self.X.shape, self.Y.shape)
        else:
             print('no training set provided')
        if self.n_hidden > 0:
            print('nbr hidden layer:', self.n_hidden)
            print('hidden layer size:', self.hidden_dim)
            print('activation function:', self.activation)
        if self.model_type == 'AMN_QP' and self.timestep > 0:
            print('gradient learn rate:', self.learn_rate)
            print('gradient decay rate:', self.decay_rate)
        if self.epochs > 0:
            print('training epochs:', self.epochs)
            print('training regression:', self.regression)
            print('training learn rate:', self.train_rate)
            print('training dropout:', self.dropout)
            print('training batch size:', self.batch_size)
            print('training validation iter:', self.niter)
            print('training xfold:', self.xfold)
            print('training early stopping:', self.es)
        if filename != '':
            sys.stdout.close()

class RC_Model:
    # To save, load & print RC models
    def __init__(self,
                 reservoirfile=None, # reservoir file (a Neural_Model)
                 scaler=False,
                 X=[], # X training data
                 Y=[], # Y training data
                 model=None, # the actual Keras model
                 input_dim=0, output_dim=0, # model IO dimensions
                 # for prior network in RC model
                 # default is n_hidden_prior=-1: no prior network
                 n_hidden_prior=-1, hidden_dim_prior=-1, activation_prior='relu',
                 # for post network in RC model
                 # defaulf is n_hidden_post=-1: no post network
                 n_hidden_post=-1, hidden_dim_post=-1, activation_post='linear',
                 # for all trainable models adam default learning rate = 1e-3
                 regression=True,
                 epochs=0, train_rate=1e-3, dropout=0.25, batch_size=100,
                 niter=0, xfold=5, # cross validation
                 es=False, # early stopping
                 verbose=False
                ):
        if reservoirfile == None:
            sys.exit('must provide a reservoir file')
        if len(X) < 1 or len(Y) < 1:
            sys.exit('must provide X and Y arrays')

        # Training parameters
        self.reservoirfile = reservoirfile
        self.scaler = float(scaler) # From bool to float
        self.X = X
        self.Y = Y
        self.input_dim = X.shape[1]
        self.output_dim = Y.shape[1]
        self.epochs = epochs
        self.regression = regression
        self.train_rate = train_rate
        self.dropout = dropout
        self.batch_size = batch_size
        self.niter = niter
        self.xfold = xfold
        self.es = es

        # Create reservoir
        self.res = Neural_Model()
        self.res.load(reservoirfile)
        self.res.get_parameter(verbose=verbose)

        # Get matrices for loss computation
        self.S = self.res.S # Stoichiometric matrix
        self.Pin = self.res.Pin # Boundary matrix from reaction to medium
        self.Pout = self.res.Pout # Measurement matrix from reactions to measures
        self.mediumbound = self.res.mediumbound

        # Set RC model type
        if 'AMN' in self.res.model_type:
            self.model_type = 'RC'
        else:
            sys.exit('AMN is the only reservoir type handled with RC')
        self.prior, self.post = None, None

        # Set prior network
        if n_hidden_prior > -1:
            if self.mediumbound == 'UB':
                # input is only the variable part of the medium (levmed>1)
                input_dim  = self.input_dim
                for i in range(len(self.res.levmed)):
                    if self.res.levmed[i] == 1:
                        input_dim = input_dim - 1
                        # Scale X with valmed now done in input_RC
                        # self.X[:,i] = self.res.valmed[i]
                output_dim = input_dim
            else:
                input_dim, output_dim = self.input_dim, self.res.input_dim
            # Create prior network
            self.prior = Neural_Model(model_type = 'ANN_Dense',
                           input_dim=input_dim, output_dim=output_dim,
                           n_hidden = n_hidden_prior, hidden_dim = hidden_dim_prior,
                           activation = activation_prior)

        # Set post network input_dim = output_dim
        # take as input the objective of the reservoir !!
        if n_hidden_post > -1:
            self.post = Neural_Model(model_type = 'ANN_Dense',
                          input_dim=self.output_dim, output_dim=self.output_dim,
                          n_hidden = n_hidden_post, hidden_dim = hidden_dim_post,
                          activation = activation_post)

    def printout(self,filename=''):
        if filename != '':
            sys.stdout = open(filename, 'a')
        print('RC reservoir file:', self.reservoirfile)
        print('RC model type:', self.model_type)
        print('RC scaler:', self.scaler)
        print('RC model input dim:', self.input_dim)
        print('RC model output dim:', self.output_dim)
        print('RC model medium bound:', self.mediumbound)
        print('training set size', self.X.shape, self.Y.shape, self.LB.shape)
        print('reservoir S, Pin, Pout matrices',
              self.S.shape, self.Pin.shape, self.Pout.shape)
        if self.epochs > 0:
            print('RC training epochs:', self.epochs)
            print('RC training regression:', self.regression)
            print('RC training learn rate:', self.train_rate)
            print('RC training dropout:', self.dropout)
            print('RC training batch size:', self.batch_size)
            print('RC training validation iter:', self.niter)
            print('RC training xfold:', self.xfold)
            print('RC training early stopping:', self.es)
        if self.prior:
            print('--------prior network --------')
            self.prior.printout(filename)
        print('--------reservoir network-----')
        self.res.printout(filename)
        if self.post:
            print('--------post network ---------')
            self.post.printout(filename)
        if filename != '':
            sys.stdout.close()
