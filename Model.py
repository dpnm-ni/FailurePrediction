import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
tf.get_logger().setLevel('INFO')
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, GRU, SimpleRNN
from tensorflow.keras.layers import Dense, Concatenate, Flatten, Input, LSTM, Reshape
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from keras import backend as K
from Learning import generator
import pickle as pkl

embed_size=100

def evaluation(model,test_X,test_y, rnn_based=False,fnn_based=False, threshold=0.8,print_result=False):
    if rnn_based:
        model.reset_states()
        y_pred = model.predict_generator(generator(test_X, test_y), steps= len(test_X))
    elif fnn_based:
        y_pred=model.predict(test_X)
    else:
        y_pred = model.predict_generator(generator(test_X, test_y), steps= len(test_X))
    y_pred=np.where(y_pred>threshold, 1,0)
    test_y=np.array(test_y)
    #y_test=np.where(test_y>0.65, 1,0)
    y_test=np.where(test_y<1, 0,1)
    right=0
    tp=0
    fn=0
    fp=0
    for i in range(len(y_pred)):
        if y_pred[i]==y_test[i]:
            right+=1.0
        if y_pred[i]:
            if y_test[i]:
                tp+=1.0
            else:
                fp+=1.0
        elif y_test[i]:
            fn+=1.0
    if rnn_based:
        return right, tp, fp, fn
    accuracy = right/(len(y_pred)+K.epsilon())
    precision=tp/(tp+fp+ K.epsilon())
    recall=tp/(tp+fn+ K.epsilon())
    f1=2*recall*precision/(recall+precision+K.epsilon())
    if print_result:
        print('tp : %d'%(tp)+' fp : %d'%(fp)+ ' fn : %d'%(fn))
        print('acc : %4f'%(right/(len(y_pred)+ K.epsilon()))+' precision : %4f'%(precision)+
               ' recall : %4f'%(recall)+' f1 : %4f'%(f1))
        print('-----------------------------------------------')
    return accuracy, recall, f1

def save_for_roc(model, X_test, y_test, file_name=''):
    #y_pred = model.predict_generator(generator(X_test, y_test), steps= len(X_test))
    y_pred=model.predict(X_test)
    roc=[y_pred ,y_test]
    if file_name != '':
        file_name='_'+file_name
    with open("models/roc"+file_name+".bin", 'wb') as f:
        pkl.dump(roc, f)
    print("roc data saved")

def KL_dv_loss(y_true, y_pred):
    eps=K.epsilon()
    class_weight=100
    y_true=tf.cast(y_true, tf.float32)
    custom_loss=class_weight*y_true*tf.math.log(y_true/(y_pred+eps)+eps)+(1-y_true)*tf.math.log((1-y_true)/(1-y_pred+eps)+eps)
    return custom_loss

def CNN_model(recurrent_model=None, filter_num=3):
    if recurrent_model:
        model_input=Input(batch_shape=(1,None,1))
    else:
        model_input=Input(shape=(None,1))
    submodels=[]
    for kw in range(3,3+filter_num):
        conv=Conv1D(100, kernel_size=(kw*embed_size,), padding='valid', activation='relu', kernel_regularizer=l2(0.01),strides=embed_size)(model_input)
        conv=GlobalMaxPooling1D()(conv)
        submodels.append(conv)
    z=Concatenate()(submodels)
    z=Dropout(0.3)(z)
    if recurrent_model:
        z=Reshape(target_shape=((1,300)))(z)
        if recurrent_model=='gru':
            z=GRU(128, stateful=True,batch_input_shape=(1,300,1), dropout=0.2, recurrent_dropout=0.2)(z)
        elif recurrent_model=='lstm':
            z=LSTM(128, stateful=False)(z)
        elif recurrent_model=='rnn':
            z=SimpleRNN(128,stateful=True)(z)
        #Output shape is (Batchsize, 128)
        z=Dropout(0.3)(z)
        model_output=Dense(1,activation='sigmoid')(z)
        #Output shape is (Batchsize, 1)
    else:
        z=Dense(100, activation='relu', kernel_regularizer=l2(0.01))(z)
        z=Dropout(0.3)(z)
        model_output=Dense(1,activation='sigmoid')(z)
    model=Model(model_input, model_output)
    #model.summary()
    model.compile(optimizer='adam', loss = KL_dv_loss, metrics = ['acc'])
    return model

def FNN_model(input_len):
    #FNN need input with fixed dimension.
    #Please use data with padded.
    model_input=Input(shape=(input_len,))
    z=Dense(100,activation='relu',kernel_regularizer=l2(0.1))(model_input)
    z=Dropout(0.9)(z)
    #z=Dense(50, activation='relu', kernel_regularizer=l2(0.01))(z)
    #z=Dropout(0.7)(z)
    model_output=Dense(1,activation='sigmoid')(z)
    model=Model(model_input, model_output)
    #model.summary()
    model.compile(optimizer='adam', loss = KL_dv_loss, metrics = ['acc'])
    return model

def RNN_model():
    model_input=Input(shape=(None,1))
    z=SimpleRNN(256,activation='relu',kernel_regularizer=l2(0.01))(model_input)
    z=Dense(100, activation='relu', kernel_regularizer=l2(0.01))(z)
    z=Dropout(0.3)(z)
    model_output=Dense(1,activation='sigmoid')(z)
    model=Model(model_input, model_output)
    #model.summary()
    model.compile(optimizer='adam', loss = KL_dv_loss, metrics = ['acc'])
    return model

def GRU_model():
    model_input=Input(shape=(None,1))
    z=GRU(256,activation='relu',kernel_regularizer=l2(0.01))(model_input)
    z=Dense(100, activation='relu', kernel_regularizer=l2(0.01))(z)
    z=Dropout(0.3)(z)
    model_output=Dense(1,activation='sigmoid')(z)
    model=Model(model_input, model_output)
    #model.summary()
    model.compile(optimizer='adam', loss = KL_dv_loss, metrics = ['acc'])
    return model

