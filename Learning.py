import nltk
import argparse
import yaml
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from keras import backend as K
from sklearn.model_selection import KFold
from datetime import datetime, timedelta
import Model
from DataReading import *

vnf_list=[
    {'vnf_num' : 1, 'vnf_id' : 'b21b38dc-eee0-4f27-9863-1b600705a126', 'vnf_name' : '226-4c-1', 'ip': '10.10.10.226'},
    {'vnf_num' : 2, 'vnf_id' : '3f97a1f5-f766-4b9c-89dd-ef7f8ef9adb7', 'vnf_name' : '225-4c-1', 'ip' : '10.10.10.148'},
    {'vnf_num' : 3, 'vnf_id' : '65d6314d-c3cc-46d0-9856-67fc95b953f7', 'vnf_name' : '225-2c-1', 'ip' : '10.10.10.126'},
    {'vnf_num' : 4, 'vnf_id' : '75389a36-510b-41ac-9a19-380b9716c30c', 'vnf_name' : '225-2c-2', 'ip' : '10.10.10.14'},
    {'vnf_num' : 5, 'vnf_id' : 'cbe5d6e7-a490-4326-aea4-8035dc8b3d46', 'vnf_name' : '225-2c-3', 'ip' : '10.10.10.124'}]
#    {'vnf_num' : 6, 'vnf_id' : 'dc5422d7-9e9d-4dc4-a381-5e2bcc986667', 'vnf_name' : '225-2c-4', 'ip' : '10.10.10.26'}]
    #server
#    {'vnf_num' : 225, 'vnf_id' : 'server', 'vnf_name' : 'dpnm-82-225', 'ip' : ''},
#    {'vnf_num' : 226, 'vnf_id' : 'server', 'vnf_name' : 'dpnm-82-226', 'ip' : ''}]
len_fault=lambda x:sum(np.where(np.array(x)>0.8, 1,0))

def generator(inputs, labels):
    #to fit with different input dimension
    #https://github.com/keras-team/keras/issues/1920#issuecomment-410982673
    i = 0
    while True:
        inputs_batch = np.expand_dims([inputs[i%len(inputs)]], axis=2)
        labels_batch = np.array([labels[i%len(inputs)]])
        yield inputs_batch, labels_batch
        i+=1

def model_learning_with_validation(model,X_train,y_train,X_valid,y_valid,class_overweight=1, verbose=0):
    if len(X_train)==0:
        return model
    assert len(X_train)==len(y_train)
    assert len(X_valid)==len(y_valid)
    fault_len=len_fault(y_train)
    if fault_len==0:
        class_weight={0:0.3,1:1}
    else:
        weight_for_0=(1/(len(X_train)-fault_len))*len(X_train)
        weight_for_1=(1/(fault_len))*len(X_train)*class_overweight
        class_weight={0:weight_for_0, 1:weight_for_1}
    #shuffle the data
    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 3)
    mc = ModelCheckpoint('best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 0, save_best_only = True)
    model.fit_generator(
        generator(X_train, y_train),
        validation_data=generator(X_valid, y_valid),
        steps_per_epoch=len(X_train),
        validation_steps=len(X_valid),
        epochs=1000, verbose=verbose, callbacks=[es,mc],
        class_weight=class_weight)
    return model

def model_learning(model,X,y, fault_len,class_overweight=1, verbose=0):
    assert len(X)==len(y)
    if fault_len==0:
        class_weight={0:0.2,1:1}
    else:
        weight_for_0=(1/(len(X)-fault_len))*len(X)
        weight_for_1=(1/(fault_len))*len(X)*class_overweight
        class_weight={0:weight_for_0, 1:weight_for_1}
    es = EarlyStopping(monitor = 'loss', mode = 'min', verbose = 1, patience =5)
    mc = ModelCheckpoint('best_model.h5', monitor = 'acc', mode = 'max', verbose = 0, save_best_only = True)
    model.fit_generator(generator(X, y), steps_per_epoch=len(X), epochs=1000, verbose=verbose, callbacks=[mc,es], class_weight=class_weight)
    return model

def CNN_learning(model,gap,win_size):
    test_all_X=[]
    test_all_y=[]
    all_fault_len=0
    today=datetime.today().strftime("%m-%d")
    today=("05-24")
    for vnf in vnf_list:
        vnf_=vnf['vnf_name']
        date_list=date_range('05-11',today, vnf_)
        fault_=get_fault_history(vnf['vnf_num'])
        print('Reading %d th VNF, %s'%(vnf['vnf_num'], vnf_))
        train_X, train_y, test_X, test_y, train_fault_len, test_fault_len =make_data(
                vnf,fault_,win_size,gap,date_list, over_sampling=10,under_sampling=30)
        model=model_learning_with_validation(model, train_X,train_y, train_fault_len, class_overweight=4, verbose=0)
        if len(test_X)>0:
            test_all_X.extend(test_X)
            test_all_y.extend(test_y)
            #loss, accuracy, f1_score, precision, recall = model.evaluate_generator(
            #        generator(test_X, test_y), steps= len(test_X))
            #print("original F1 : %.4f, Acc : %.4f, Prec : %.4f, Rec : %.4f"%(f1_score, accuracy, precision, recall))
            Model.evaluation(model, test_X, test_y)
        all_fault_len+=fault_len
        model.save("models/CNN_gap%d_win%d"%(gap, win_size))
        print("model saved")
    #loss, accuracy, f1_score, precision, recall = model.evaluate_generator(
    #        generator(test_all_X, test_all_y), steps= len(test_all_X))
    print("-------Learning End. Final performance is here---------")
    print("total test data len is %d and total fault len is %d"%(len(test_all_X),sum(np.where(np.array(test_all_y)>0.8, 1,0))))
    #print("original F1 : %.4f, Acc : %.4f, Prec : %.4f, Rec : %.4f"%(f1_score, accuracy, precision, recall))
    Model.evaluation(model,test_all_X,test_all_y)
    data=[test_all_X,test_all_y]
    with open("models/cnn_data_win%d_gap%d.bin"%(win_size,gap), 'wb') as f:
        pkl.dump(data, f)
    print("test data saved")

def CNN_learning_allinone(gap,win_size):
    X=[]
    y=[]
    test_all_X=[]
    test_all_y=[]
    all_train_fault_len=0
    all_test_fault_len=0
    today=datetime.today().strftime("%m-%d")
    today=("05-24")
    for vnf in vnf_list:
        vnf_=vnf['vnf_name']
        date_list=date_range('05-11',today, vnf_)
        fault_=get_fault_history(vnf['vnf_num'])
        print('Reading %d th VNF, %s'%(vnf['vnf_num'], vnf_))
        train_X,train_y, test_X, test_y, train_fault_len, test_fault_len= make_data(
                vnf,fault_,win_size,gap,date_list, over_sampling=10,under_sampling=30)
        X.extend(train_X)
        y.extend(train_y)
        test_all_X.extend(test_X)
        test_all_y.extend(test_y)
        all_train_fault_len+=train_fault_len
        all_test_fault_len+=test_fault_len
    X=np.array(X)
    y=np.array(y)
    skf = KFold(n_splits=5, shuffle=True)
    best_model=Model.CNN()
    best_f1=0
    for train, valid in skf.split(X,y):
        model =model_learning_with_validation(best_model,
                X[train], y[train], X[valid], y[valid], class_overweight=2.5, verbose=0)
        f1_tmp=Model.f1_eval(model,test_all_X, test_all_y)
        print(f1_tmp)
        if f1_tmp>best_f1:
            best_f1=f1_tmp
            best_model=model
    print("-------Learning End. Final performance is here---------")
    print("total test data len is %d and %d of them are fault"%(len(test_all_X),all_test_fault_len))
    Model.evaluation(best_model,test_all_X,test_all_y)
    data=[X,y, test_all_X, test_all_y]
    with open("models/cnn_data_win%d_gap%d.bin"%(win_size,gap), 'wb') as f:
        pkl.dump(data, f)
    print("data saved")

def CNN_learning_with_data(model,gap,win_size,data):
    all_X,all_y = data
    fault_len=sum(np.where(np.array(all_y)>0.8, 1,0))
    model, test_X,test_y=model_learning_with_validation(model, all_X,all_y, fault_len, class_overweight=2, verbose=0)
    print("-------Learning End. Final performance is here---------")
    print("test data len is %d and %d of them are fault"%(len(test_X),sum(np.where(np.array(test_y)>0.8,1,0))))
    Model.evaluation(model,test_X,test_y)

def CNN_learning_with_data_include_test(gap,win_size,data):
    train_X,train_y , test_X, test_y= data
    train_X=np.array(train_X)
    train_y=np.array(train_y)
    s=np.arange(len(train_X))
    np.random.shuffle(s)
    train_X=train_X[s]
    train_y=train_y[s]
    skf = KFold(n_splits=5, shuffle=True)
    best_model=Model.CNN()
    best_f1=0
    for train, valid in skf.split(train_X,train_y):
        model =model_learning_with_validation(best_model,
                train_X[train], train_y[train], train_X[valid], train_y[valid], class_overweight=2, verbose=0)
        f1_tmp=Model.f1_eval(model,test_X, test_y)
        print(f1_tmp)
        if f1_tmp>best_f1:
            best_f1=f1_tmp
            best_model=model
    print("-------Learning End. Final performance is here---------")
    print("total test data len is %d and %d of them are fault"%(len(test_X),len_fault(test_y)))
    Model.evaluation(best_model,test_X,test_y)

def CRNN_learning(model):
    gap=3
    win_size=5
    today=datetime.today().strftime("%m-%d")
    for vnf in vnf_list:
        vnf_=vnf['vnf_name']
        date_list=date_range('05-11',today, vnf_)
        random.shuffle(date_list)
        #Seperate dates as 8:2 to learning and test
        date_learning=date_list[:int(0.8*len(date_list))]
        date_test=date_list[int(0.8*len(date_list)):]
        fault_=get_fault_history(vnf['vnf_num'])
        print('Reading %d th VNF, %s'%(vnf['vnf_num'], vnf_))
        #TODO: make_data function changed.
        X,y, fault_len= make_data(vnf, fault_, win_size,gap,date_learning,sliding=5,use_emptylog=True, under_sampling=2)
        model=model_learning(model, X,y,fault_len,  verbose=1)
        test_X,test_y, _= make_data(vnf, fault_, win_size,gap,date_test,sliding=5,use_emptylog=True, under_sampling=4)
        if not  len(test_X)==0:
            loss, accuracy, f1_score, precision, recall = model.evaluate_generator(
                    generator(test_X, test_y), steps= len(test_X))
            print("F1 : %.4f, Acc : %.4f, Prec : %.4f, Rec : %.4f"%(f1_score, accuracy, precision, recall))
        model.save("models/CRNN_gap%d_win%d"%(gap, win_size))
        print("model saved")

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--type', type=str, help='ML type. CNN or CRNN')
    parser.add_argument('--model', type=str, help='model file path')
    parser.add_argument('--emb', type=str, help='Word Embedding file path', default='embedding_with_log')
    parser.add_argument('--use-data', action='store_true',help='if you want to use saved data')
    args=parser.parse_args()
    model_file=False
    if args.model:
        #model = load_model(args.model, custom_objects={"f1_m":Model.f1_m, "precision_m":Model.precision_m, "recall_m":Model.recall_m})
        model = load_model(args.model, custom_objects={"f1_m":Model.f1_m, "precision_m":Model.precision_m, "recall_m":Model.recall_m})
    if args.type =='cnn':
        gap=3
        win_size=5
        print("-------------gap : %d  , win_size : %d Start -------------"%(gap,win_size))
        if not args.use_data:
            CNN_learning_allinone(gap,win_size)
        else:
            with open("models/cnn_data_win%d_gap%d.bin"%(win_size,gap), 'rb') as f:
                data=pkl.load(f)
            CNN_learning_with_data_include_test(gap,win_size,data)
    elif args.type=='crnn':
        CRNN_learning(model)
