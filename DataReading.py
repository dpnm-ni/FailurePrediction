from tqdm import trange
import string
import pickle as pkl
from string import digits
from datetime import datetime, timedelta
from influxdb import DataFrameClient
import numpy as np
import pandas as pd
import yaml
import scp
import paramiko
from scp import SCPClient, SCPException
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from transformers import AutoTokenizer
import torch
from torch.utils.data import SequentialSampler, TensorDataset, DataLoader
import os
import re
import random
import time

local_path='/home/dpnm/tmp/'
remote_path='/home/dpnm/log/hosts/'
server_info_path='/home/dpnm/server_info.yaml'
max_seq_length = 50
max_sent_length=350

#read data in real time. 
def read_current_log_for_vnf(vnf, win_size, use_emptylog=False, add_oov=False,use_wv=False):
    if use_wv:
        wv= Word2Vec.load('model/embedding_with_log')
    else:
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f'Read last {win_size} min of log data of {vnf}')
    embed_size=100
    same_limit=5
    today=datetime.today().strftime("%m-%d")
    path_=remote_path+vnf+'/'+today+'/'
    log_file_list=get_file_list(path_)
    #print(log_file_list)
    log_corpus=[]
    for file_name in log_file_list:
        if file_name in ['sudo.log', 'CRON.log', 'stress-ng.log']:
            continue
        log = download_log(path_,file_name,local_path)
        if file_name=='kernel.log':
            log_token = pre_process_error_only(log)
        else:
            log_token=pre_process(log)
        log_corpus.extend(log_token)

    #Delete Same log. This could be dangerous.
    log_corpus=sorted(list(set(log_corpus)), key= lambda x : x[0])
    date = datetime.now()-timedelta(minutes=win_size)
    input_log=[]
    sentence_pool=[]
    for log in log_corpus:
        if date<log[0]:
            continue
        if log[0] > date+timedelta(minutes=win_size):
            break
        ######If first words are same, it seems similar log. pass it####
        already_in_sentence_pool=False
        for sentence in sentence_pool:
            if sentence==list(log[1])[:same_limit]:
                already_in_sentence_pool=True
                break
        if already_in_sentence_pool:
            continue
        sentence_pool.append(list(log[1])[:same_limit])
        ######Checking similar log end####
        for word in log[1]:
            try:
                input_log.extend(wv.wv.get_vector(word))
            except:
                if add_oov:
                    input_log.extend(np.zeros(embed_size))
                else:
                    pass #Do not add OOV
    if use_emptylog and len(input_log) ==0:
        input_log=np.zeros(5*embed_size)
    assert len(input_log)%embed_size==0
    return input_log

def read_current_log_BERT(vnf, win_size):
    print(f'Read last {win_size} min of log data of {vnf}')
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    same_limit=5
    today=datetime.today().strftime("%m-%d")
    today='05-21'
    path_=remote_path+vnf+'/'+today+'/'
    log_file_list=get_file_list(path_)
    log_corpus=[]
    for file_name in log_file_list:
        if file_name in ['sudo.log', 'CRON.log', 'stress-ng.log']:
            continue
        log = download_log(path_,file_name,local_path)
        if file_name=='kernel.log':
            log_token = pre_process_error_only(log)
        else:
            log_token=pre_process(log)
        log_corpus.extend(log_token)
    #Delete Same log. 
    log_corpus=sorted(list(set(log_corpus)), key= lambda x : x[0])
    date = datetime.now()-timedelta(minutes=win_size)
    date = '05-21 22:15'.strptime('%m-%d %H:%M')
    input_log=[]
    sentence_pool=[]
    for log in log_corpus:
        if date<log[0]:
            continue
        if log[0] > date+timedelta(minutes=win_size):
            break
        ######If first words are same, it seems similar log. pass it####
        already_in_sentence_pool=False
        for sentence in sentence_pool:
            if sentence==list(log[1])[:same_limit]:
                already_in_sentence_pool=True
                break
        if already_in_sentence_pool:
            continue
        sentence_pool.append(list(log[1])[:same_limit])
        sentence_token = tokenizer.encode(log[1], is_split_into_words=True,add_special_tokens=True,
            max_length=max_seq_length, padding="max_length", truncation=True)
        input_log.append(sentence_token)
    return input_log

def read_data(vnf, fault_, win_size, gap, date_list, sliding=1, use_emptylog=False):
    wv= Word2Vec.load('model/embedding_with_log')
    embed_size=100
    same_limit=5
    normal_X=[]
    normal_y=[]
    fault_X=[]
    fault_end_date=None
    for j in trange(len(date_list)):
        log_dir = date_list[j]
        if fault_end_date:
            if datetime.strptime(log_dir,'%m-%d') < datetime.strptime(fault_end_date[:fault_end_date.find('-',4)],'%b-%d'):
                continue
        path_=remote_path+vnf['vnf_name']+'/'+log_dir+'/'
        log_file_list=get_file_list(path_)
        log_corpus=[]
        for file_name in log_file_list:
            if vnf['vnf_id']=='server':
                if file_name in ['sudo.log', 'CRON.log', 'nova-compute.log', 'neutron-openvswitch-agent.log', 'apache-access.log']:
                    continue
            elif file_name in ['sudo.log', 'CRON.log', 'stress-ng.log']:
                continue
            log = download_log(path_,file_name,local_path)
            if file_name=='kernel.log':
                log_token = pre_process_error_only(log)
            else:
                log_token=pre_process(log)
            log_corpus.extend(log_token)

        #Delete Same log. This could be dangerous.
        log_corpus=sorted(list(set(log_corpus)), key= lambda x : x[0])
        if fault_end_date:
            date=datetime.strptime(fault_end_date,'%b-%d-%H:%M')
            fault_end_date=None
        else:
            date=datetime.strptime(log_dir,'%m-%d')
        while(date<datetime.strptime(log_dir,'%m-%d')+timedelta(days=1)):
            input_log=[]
            sentence_pool=[]
            for log in log_corpus[:]:
                if date>log[0]:
                    log_corpus.remove(log)
                if log[0] > date+timedelta(minutes=win_size):
                    break
                ######If first words are same, it seems similar log. pass it####
                already_in_sentence_pool=False
                for sentence in sentence_pool:
                    if sentence==list(log[1])[:same_limit]:
                        already_in_sentence_pool=True
                        break
                if already_in_sentence_pool:
                    continue
                sentence_pool.append(list(log[1])[:same_limit])
                ######Checking similar log end####
                for word in log[1]:
                    try:
                        input_log.extend(wv.wv.get_vector(word))
                    except:
                        pass #Do not add OOV
                        #input_log.extend(np.zeros(embed_size))
            if use_emptylog and len(input_log) ==0:
                input_log=np.zeros(5*embed_size)
            assert len(input_log)%embed_size==0
            abnormal=False
            fault=False
            for i in range(sliding):
                if (date+timedelta(minutes=gap+win_size+i)).strftime('%b-%d-%H:%M') in fault_['abnormal']:
                    abnormal=True
                    break
                for fault_range in fault_['fault']:
                    if (date+timedelta(minutes=gap+win_size+i)).strftime('%b-%d-%H:%M') == fault_range['start']:
                        fault=True
                        if 'end' in fault_range:
                            fault_end_date=fault_range['end']
                        break
                if fault:
                    break
            if not len(input_log)==0:
                if len(input_log)<5*embed_size:
                    input_log.extend(np.zeros(5*embed_size-len(input_log)))
                if fault or abnormal:
                    #Not seperate right now.
                    #TODO: seperate learning fault and abnormal
                    fault_X.append(input_log)
                    if fault:
                        if fault_end_date:
                            if datetime.strptime(log_dir,'%m-%d') < datetime.strptime(fault_end_date[:fault_end_date.find('-',4)],'%b-%d'):
                                #Go to next date.
                                break
                            else:
                                date=datetime.strptime(fault_end_date,'%b-%d-%H:%M')
                                fault_end_date=None
                                continue
                        else:
                            print("Read total %d number of data and %d of them are fault"%(len(normal_X)+len(fault_X),len(fault_X) ))
                            return normal_X, normal_y, fault_X
                    date+=timedelta(minutes=gap+win_size+sliding-1) ## Slide to after of abnormal
                    continue
                else:
                    #find in near.
                    abnormal=False
                    fault=False
                    for i in range(5):
                        if (date+timedelta(minutes=gap+win_size+sliding+i)).strftime('%b-%d-%H:%M') in fault_['abnormal']:
                            abnormal=True
                            break
                        for fault_range in fault_['fault']:
                            if (date+timedelta(minutes=gap+win_size+sliding+i)).strftime('%b-%d-%H:%M') == fault_range['start']:
                                fault=True
                                break
                        if fault:
                            break
                    if abnormal or fault:
                        normal_X.append(input_log)
                        normal_y.append(0.6)
                    else:
                        normal_X.append(input_log)
                        normal_y.append(0)
            date+=timedelta(minutes=sliding)
    print("Read total %d number of data and %d of them are fault"%(len(normal_X)+len(fault_X),len(fault_X) ))
    return normal_X,normal_y,fault_X

def make_data_seperate(vnf, fault_, win_size, gap, date_list, sliding=1, use_emptylog=False, over_sampling=1, under_sampling=0):
    #make test data and train data completely seperate
    #use only when faults data are many
    normal_X, normal_y, fault_X = read_data(vnf,fault_,win_size,gap,date_list,sliding,use_emptylog)
    test_X=[]
    test_y=[]
    train_X=[]
    train_y=[]
    train_fault_len=0
    test_fault_len=0
    for i in range(len(normal_X)):
        if random.choice([True ]+[False for _ in range(under_sampling)]):
            if random.choice([True, False, False, False, False]):
                test_X.append(normal_X[i])
                test_y.append(normal_y[i])
            else:
                train_X.append(normal_X[i])
                train_y.append(normal_y[i])
    for i in range(len(fault_X)):
        if random.choice([True, False, False, False, False]):
            test_X.append(fault_X[i])
            test_y.append(1)
            test_fault_len+=1
        else:
            train_X.extend([fault_X[i] for _ in range(over_sampling)])
            train_y.extend([1]*over_sampling)
            train_fault_len+=over_sampling
    return train_X, train_y, test_X, test_y, train_fault_len, test_fault_len

def make_data(vnf, fault_, win_size, gap, date_list, sliding=1, use_emptylog=False, over_sampling=1, under_sampling=0):
    normal_X, normal_y, fault_X = read_data(vnf,fault_,win_size,gap,date_list,sliding,use_emptylog)
    test_X=[]
    test_y=[]
    train_X=[]
    train_y=[]
    train_fault_len=0
    test_fault_len=0
    for i in range(len(normal_X)):
        if random.choice([True ]+[False for _ in range(under_sampling)]):
            if random.choice([True, False, False, False, False]):
                test_X.append(normal_X[i])
                test_y.append(normal_y[i])
            else:
                train_X.append(normal_X[i])
                train_y.append(normal_y[i])
    fault_X=fault_X*over_sampling
    for i in range(len(fault_X)):
        if random.choice([True, False, False, False, False]):
            test_X.append(fault_X[i])
            test_y.append(1)
            test_fault_len+=1
        else:
            train_X.append(fault_X[i])
            train_y.append(1)
            train_fault_len+=1
    assert len(train_X)==len(train_y)
    assert len(test_X)==len(test_y)
    return train_X, train_y, test_X, test_y, train_fault_len, test_fault_len

def pre_process(text):
    #https://blog.naver.com/PostView.nhn?blogId=timtaeil&logNo=221361106051&redirect=Dlog&widgetTypeCall=true&directAccess=false
    #Delete former words of ':' Because it contain dates and host name.
    #Example : Mar 23 06:37:33 225-2c-1 10freedos: debug: /dev/vda15 is a FAT32 partition
    #Remove number, symbol, stop world
    #Tokenizing
    #clean = [[x.lower() for x in each[each.find(':',19)+1:].translate(translator).split() \
    #       if x.lower() not in stop_words] for each in text.split('\n')]
    if text=='':
        return []
    translator = str.maketrans(string.punctuation, ' '*(len(string.punctuation)))
#    text = re.sub(r'[0-9]+', '', text)
    clean =[]
    for each in text.split('\n'):
        if not each:
            continue
        #Todo : How to use number information??
        clean.append(tuple([datetime.strptime(each[:12], "%b %d %H:%M"), tuple([x.lower() for x in  re.sub(r'[0-9]+',
            '', each[each.find(':',19)+1:]).translate(translator).split()] )]))
    return clean
def pre_process_error_only(text):
    if text=='':
        return []
    translator = str.maketrans(string.punctuation, ' '*(len(string.punctuation)))
    clean =[]
    for each in text.split('\n'):
        if (not 'error' in each) and (not 'fail' in each) and (not 'Reset' in each) and (not 'Reset' in each):
            continue
        #Todo : How to use number information??
        clean.append(tuple([datetime.strptime(each[:12], "%b %d %H:%M"), tuple([x.lower() for x in  re.sub(r'[0-9]+',
            '', each[each.find(':',19)+1:]).translate(translator).split() ])]))
    return clean

def download_log(remote_path, file_name, local_path):
    with open (server_info_path) as f:
        server_info=yaml.load(f)['log']
    try:
        cli=paramiko.SSHClient()
        cli.set_missing_host_key_policy(paramiko.AutoAddPolicy)
        cli.connect(server_info['ip'], port=22, username='root', password=server_info['pwd'])
        with SCPClient(cli.get_transport()) as scp:
            scp.get(remote_path+file_name, local_path)
    except SCPException as e:
        print("Operation error : %s"%e)
    try:
        with open(local_path+file_name) as f:
            text = f.read()
    except:
        try:
            with open(local_path+file_name, encoding='ISO-8859-1') as f:
                text = f.read()
        except Exception as e:
            print("Opreation error at reading file %s : %s"%(file_name, e))
    os.remove(local_path+file_name)
    return text

def get_file_list(path):
    with open (server_info_path) as f:
        server_info=yaml.load(f)['log']
    cli=paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    cli.connect(server_info['ip'], port=22, username='root', password=server_info['pwd'])
    stdin, stdout, stderr = cli.exec_command('ls '+path)
    rslt=stdout.read()
    file_list=rslt.split()
    del stdin, stdout, stderr
    file_list = [file_name.decode('utf-8') for file_name in file_list]
    return file_list

def date_range(start, end, vnf_name):
    start=datetime.strptime(start, "%m-%d")
    end = datetime.strptime(end, "%m-%d")
    dates = [(start + timedelta(days=i)).strftime("%m-%d") for i in range((end-start).days+1)]
    log_dir_list=get_file_list(remote_path+vnf_name+'/')
    for date in dates[:]:
        if date not in log_dir_list:
            dates.remove(date)
    return dates

def fault_tagging(vnf_num):
    #Tagging based on Packet Processing Time
    #Not Use
    with open (server_info_path) as f:
        server_info=yaml.load(f)['InDB']
    user, password, host = server_info['id'], server_info['pwd'], server_info['ip']
    client=DataFrameClient(host, 8086,user, password, 'pptmon')
    ppt = client.query('select * from "%d"'%vnf_num)
    ppt=list(ppt.values())[0].tz_convert('Asia/Seoul')
    ppt.index=ppt.index.map(lambda x : x.replace(microsecond=0, second=0))
    ppt.reset_index(inplace = True)
    ppt.rename(columns={'index' : 'time'}, inplace=True)
    fault= ppt[ppt['value']>10000][['time']].values.tolist()
    fault = [x[0].strftime("%m-%d %H:%M") for x in fault]
    return fault

def get_fault_history(vnf_num):
    print("Get Fault history")
    fault_history={}
    #with open (server_info_path) as f:
    #    server_info=yaml.load(f)['FaultHistory']
    #cli=paramiko.SSHClient()
    #cli.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    #cli.connect(server_info['ip'], port=22, username=server_info['id'], password=server_info['pwd'])
    #try:
    #    with SCPClient(cli.get_transport()) as scp:
    #        scp.get('/home/ubuntu/fault_history.yaml',local_path)
    #except SCPException:
    #    raise SCPException.message
    with open(local_path+'fault_history.yaml')as f:
        fault_history=yaml.load(f)
    return fault_history[vnf_num]

def generate_data_loader(input_examples, label_masks, batch_size,balance_label_examples = False,use_wv=False):
    '''
    Generate a Dataloader given the input examples, eventually masked if they are
    to be considered NOT labeled.
    '''
    sampler = SequentialSampler
    if use_wv:
        input_ids = torch.tensor(input_examples,dtype=float)
        label_id_array = torch.tensor(label_masks)
        dataset = TensorDataset(input_ids, label_id_array)
        return DataLoader(
            dataset,  # The training samples.
            sampler = sampler(dataset),
            batch_size = batch_size) # Trains with this batch size.

    examples = []
    # if required it applies the balance
    log_mask=[]
    for index, ex in enumerate(input_examples):
        examples.append((ex[1], label_masks[index]))
        log_mask.append(ex[0])

    #-----------------------------------------------
    # Generate input examples to the Transformer
    #-----------------------------------------------
    input_ids = []
    input_mask_array = []
    label_id_array = []

    for (token_list, label_mask) in examples:
        input_ids.append(token_list)
        att_mask=[]
        for sentence_token in token_list:
            att_mask.append( [int(token_id > 0) for token_id in sentence_token])
        input_mask_array.append(att_mask)
        label_id_array.append(label_mask)

    # Convertion to Tensor
    input_ids = torch.tensor(input_ids)
    input_mask_array = torch.tensor(input_mask_array)
    label_id_array = torch.tensor(label_id_array, dtype=torch.float16)
    log_mask= torch.tensor(log_mask)

    # Building the TensorDataset
    # Still, input_ids is list of token
    dataset = TensorDataset(input_ids, input_mask_array, label_id_array,log_mask)

    # Building the DataLoader
    return DataLoader(
        dataset,  # The training samples.
        sampler = sampler(dataset),
        batch_size = batch_size) # Trains with this batch size.