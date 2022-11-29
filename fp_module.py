from __future__ import print_function
import ni_nfvo_client
import ni_mon_client
from datetime import datetime, timedelta
from ni_nfvo_client.rest import ApiException as NfvoApiException
from ni_mon_client.rest import ApiException as NimonApiException
from pprint import pprint
from config import cfg
from tensorflow.keras.models import load_model
from DataReading import *
from Model import *
import torch.nn.functional as F
from Learning import generator
import torch
import torch.nn as nn
from transformers import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

import random
import numpy as np
import datetime as dt

import csv
import subprocess
import json

class CNN(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim, dropout=0.2):
        super().__init__()
        self.convs= nn.ModuleList([
            nn.Conv1d(in_channels = embedding_dim, out_channels = n_filters,
                    kernel_size = fs, stride=embedding_dim) for fs in filter_sizes
            ])
        self.fc=nn.Linear(len(filter_sizes)*n_filters, output_dim)
        self.dropout=nn.Dropout(dropout)
    def forward(self, embedding):
        conved=[F.relu(conv(embedding)) for conv in self.convs]
        pooled=[F.max_pool1d(conv,conv.shape[2]).squeeze(2) for conv in conved]
        cat=self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)
    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

# get_monitoring_api(): get ni_monitoring_client api to interact with a monitoring module
# Input: null
# Output: monitoring moudle's client api
def get_monitoring_api():

    ni_mon_client_cfg = ni_mon_client.Configuration()
    ni_mon_client_cfg.host = cfg["ni_mon"]["host"]
    ni_mon_api = ni_mon_client.DefaultApi(ni_mon_client.ApiClient(ni_mon_client_cfg))

    return ni_mon_api

# get_vnf_info(sfc_prefix, sfc_vnfs): get each VNF instance ID and information from monitoring module
# Input: Prefix of VNF instance name, SFC order tuple [example] ("client", "firewall", "dpi", "ids", "proxy")
# Output: Dict. object = {'vnfi_info': vnfi information, 'num_vnf_type': number of each vnf type}
def get_vnf_info(sfc_prefix, sfc_vnfs):

    # Get information of VNF instances which are used for SFC
    ni_mon_api = get_monitoring_api()
    query = ni_mon_api.get_vnf_instances()

    selected_vnfi = [ vnfi for vnfi in query for vnf_type in sfc_vnfs if vnfi.name.startswith(sfc_prefix + vnf_type) ]
    node_ids = [ vnfi.node_id for vnfi in selected_vnfi ]
    node_ids = list(set(node_ids))

    vnfi_list = []
    num_vnf_type = []
    temp = []

    # Sort VNF informations for creating states
    for vnf_type in sfc_vnfs:
        i =  sfc_vnfs.index(vnf_type)

        temp.append([])

        temp[i] = [ vnfi for vnfi in selected_vnfi if vnfi.name.startswith(sfc_prefix + vnf_type) ]
        temp[i].sort(key=lambda vnfi: vnfi.name)

        for vnfi in temp[i]:
            vnfi.node_id = node_ids.index(vnfi.node_id)

        vnfi_list = vnfi_list + temp[i]
        num_vnf_type.append(len(temp[i]))

    return {'vnfi_list': vnfi_list, 'num_vnf_type': num_vnf_type}


# get Failure prediction of 5 minutes results in real-time
# input: vnf_instance name prefix, VNF types
# Output: result (string) 
def get_failure_prediction_result(prefix, sfc_vnfs):
    win_size=5
    gap=5
    result=""
    model_path='/home/dpnm/tmp/runs/best_model_0.8235288650522071'
    bert_path='/home/dpnm/tmp/runs/best_model_bert_0.8235288650522071'
    model_name = "bert-base-uncased"
    device = torch.device("cpu")
    transformer = AutoModel.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    hidden_size = int(config.hidden_size)
    model = CNN(hidden_size, n_filters, filter_sizes, 1)
    if torch.cuda.is_available():
        model.cuda()
        transformer.cuda()
    model.load_state_dict(torch.load(model_path, map_location=device))
    transformer.load_state_dict(torch.load(bert_path,  map_location=device))
    result_dict={}
    prediction_result={}
    vnfi_info = get_vnf_info(prefix, sfc_vnfs)
    vnfi_list = vnfi_info["vnfi_list"]
    log_embedding=[read_current_log_for_vnf(vnfi.name.lower().replace('_','-'), win_size=5,use_emptylog=True) for vnfi in vnfi_list]
    y_pred=model.predict_generator(generator(log_embedding, [0 for _ in range(len(log_embedding))]), steps= len(log_embedding))
    for (i, vnfi) in enumerate(vnfi_list):
        if y_pred[i] <0.8:
            prediction_result[vnfi.name]="Normal"
        else:
            prediction_result[vnfi.name]="Normal"
    result_dict['detection_result'] = prediction_result
    result_dict['time'] = dt.datetime.now()

    return result_dict

def get_failure_prediction_result_for_server(prefix, sfc_vnfs):
    win_size=5
    gap=5
    n_filters=40
    filter_sizes = [3,4,5]
    max_kernel=max(filter_sizes)
    result=""
    with open('server_info.yaml') as f:
        conf=yaml.load(f)
    model_path='/home/dpnm/tmp/runs/best_model'
    bert_path='/home/dpnm/tmp/runs/best_model_bert'
    model_name = "bert-base-uncased"
    device = torch.device("cpu")
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    transformer = AutoModel.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    hidden_size = int(config.hidden_size)
    model = CNN(hidden_size, n_filters, filter_sizes, 1)
    if torch.cuda.is_available():
        model.cuda()
        transformer.cuda()
    model.load_state_dict(torch.load(model_path, map_location=device))
    transformer.load_state_dict(torch.load(bert_path,  map_location=device))
    result_dict={}
    prediction_result={}
    
    #vnfi_info = get_vnf_info(prefix, sfc_vnfs)
    #vnfi_list = vnfi_info["vnfi_list"]
    input_log_token=[]
    for vnf in conf:
        if vnf['server'] == prefix:
            input_log_token.extend(read_current_log_for_vnf(vnf['name'], win_size=10))        
    if len(input_log_token) > max_sent_length:
        input_log_token=input_log_token[:max_sent_length]
    tmp_len=len(input_log_token)
    input_log_token.extend([[0 for _ in range(max_seq_length)] for __ in range(max_sent_length-tmp_len)])
    assert len(input_log_token) == max_sent_length
    input_log_token = (tmp_len, input_log_token)
    test_dataloader = generate_data_loader([input_log_token],[0], 1)
    for batch in test_dataloader:
        continue
    model.eval()
    transformer.eval()

    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)
    b_log_mask=batch[3].to(device)

    # Encode real data in the Transformer
    hidden_states=[]
    log_token = b_input_ids[0]
    log_token=log_token[:b_log_mask[0]]
    hidden_states=transformer(log_token, attention_mask=b_input_mask[0][:b_log_mask[0]])[0].view([-1,hidden_size])
    if hidden_states.shape[0] < max_kernel:
        hidden_states=F.pad(hidden_states,(0,0,0,max_kernel-hidden_states.shape[0]),'constant', 0)
    hidden_states=torch.transpose(hidden_states,0,1).unsqueeze(0)
    # input of discriminator is 1d vector with 768(hidden_size)*n length

    # Then, we select the output of the disciminator

    # Tell pytorch not to bother with constructing the compute graph during
    # the forward pass, since this is only needed for backprop (training).
    with torch.no_grad():
        predictions = model(hidden_states).squeeze(1)
    results=torch.sigmoid(predictions)[0].detach().cpu().numpy()
    if results <0.8:
        prediction_result[prefix]="Normal"
    else:        
        prediction_result[prefix]="Failure"
    #prediction_result[prefix]=results
    result_dict['detection_result'] = prediction_result
    result_dict['time'] = dt.datetime.now()

    return result_dict

# get VM log of 5 minutes in real-time
# input: vnf_instance name prefix, VNF types
# Output: result (string) 
def get_vm_log(prefix, sfc_vnfs):
    win_size=5
    gap=5
    result=""
    result_dict={}
    log_data={}

    vnfi_info = get_vnf_info(prefix, sfc_vnfs)
    vnfi_list = vnfi_info["vnfi_list"]
    #print(vnfi_list)
    for vnfi in vnfi_list:
        log_data[vnfi.name]=read_current_data(vnfi.name.lower().replace('_','-'))
    result_dict['log_data'] = log_data
    result_dict['time'] = dt.datetime.now()
    return result_dict

def convert_vnf_info(vnfi_list):
    
    response = []

    for i in range(0,len(vnfi_list)):

        instance_dict = {}
        temp = vnfi_list[i].__dict__

        instance_dict['flavor_id'] = temp['_flavor_id']
        instance_dict['id'] = temp['_id']
        instance_dict['node_id'] = temp['_node_id']
        instance_dict['name'] = temp['_name']
        instance_dict['ports'] = convert_network_port_object(temp['_ports'])
        instance_dict['status'] = temp['_status']

        response.append(instance_dict)

    return response


def convert_network_port_object(ports):

    response = []

    for i in range(0,len(ports)):
        port_dict = {}
        temp = ports[i].__dict__
        
        port_dict['ip_addresses'] = temp['_ip_addresses']
        port_dict['network_id'] = temp['_network_id']
        port_dict['port_id'] = temp['_port_id']
        port_dict['port_name'] = temp['_port_name']

        response.append(port_dict)

    return response



