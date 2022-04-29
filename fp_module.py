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
from Learning import generator

import random
import numpy as np
import datetime as dt

import csv
import subprocess
import json

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


# get SLA detection results in real-time
# input: vnf_instance name prefix, VNF types
# Output: result (string) 

def get_failure_prediction_result(prefix, sfc_vnfs):
    win_size=5
    gap=5
    result=""
    result_dict={}
    prediction_result={}

    vnfi_info = get_vnf_info(prefix, sfc_vnfs)
    vnfi_list = vnfi_info["vnfi_list"]
    #print(vnfi_list)
    log_embedding=[read_current_data(vnfi.name.lower().replace('_','-'), win_size=5,use_emptylog=True) for vnfi in vnfi_list]
    model = load_model('model/gap_5_win_5_best',custom_objects={"KL_dv_loss":KL_dv_loss})
    y_pred=model.predict_generator(generator(log_embedding, [0 for _ in range(len(log_embedding))]), steps= len(log_embedding))
    for (i, vnfi) in enumerate(vnfi_list):
        if y_pred[i] <0.8:
            prediction_result[vnfi.name]="Normal"
        else:
            prediction_result[vnfi.name]="Normal"
    result_dict['detection_result'] = prediction_result
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



