import connexion
import six
#import json

from swagger_server.models.vnf_instance import VNFInstance  # noqa: E501
from swagger_server import util

import fp_module as fp
#from swagger_server.models.ad_info import adInfo
#from swagger_server.models.network_port import NetworkPort

def get_vnf_info(prefix):
    
    sfc_vnfs = ["1_Firewall", "2_ntopng", "3_Haproxy_2"]
    vnf_info = fp.get_vnf_info(prefix, sfc_vnfs)
    vnfi_list = vnf_info["vnfi_list"]
    result = fp.convert_vnf_info(vnfi_list)

    return result

def get_failure_prediction_result(prefix):

    sfc_vnfs = ["1_Firewall", "2_ntopng", "3_Haproxy_2"]
    result = fp.get_failure_prediction_result(prefix, sfc_vnfs)

    return result
