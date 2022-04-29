# FailurePrediction
VNF failure prediction with log 

## Usage

After installation and configuration of this module, you can run this module by using the command as follows.

```
sudo python3 -m swagger_server
```

This module provides web UI based on Swagger (default port number is 8006):

```
http://<host IP running this module>:<port number>/ui/
```

To detect the VNFs' real-time status in OpenStack testbed, this module processes a HTTP GET message including in its body.
You can generate an request by using web UI or using other library creating HTTP messages.

Required data to create HTTP request is VNF instances' prefix.
(We assume that the prefix of the VNF instances' name that consists of the SFC is the same.)

- **prefix**: a prefix to identify instances which can be components of an SFC from OpenStack