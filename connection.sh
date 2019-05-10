#!/bin/bash

# Change the IP address (or machine name) with each restart.

ADDR=3.14.251.105             # change each time you start your machine to its current dns name or ip address
NAME=ubuntu
LHOST=localhost
SSHKEY="/Users/yuxuan/Desktop/yuxuanxray.pem"          # change if necessary to the name of your private key file

for i in `seq 8888 8900`; do
    FORWARDS[$((2*i))]="-L"
    FORWARDS[$((2*i+1))]="$i:${LHOST}:$i"
done

ssh -i ${SSHKEY} -X ${FORWARDS[@]} -l ${NAME} ${ADDR}