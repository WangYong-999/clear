#!/bin/bash

# set working root and number of scene
# cd /home/ubuntu/Users/yong/.clear/DREDS/DepthSensorSimulator
cd /home/ubuntu/Users/yong/.clear/bioclear/utils

# run renderer.py
# scene id: 0~2999
mycount=0;
while (( $mycount < 1 )); do
    /home/ubuntu/Users/yong/.clear/bioclear/thirty/blender-2.93.3-linux-x64/blender /home/ubuntu/Users/yong/.clear/DREDS/DepthSensorSimulator/material_lib_v2.blend --background --python render.py -- $mycount;
((mycount=$mycount+1));
done;