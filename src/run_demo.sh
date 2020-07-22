#!/usr/bin/env bash
pip install -r requirements.txt
mkdir "models"
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-0001 -o models/
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name head-pose-estimation-adas-0001 -o models/
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name facial-landmarks-35-adas-0002 -o models/
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 -o models/
#python src/app.py -fd models/intel/face-detection-adas-0001/FP16/face-detection-adas-0001 -fl models/intel/facial-landmarks-35-adas-0002/FP16/facial-landmarks-35-adas-0002 -hp models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001 -ge models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002 -i video -f bin/demo.mp4 -pt 0.5 -d CPU -v gaze,stats
