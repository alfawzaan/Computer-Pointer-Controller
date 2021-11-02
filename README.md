# Computer Pointer Controller

This project uses gaze detection model to control the mouse pointer of your computer. The inference is done using the Intel OpenVINO toolkits.
Inferences are performed through four models. The final output then can be used to control the mouse.
## Project Set Up and Installation

1. Install the intel openVINO toolkit using this link as a guide

     [OpenVINO Toolkit](https://docs.openvinotoolkit.org/latest/index.html)

2. Clone or download the repository into your local machine.

3. Navigate to project's root directory

4. Create and activate a virtual environment using these commands
   on  Mac OS / Linux
    
        virtualenv venv --python=python3.7
        source venv/bin/activate
    
   on Windows  
  
        virtualenv venv --python py37
        venv\Scripts\activate
   ##At the time of this documentation, OpenVINO does not support Python 3.8 and Above 

5. Execute the script 'run_demo.sh' to run a demo

        pip3 install -r requirements.txt

6. Initialize OpenVINO environment

        source /opt/intel/openvino/bin/setupvars.sh -pyver 3.7

7. Download the models which will be download into the intel directory.
        
        python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-binary-0001 -o ../
        python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name head-pose-estimation-adas-0001 -o ../
        python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name landmarks-regression-retail-0009 -o ../
        python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 -o ../
    
Project Structure
        
   <img src="https://github.com/alfawzaan/Computer-Pointer-Controller/blob/master/img/project_structure.png" />    
## Demo
After successfully completing the setup and installation precedures. you are now good to run a demo. From the folder src in the src folder in the root directory of the project, run the blow command.
    
    python3 app.py -fd ../intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 -fl ../intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009 -hp ../intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001 -ge ../intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002 -i video -f ../bin/demo.mp4 -pt 0.5 -d CPU -c frame,stats,gaze

## Documentation

Command Line Arguments:

   <img src="https://github.com/alfawzaan/Computer-Pointer-Controller/blob/master/img/command_args.png" />


## Benchmarks

Using the OpenVINO Deep Learning Workbench, the models were benchmarked and the following outputs were observed.
The benchmark was performed on an Intel(R) Core(TM) i5-3210M CPU @ 2.50GHz using an auto generated dataset

Face Detection FP32:

   <img src="https://github.com/alfawzaan/Computer-Pointer-Controller/blob/master/img/face_detectionFP32.png" />

Gaze Estimation FP32:

   <img src="https://github.com/alfawzaan/Computer-Pointer-Controller/blob/master/img/gaze_estimationFP32.png" />
    
Gaze Estimation FP16:
   <img src="https://github.com/alfawzaan/Computer-Pointer-Controller/blob/master/img/gaze_estimationFP16.png" />

    
Gaze Estimation FP16-INT8:

   <img src="https://github.com/alfawzaan/Computer-Pointer-Controller/blob/master/img/gaze_estimationFP16I8.png" />

Head Pose FP32:

   <img src="https://github.com/alfawzaan/Computer-Pointer-Controller/blob/master/img/head_poseFP32.png" />

Head Pose FP16:

   <img src="https://github.com/alfawzaan/Computer-Pointer-Controller/blob/master/img/head_poseFP16.png" />

Head Pose FP16-INT8:

   <img src="https://github.com/alfawzaan/Computer-Pointer-Controller/blob/master/img/head_poseFP16I8.png" />
    
Facial Landmark FP32:

   <img src="https://github.com/alfawzaan/Computer-Pointer-Controller/blob/master/img/facial_landmarkFP32.png" />
    
Facial Landmark FP16:

   <img src="https://github.com/alfawzaan/Computer-Pointer-Controller/blob/master/img/facial_landmarkFP16.png" />

Facial Landmark FP16-INT8:

   <img src="https://github.com/alfawzaan/Computer-Pointer-Controller/blob/master/img/facial_landmarkFP16I8.png" />
    
## Results

It was observed that models with higher precisions are larger and takes more time to load. The inferences are more accurate than lower precision models. From the benchmark output, it was observed that lower precision model were faster in performing inference than higher precision models, but the are less accurate as compared to higher precision models. 
With a combination of these models, taking note of what trade-off to make between accuracy and speed for the use case will give a better result.


## Stand Out Suggestions

This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
When I used async inference, the inference was observed to perform better. This is because it tries to utilize the cpu for multithreading.

### Edge Cases

1. When the model detected more than 1 faces.
2. The app was observed to have poor performance when there is less light. Which makes it difficult for the model to accurately detect gaze.
