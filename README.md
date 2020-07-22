# Computer Pointer Controller

*TODO:* Write a short introduction to your project
This project uses gaze detection model to control the mouse pointer of your computer. The inference is done using the Intel openVINO toolkits.

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

1. Install the intel openVINO toolkit using this link as a guide

     [OpenVINO Toolkit](https://docs.openvinotoolkit.org/latest/index.html)

2. Clone or download the repository into your local machine.

3. Navigate to the folder src in the project's root directory

4. Create and activate a virtual environment using these commands
   on  Mac OS / Linux
    
        virtualenv venv --python=python3.7
        source venv/bin/activate
    
   on Windows  
  
        virtualenv venv --python py37
        venv\Scripts\activate

5. Execute the script 'run_demo.sh' to run a demo

        pip3 install -r requirements.txt

6. Initialize OpenVINO environment

        source /opt/intel/openvino/bin/setupvars.sh -pyver 3.7

7. Download the models which will be download into the intel directory.
        
        python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-binary-0001 -o ../
        python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name head-pose-estimation-adas-0001 -o ../
        python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name landmarks-regression-retail-0009 -o ../
        python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 -o ../
    

## Demo
*TODO:* Explain how to run a basic demo of your model.
After successfully completing the setup and installation precedures. you are now good to run a demo. From the folder src in the root directory of the project, run the blow command.
    
    python3 src/app.py -fd ../intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 -fl ../intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009 -hp ../intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001 -ge ../intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002 -i video -f bin/demo.mp4 -pt 0.5 -d CPU -v gaze,stats

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

Command Line Arguments:

   <img src="https://github.com/alfawzaan/Computer-Pointer-Controller/blob/master/img/command_args.png" />


## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardware and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

Using the OpenVINO Deep Learning Workbench, the models were benchmarked and the following outputs were observed.
The benchmark was performed on an Intel(R) Core(TM) i5-3210M CPU @ 2.50GHz

Face Detection FP32:

   ![]("imgs/face_detectionFP32.png")

Gaze Estimation FP32:

   ![]("/imgs/gaze_estimationFP32.png")
    
Gaze Estimation FP16:

   ![]("/imgs/gaze_estimationFP16.png")
    
Gaze Estimation FP16-INT8:

   ![]("/imgs/gaze_estimationFP16I8.png")


Head Pose FP32:

   ![]("/imgs/head_poseFP32.png")

Head Pose FP16:

   ![]("/imgs/head_poseFP16.png")

Head Pose FP16-INT8:

   ![]("/imgs/head_poseFP16I8.png")
    
Facial Landmark FP32:

   ![]("/imgs/facial_landmarkFP32.png")
    
Facial Landmark FP16:

   ![]("/imgs/facial_landmarkFP16.png")

Facial Landmark FP16-INT8:

   ![]("/imgs/facial_landmarkFP16I8.png")
    
## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.
When I used async inference, the inference was observed to perform better. This is because it tries to utilize the cpu for multithreading.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
Some of the edge cases I encountered are:

1. When the model detected more than 1 faces.
2. The app was observed to have poor performance when there is less light. Which makes it difficult for the model to accurately detect gaze.
