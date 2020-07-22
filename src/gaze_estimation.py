'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import math
import sys
import logging as log
import cv2
from openvino.inference_engine import IENetwork, IECore
import pyautogui


class Model_Gaze_Estimation:
    '''
    Class for the Gaze Estimation Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.5):
        '''
        Setting Instance variables.
        '''
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.threshold = threshold
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.input_shape = None

    def load_model(self):
        '''
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        print("GE In Load")
        model_structure = self.model_name + ".xml"
        model_weight = self.model_name + ".bin"
        self.plugin = IECore()
        self.network = IENetwork(model=model_structure, weights=model_weight)

        if self.extensions is not None and "CPU" in self.device:
            self.plugin.add_extension(self.extensions, self.device)

        self.exec_network = self.plugin.load_network(self.network, self.device)
        self.check_model()
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        self.input_shape = self.network.inputs[self.input_blob].shape

    def predict(self, left_eye, right_eye, gaze):
        '''
        This method is meant for running predictions on the input image.
        '''
        l_eye = self.preprocess_input(left_eye)
        r_eye = self.preprocess_input(right_eye)
        inference_result = self.exec_network.start_async(request_id=0,
                                                         inputs={"left_eye_image": l_eye,
                                                                 "right_eye_image": r_eye,
                                                                 "head_pose_angles": gaze})
        if self.wait() == 0:
            inference_result = self.get_output_result()
        return self.preprocess_output(inference_result, gaze)

    def check_model(self):
        if "CPU" in self.device:
            supported_layers = self.plugin.query_network(self.network, "CPU")
            none_supported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(none_supported_layers) > 0:
            log.error("Following layers are not supported by the plugin for the specified device {}:\n {}".
                      format(self.device, ', '.join(none_supported_layers)))
            log.error(
                "Please try to specify cpu extensions library path in sample's command line parameters using -l "
                "or --cpu_extension command line argument")
            sys.exit(1)

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        try:
            pre_pro_frame = cv2.resize(image, (60, 60))
            print(len(pre_pro_frame))
            pre_pro_frame = pre_pro_frame.transpose((2, 0, 1))
            pre_pro_frame = pre_pro_frame.reshape(1, *pre_pro_frame.shape)
        except:
            from app import write_text_img
            write_text_img([0], "Failed to process both eyes successfully", 400)
            return 0
        return pre_pro_frame

    def preprocess_output(self, outputs, gaze):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        eye_roll = gaze[2]
        outputs = outputs[0]
        print(eye_roll)
        conv_sin = math.sin(eye_roll * math.pi/180)
        conv_cos = math.cos(eye_roll * math.pi/180)
        x = outputs[0] * conv_cos + outputs[1] * conv_sin
        y = outputs[1] * conv_cos - outputs[0] * conv_sin

        return x, y

    def wait(self):
        inf_status = self.exec_network.requests[0].wait(-1)
        return inf_status

    def get_output_result(self):
        return self.exec_network.requests[0].outputs[self.output_blob]
