'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from os import path
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore
import cv2 as cv


class Model_Face_Detection:
    '''
    Class for the Face Detection Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.5):
        '''
        Setting Instance variables.
        '''
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.plugin = None
        self.network = None
        self.exec_network = None
        self.input_blob = None
        self.output_blob = None
        self.input_shape = None
        self.threshold = threshold

    def load_model(self):
        print("FD In Load")
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

    def predict(self, image):
        '''
        This method is meant for running predictions on the input image.
        '''
        pre_pro_img = self.preprocess_input(image)
        self.exec_network.start_async(request_id=0, inputs={self.input_blob: pre_pro_img})
        if self.wait() == 0:
            inference_result = self.get_output_result()
            cords = []
            for boxes in inference_result[0][0]:
                if boxes[2] > self.threshold:
                    output_cords = self.preprocess_output(boxes, image)
                    cords.append(output_cords)
            print(type(cords))
            if len(cords) == 0:
                return 0, 0
        return image[output_cords[1]:output_cords[3], output_cords[0]:output_cords[2]], cords

    def check_model(self):
        if "CPU" in self.device:
            supported_layers = self.plugin.query_network(self.network, "CPU")
            not_supported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for the specified device {}:\n {}".
                      format(self.device, ', '.join(not_supported_layers)))
            log.error(
                "Please try to specify cpu extensions library path in sample's command line parameters using -l "
                "or --cpu_extension command line argument")
            sys.exit(1)

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        n, c, h, w = self.input_shape
        pre_pro_frame =cv.resize(image, (w, h))
        pre_pro_frame = pre_pro_frame.transpose((2, 0, 1))
        pre_pro_frame = pre_pro_frame.reshape((n, c, h, w))
        return pre_pro_frame

    def preprocess_output(self, boxes, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        width, height = (image.shape[1], image.shape[0])
        return [int(width * boxes[3]), int(height * boxes[4]), int(width * boxes[5]), int(height * boxes[6])]

    def wait(self):
        inf_status = self.exec_network.requests[0].wait(-1)
        return inf_status

    def get_output_result(self):
        return self.exec_network.requests[0].outputs[self.output_blob]
