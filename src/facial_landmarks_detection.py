'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import sys
import logging as log
import cv2
from openvino.inference_engine import IENetwork, IECore


class Model_Facial_Landmarks:
    '''
    Class for the Face Landmarks Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.5):
        '''
        Setting Instance variables.
        '''
        self.model_name = model_name
        self.device = device
        self.extension = extensions
        self.threshold = threshold
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.input_shape = None

    def load_model(self):
        model_structure = self.model_name + ".xml"
        model_weight = self.model_name + ".bin"
        self.network = IENetwork(model=model_structure, weights=model_weight)
        self.plugin = IECore()

        if self.extension is not None and "CPU" in self.device:
            self.plugin.add_extension(self.extension, self.device)

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
            eyes_coordinates = self.preprocess_output(inference_result[0], image)
            left_eye = image[eyes_coordinates[0][2]:eyes_coordinates[0][3],
                       eyes_coordinates[0][0]:eyes_coordinates[0][1]]
            right_eye = image[eyes_coordinates[1][2]:eyes_coordinates[1][3],
                        eyes_coordinates[1][0]:eyes_coordinates[1][1]]
        return eyes_coordinates, left_eye, right_eye

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
        n, c, h, w = self.input_shape
        pre_pro_frame = cv2.resize(image, (w, h))
        pre_pro_frame = pre_pro_frame.transpose((2, 0, 1))
        pre_pro_frame = pre_pro_frame.reshape(n, c, h, w)
        return pre_pro_frame

    def preprocess_output(self, landmark, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        # outputs = landmark[self.output_name][0]
        left_eye_x_coordinate = int(landmark[0] * image.shape[1])
        left_eye_y_coordinate = int(landmark[1] * image.shape[0])
        right_eye_x_coordinate = int(landmark[2] * image.shape[1])
        right_eye_y_coordinate = int(landmark[3] * image.shape[0])
        return [[left_eye_x_coordinate - 15, left_eye_x_coordinate + 15, left_eye_y_coordinate - 15,
                 left_eye_y_coordinate + 15], [right_eye_x_coordinate - 15, right_eye_x_coordinate + 15,
                                               right_eye_y_coordinate - 15, right_eye_y_coordinate + 15]]


    def wait(self):
        inf_status = self.exec_network.requests[0].wait(-1)
        return inf_status

    def get_output_result(self):
        return self.exec_network.requests[0].outputs[self.output_blob]
