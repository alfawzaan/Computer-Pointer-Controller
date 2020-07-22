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
    Class for the Face Detection Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.5):
        '''
        TODO: Use this to set your instance variables.
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
        print("FL In Load")
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
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        print("FL In Predict")
        pre_pro_img = self.preprocess_input(image)
        self.exec_network.start_async(request_id=0, inputs={self.input_blob: pre_pro_img})
        if self.wait() == 0:
            inference_result = self.get_output_result()
            eyes_coordinates = self.preprocess_output(inference_result[0], image)
            # cv2.rectangle(image, (eyes_coordinates[0][0], eyes_coordinates[0][2]),
            #               (eyes_coordinates[0][1], eyes_coordinates[0][3]),
            #               color=(256, 256, 0), thickness=1)
            # cv2.rectangle(image, (eyes_coordinates[1][0], eyes_coordinates[1][2]),
            #               (eyes_coordinates[1][1], eyes_coordinates[1][3]),
            #               color=(256, 256, 0), thickness=1)
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

        # left1_x = landmark[24]  # 0
        # left1_y = landmark[25]
        # left2_x = landmark[28]  # 1
        # left2_y = landmark[29]
        # right1_x = landmark[30]  # 2
        # right1_y = landmark[31]
        # right2_x = landmark[34]  # 3
        # right2_y = landmark[35]
        # left_eye_brow_x = landmark[26]
        # left_eye_brow_y = landmark[27]
        # right_eye_brow_y = landmark[32]
        # right_eye_brow_y = landmark[33]
        # height, width = (image.shape[0], image.shape[1])
        # return [[int(width * left1_x), int(width * left2_x), int(left_eye_brow_y * height),
        #          int(height * (left_eye_brow_y + (left2_x - left1_x)))], [
        #             int(width * right1_x), int(width * right2_x), int(right_eye_brow_y * height),
        #             int(height * (right_eye_brow_y + (right2_x - right1_x)))]]

    def wait(self):
        inf_status = self.exec_network.requests[0].wait(-1)
        return inf_status

    def get_output_result(self):
        return self.exec_network.requests[0].outputs[self.output_blob]
