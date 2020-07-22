'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import sys
import logging as log
import cv2
from openvino.inference_engine import IENetwork, IECore


class Model_Head_Pose_Estimation:
    '''
    Class for the Face Detection Model.
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
        '''
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
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
        inference_result=self.exec_network.infer(inputs={self.input_blob: pre_pro_img})
        # if self.wait() == 0:
        # inference_result = self.get_output_result()
        type(inference_result)
        head_pose = self.preprocess_output(inference_result)
        return head_pose

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

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        return [outputs["angle_y_fc"].tolist()[0][0], outputs["angle_p_fc"].tolist()[0][0], outputs["angle_r_fc"].tolist()[0][0]]

    def wait(self):
        inf_status = self.exec_network.requests[0].wait(-1)
        return inf_status

    def get_output_result(self):
        return self.exec_network.requests[0].outputs[self.output_blob]
