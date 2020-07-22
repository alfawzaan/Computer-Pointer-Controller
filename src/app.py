from argparse import ArgumentParser

import imutils as imutils

from face_detection import Model_Face_Detection
from facial_landmarks_detection import Model_Facial_Landmarks
from gaze_estimation import Model_Gaze_Estimation
from head_pose_estimation import Model_Head_Pose_Estimation
from input_feeder import InputFeeder
from mouse_controller import MouseController
import cv2
from logging import log
import sys
import numpy as np


def get_argument_parser():
    parser = ArgumentParser()
    parser.add_argument("-fd", "--model_path_fd", required=True,
                        help="path to the Face Detection Model IR xml files")
    parser.add_argument("-hp", "--model_path_hp", required=True,
                        help="path to the Head Pose Estimation Model IR xml files")
    parser.add_argument("-fl", "--model_path_fl", required=True,
                        help="path to the Facial Landmarks Model IR xml files")
    parser.add_argument("-ge", "--model_path_ge", required=True,
                        help="path to the Gaze Estimation Model IR xml files")
    parser.add_argument("-i", "--input_type", required=True,
                        help="Specify image, video or cam to use your WebCam")
    parser.add_argument("-f", "--input_file", default=None,
                        help="Path to the image or video file. Leave empty if you intend to use a WebCam")
    parser.add_argument("-l", "--cpu_extension", required=False, default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", default="CPU",
                        help="Indicate the type of device you would like to run"
                             " the inference on. eg. CPU, GPU, FPGA, MYRIAD. If none of "
                             "these is specified, CPU will be used")
    parser.add_argument("-pt", "--threshold", type=float, default=0.5,
                        help="Probability threshold for face detection"
                             "(0.5 by default)")
    parser.add_argument("-c", "--toggle", default=[],
                        help="Customize what should be displayed on the screen. [stats,frame]")
    return parser


def run_app(args):
    face_detection_model = Model_Face_Detection(args.model_path_fd, args.device, args.cpu_extension,
                                                threshold=args.threshold)
    face_detection_model.load_model()
    head_pose_model = Model_Head_Pose_Estimation(args.model_path_hp, args.device, args.cpu_extension)
    head_pose_model.load_model()
    face_landmark_model = Model_Facial_Landmarks(args.model_path_fl, args.device, args.cpu_extension)
    face_landmark_model.load_model()
    gaze_model = Model_Gaze_Estimation(args.model_path_ge, args.device, args.cpu_extension)
    gaze_model.load_model()

    input_feeder = InputFeeder(args.input_type, args.input_file, )
    input_feeder.load_data()
    mouse_controller = MouseController("medium", "fast")
    # while input_feeder.cap.isOpened():
    # feed_out=input_feeder.next_batch()

    frame_count = 0
    custom = args.toggle

    for frame in input_feeder.next_batch():

        if frame is None:
            break
        key_pressed = cv2.waitKey(60)
        frame_count += 1
        face_out, cords = face_detection_model.predict(frame.copy())

        # When no face was detected
        if cords == 0:
            inf_info = "No Face Detected in the Frame"
            write_text_img(frame, inf_info, 400)
            continue

        eyes_cords, left_eye, right_eye = face_landmark_model.predict(face_out.copy())
        head_pose_out = head_pose_model.predict(face_out.copy())
        gaze_out = gaze_model.predict(left_eye, right_eye, head_pose_out)

        # Faliure in processing both eyes
        if gaze_out is None:
            continue
        x, y = gaze_out
        if frame_count % 5 == 0:
            mouse_controller.move(x, y)
        inf_info = "Head Pose (y: {:.2f}, p: {:.2f}, r: {:.2f})".format(head_pose_out[0],
                                                                        head_pose_out[1],
                                                                        head_pose_out[2])
        # Process Visualization
        if 'frame' in custom:
            visualization(frame, cords, face_out, eyes_cords)

        if 'stats' in custom:
            print("Statistics")
            write_text_img(face_out, inf_info, 400)
            inf_info = "Gaze Angle: x: {:.2f}, y: {:.2f}".format(x, y)
            write_text_img(face_out, inf_info, 400, 15)

        out_f = np.hstack((cv2.resize(frame, (400, 400)), cv2.resize(face_out, (400, 400))))
        cv2.imshow('Visualization', out_f)
        if key_pressed == 27:
            break
    input_feeder.close()
    cv2.destroyAllWindows()


def write_text_img(image, text, frame_size, position=None):
    y = 15
    if position is not None:
        y += position
    cv2.putText(image, text, (15, y),
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (200, 10, 10), 1)
    return imutils.resize(image, height=frame_size, width=frame_size)


def visualization(frame, face_cords, image, eyes_coords):
    for cord in face_cords:
        cv2.rectangle(frame, (cord[0], cord[1]), (cord[2], cord[3]),
                      color=(256, 256, 256), thickness=1)
    cv2.rectangle(image, (eyes_coords[0][0], eyes_coords[0][2]),
                  (eyes_coords[0][1], eyes_coords[0][3]),
                  color=(256, 256, 0), thickness=1)
    cv2.rectangle(image, (eyes_coords[1][0], eyes_coords[1][2]),
                  (eyes_coords[1][1], eyes_coords[1][3]),
                  color=(256, 256, 0), thickness=1)


def main():
    print("Main Run App")
    args = get_argument_parser().parse_args()
    run_app(args)


if __name__ == '__main__':
    main()
