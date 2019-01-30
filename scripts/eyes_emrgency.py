#!/usr/bin/env python
# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import rospy
from std_msgs.msg import Int32

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import roslib
roslib.load_manifest("eyes_emergency")


def frame_grabber(image):
    # print("Got frame")
    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(image, "bgra8")
    except CvBridgeError as e:
      print(e)
    cv2.imshow("Frame", cv_image)
    key = cv2.waitKey(1) & 0xFF



def main():
    rospy.init_node("eyes_emergency")
    rospy.Subscriber("/d435/ColorImage",Image, frame_grabber)
    rospy.spin()


if __name__ == "__main__":
    main()