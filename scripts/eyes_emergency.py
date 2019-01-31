#!/usr/bin/env python
# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
import sys
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
from std_msgs.msg import Bool

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import roslib
roslib.load_manifest("eyes_emergency")

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.28
EYE_AR_CONSEC_FRAMES = 3
EYE_COUNTER_THRESH = 20

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
FRAME = 0

def eye_aspect_ratio(eye):
	# compute the euclivs.read()dean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
    return ear



def frame_grabber(image, args):
    global FRAME
    FRAME += 1
    detector = args[0]
    predictor = args[1]
    emergency_pub = args[2]
    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    bridge = CvBridge()
    try:
        frame = bridge.imgmsg_to_cv2(image, "bgra8")
    except CvBridgeError as e:
        print(e)

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    # For every face in the frame
    for rect in rects:
        # Find the eyes and calculate the EARs
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Mark the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        global COUNTER
        if leftEAR <= EYE_AR_THRESH or rightEAR <= EYE_AR_THRESH:
            COUNTER += 1
        else:
            COUNTER = 0
        
        # Write info on frame
        cv2.putText(frame, "L_EAR: {:.2f}".format(leftEAR), (300, 60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Left Counter: {}".format(COUNTER), (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	
    if COUNTER > 10:
        emergency_pub.publish(True)
        cv2.putText(frame, "LEFT EAR FALLBACK!!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		# print("LEFT EAR FALLBACK!!! @ ", FRAME)
    else:
        emergency_pub.publish(False)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

def main():
    rospy.init_node("eyes_emergency")
    rospy.sleep(1)
    
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    predictor_file = sys.argv[1]
    rospy.loginfo("Loading predictor from: {}".format(predictor_file))

    # Get detector and face predictor from the supplied dat file
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_file)

    global EYE_AR_THRESH, EYE_COUNTER_THRESH
    EYE_AR_THRESH = float(sys.argv[2])
    EYE_COUNTER_THRESH= float(sys.argv[3])

    rospy.loginfo("Setting EAR thresh to {} and counter thresh to {}".format(EYE_AR_THRESH, EYE_COUNTER_THRESH))

    emergency_pub = rospy.Publisher("eyes_emergency", Bool, queue_size=10)

    rospy.Subscriber("/d435/ColorImage",Image,frame_grabber,
    (detector, predictor, emergency_pub))
    rospy.spin()


if __name__ == "__main__":
    main()