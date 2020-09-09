# The following code is used to watch a video stream, detect Aruco markers, and use
# a set of markers to determine the posture of the camera in relation to the plane
# of markers.
#
# Assumes that all markers are on the same plane, for example on the same piece of paper
#
# Requires camera calibration (see the rest of the project for example calibration)

import numpy
import cv2
import cv2.aruco as aruco
import os
import pickle
import sys

# Check for camera calibration data
if not os.path.exists('calibration.pckl'):
    print("You need to calibrate the camera you'll be using. See calibration project directory for details.")
    exit()
else:
    f = open('calibration.pckl', 'rb')
    (cameraMatrix, distCoeffs, _, _) = pickle.load(f)
    f.close()
    if cameraMatrix is None or distCoeffs is None:
        print("Calibration issue. Remove ./calibration.pckl and recalibrate your camera with CalibrateCamera.py.")
        exit()

# Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_APRILTAG_36h11)
ARUCO_PARAMETERS.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG

# Create grid board object we're using in our stream
board = aruco.GridBoard_create(
        markersX=3,
        markersY=3,
        markerLength=0.021,
        markerSeparation=0.00925,
        dictionary=ARUCO_DICT)

# Create vectors we'll be using for rotations and translations for postures
rvecs, tvecs = None, None

cam = cv2.VideoCapture(cv2.CAP_DSHOW)

codec = 0x47504A4D # MJPG
cam.set(cv2.CAP_PROP_FOURCC, codec)

cam.set(3, 3840)
cam.set(4, 2160)
cam.set(10, 12)
cam.set(21, 0)
cam.set(15, -6)
cam.set(39, 0)
cam.set(28, 50)
cam.set(5, 30)

while(cam.isOpened()):
    # Capturing each frame of our video stream
    ret, QueryImg = cam.read()
    if ret == True:
        # grayscale image
        gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)
        pyrdown = cv2.pyrDown(gray)
    
        # Detect Aruco markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(pyrdown, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
  
        # Refine detected markers
        # Eliminates markers not part of our board, adds missing markers to the board
        corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
                image = pyrdown,
                board = board,
                detectedCorners = corners,
                detectedIds = ids,
                rejectedCorners = rejectedImgPoints,
                cameraMatrix = cameraMatrix,
                distCoeffs = distCoeffs)   

        ###########################################################################
        # TODO: Add validation here to reject IDs/corners not part of a gridboard #
        ###########################################################################

        # Outline all of the markers detected in our image
        #QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0, 0, 255))

        # Require 15 markers before drawing axis
        if ids is not None and len(ids) ==3:
            # Estimate the posture of the gridboard, which is a construction of 3D space based on the 2D video 
            pose, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, cameraMatrix/2, distCoeffs)
            if pose:
                # Draw the camera posture calculated from the gridboard
                
                #boardCorrection = numpy.array([[0.048], [0.015], [0.0]])
                #tvec = tvec + boardCorrection
                
                QueryImg = aruco.drawAxis(QueryImg, cameraMatrix, distCoeffs, rvec, tvec, 0.1)
                print("Frame")
                print(tvec, rvec)
                orig_stdout = sys.stdout
                f = open('out.txt', 'a')
                sys.stdout = f
                print("Frame")
                print(tvec, rvec)
                sys.stdout = orig_stdout
                f.close()
        # Display our image
        cv2.namedWindow("QueryImage", cv2.WINDOW_NORMAL)
        cv2.imshow('QueryImage', QueryImg)

    # Exit at the end of the video on the 'q' keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()