import cv2
import numpy as np
import cv2.aruco as aruco
import pickle
import os
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
#ARUCO_PARAMETERS.adaptiveThreshWinSizeMin = 3;
#ARUCO_PARAMETERS.adaptiveThreshWinSizeMax = 23;
#ARUCO_PARAMETERS.adaptiveThreshWinSizeStep = 10;
#ARUCO_PARAMETERS.adaptiveThreshConstant = 7;
#ARUCO_PARAMETERS.cornerRefinementWinSize = 5;
#ARUCO_PARAMETERS.cornerRefinementMaxIterations = 30;
#ARUCO_PARAMETERS.cornerRefinementMinAccuracy = 0.1;
ARUCO_PARAMETERS.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
ARUCO_PARAMETERS.aprilTagDeglitch = 0
ARUCO_PARAMETERS.polygonalApproxAccuracyRate = 0.05
ARUCO_PARAMETERS.maxMarkerPerimeterRate = 4
ARUCO_PARAMETERS.aprilTagMinWhiteBlackDiff = 30
ARUCO_PARAMETERS.errorCorrectionRate = 1.0
ARUCO_PARAMETERS.aprilTagMaxLineFitMse = 20
ARUCO_PARAMETERS.aprilTagCriticalRad = 0.1745329201221466 *6
ARUCO_PARAMETERS.aprilTagMaxNmaxima = 20
ARUCO_PARAMETERS.aprilTagQuadDecimate = 1.5 # THIS PARAMETER speeds up tracking



# Create vectors we'll be using for rotations and translations for postures
rvecs, tvecs = None, None

cap = cv2.VideoCapture(cv2.CAP_DSHOW)
# codec = 0x47504A4D # MJPG
# codec = 844715353.0 # YUY2

codec = 844715353.0
cap.set(cv2.CAP_PROP_FOURCC, codec)

cap.set(3, 3840)
cap.set(4, 2160)
cap.set(10, 32)
cap.set(21, 0)
cap.set(15, -6)
cap.set(39, 0)
cap.set(28, 20)
cap.set(5, 30)


while True:
    success, img = cap.read()
    imgGrey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
    # lists of ids and the corners beloning to each id
    corners, ids, rejected_img_points = aruco.detectMarkers(imgGrey, ARUCO_DICT, parameters=ARUCO_PARAMETERS, cameraMatrix=cameraMatrix, distCoeff=distCoeffs)
# Outline all of the markers detected in our image
    
    if np.all(ids is not None):  # If there are markers found by detector
    
        print('frame')
        orig_stdout = sys.stdout
        f = open('out.txt', 'a')
        sys.stdout = f
        print('frame')
        sys.stdout = orig_stdout
        f.close()
        for i in range(0, len(ids)):  # Iterate in markers
            # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
            rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.021, cameraMatrix, distCoeffs)
            (rvec - tvec).any()  # get rid of that nasty numpy value array error
            
            print(ids[i], tvec)    
            
            orig_stdout = sys.stdout
            f = open('out.txt', 'a')
            sys.stdout = f
            print(ids[i], tvec)
            sys.stdout = orig_stdout
            f.close()

            cv2.aruco.drawDetectedMarkers(img,corners,ids,)
            aruco.drawAxis(img, cameraMatrix, distCoeffs, rvec, tvec, 0.02)  # Draw Axis

    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
     break
