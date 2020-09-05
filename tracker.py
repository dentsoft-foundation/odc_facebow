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
ARUCO_PARAMETERS.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG

# Create vectors we'll be using for rotations and translations for postures
rvecs, tvecs = None, None

cap = cv2.VideoCapture(cv2.CAP_DSHOW)
# codec = 0x47504A4D # MJPG
# codec = 844715353.0 # YUY2

codec = 0x47504A4D # MJPG
cap.set(cv2.CAP_PROP_FOURCC, codec)

cap.set(3, 3840)
cap.set(4, 2160)
cap.set(10, 12)
cap.set(21, 0)
cap.set(15, -6)
cap.set(39, 0)
cap.set(28, 60)
cap.set(5, 30)

board = aruco.GridBoard_create(
        markersX=3,
        markersY=1,
        markerLength=0.021,
        markerSeparation=0.0093,
        dictionary=ARUCO_DICT)

while True:
    success, img = cap.read()
    imgGrey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #imgBlur = cv2.GaussianBlur(imgGrey,(11,11),cv2.BORDER_DEFAULT)
    imgPyrDown = cv2.pyrDown(imgGrey)

    # lists of ids and the corners beloning to each id
    corners, ids, rejected_img_points = aruco.detectMarkers(imgPyrDown, ARUCO_DICT, parameters=ARUCO_PARAMETERS, cameraMatrix=cameraMatrix, distCoeff=distCoeffs)
    for one in rejected_img_points:
        more_corners, more_ids, rej, recovered_ids = cv2.aruco.refineDetectedMarkers(imgPyrDown, board, corners, ids, one)
     
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
            rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i]*2, 0.021, cameraMatrix, distCoeffs) 
            (rvec - tvec).any()  # get rid of that nasty numpy value array error
                        
            print(ids[i], tvec)
            
            orig_stdout = sys.stdout
            f = open('out.txt', 'a')
            sys.stdout = f
            print(ids[i], tvec)
            sys.stdout = orig_stdout
            f.close()

            #cv2.aruco.drawDetectedMarkers(img,corners,ids)
            aruco.drawAxis(img, cameraMatrix, distCoeffs, rvec, tvec, 0.02)  # Draw Axis
        
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    #cv2.namedWindow("thresh1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("PyrDown", cv2.WINDOW_NORMAL)

    cv2.imshow("img", img)
    #cv2.imshow("thresh1", thresh1)
    cv2.imshow("PyrDown", imgPyrDown)

    if cv2.waitKey(1) & 0xFF ==ord('q'):
     break
cv2.destroyAllWindows()
