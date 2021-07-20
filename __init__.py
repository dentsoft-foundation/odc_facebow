

import os, sys, shutil, threading, queue
import pickle, glob
import numpy as np
from mathutils import *
import math
import bpy, bmesh
from bpy_extras.io_utils import ImportHelper

import pip

import time

def pymod_install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])

try:
    import cv2
    from cv2 import aruco
    print(cv2.__version__)
except ModuleNotFoundError as e:
    print("OpenCV not present, attempting install via pip.")
    pymod_install('opencv-contrib-python')
    try:
        import cv2
        from cv2 import aruco
    except ModuleNotFoundError as e: print("Failed to pip install dependencies!")

class aruco_tracker():
    def __init__(self, context, data_source=cv2.CAP_DSHOW, debug=False, process_in_main=False):
        if context.scene.legacy_file is False:
            if process_in_main is False:
                self.processor_thread = threading.Thread(target=self.stream_processor, args=(context, data_source, debug))
                self.processor_thread.start()
            self.queue = queue.Queue()
        else:
            self.stream_processor(context, data_source, debug, 'video')


    def update_tracking_marker(self, context, marker_obj, tvec, rvec, current_frame):
        # from https://www.learnopencv.com/rotation-matrix-to-euler-angles/
        # Checks if a matrix is a valid rotation matrix.
        def isRotationMatrix(R) :
            Rt = np.transpose(R)
            shouldBeIdentity = np.dot(Rt, R)
            I = np.identity(3, dtype = R.dtype)
            n = np.linalg.norm(I - shouldBeIdentity)
            return n < 1e-6
            
        # Calculates rotation matrix to euler angles
        # The result is the same as MATLAB except the order
        # of the euler angles ( x and z are swapped ).
        def rotationMatrixToEulerAngles(R) :
            assert(isRotationMatrix(R))
            sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
            singular = sy < 1e-6
            if  not singular :
                x = math.atan2(R[2,1] , R[2,2])
                y = math.atan2(-R[2,0], sy)
                z = math.atan2(R[1,0], R[0,0])
            else :
                x = math.atan2(-R[1,2], R[1,1])
                y = math.atan2(-R[2,0], sy)
                z = 0
            return np.array([x, y, z])

        rvec = np.array(rvec)
        r_matrix, _ = cv2.Rodrigues(rvec) #converts rotation vector to rotation matrix via Rodrigues formula
        #convert rotation matrix to euler angles
        r_euler = rotationMatrixToEulerAngles(r_matrix)
        '''final = Matrix(
            [[0.01, 0.0, 0.0, float(tvec[0][0][0])],
            [0.0, 0.01, 0.0, float(tvec[0][0][1])],
            [0.0, 0.0, 0.01, float(tvec[0][0][2])],
            [0.0, 0.0, 0.0, 1.0]]
        )'''
        #https://docs.blender.org/api/current/mathutils.html
        # https://docs.blender.org/api/current/mathutils.html#mathutils.Matrix
        # create an identitiy matrix
        mat_sca = Matrix.Scale(0.001, 4)
        print(tvec)
        
        mat_trans = Matrix.Translation((float(tvec[0][0]), float(tvec[1][0]), float(tvec[2][0]))) #structure on single markers -> ((float(tvec[0][0][0]), float(tvec[0][0][1]), float(tvec[0][0][2]))) #structure when doing POSE [[-0.20874319], [-0.13054966], [ 0.98599982]] - >((float(tvec[0][0]), float(tvec[1][0]), float(tvec[2][0])))
        # https://devtalk.blender.org/t/understanding-matrix-operations-in-blender/10148
        mat_rot_x = Matrix.Rotation(r_euler[0], 4, 'X')
        mat_rot_y = Matrix.Rotation(r_euler[1], 4, 'Y')
        mat_rot_z = Matrix.Rotation(r_euler[2], 4, 'Z')
        marker_obj.matrix_world = mat_trans @ mat_rot_x @ mat_rot_y @ mat_rot_z @ mat_sca
        dg = context.evaluated_depsgraph_get()
        dg.update()
        if current_frame is not None:
            marker_obj.keyframe_insert("location", frame = current_frame)
            marker_obj.keyframe_insert("rotation_euler", frame = current_frame)

    def update_marker_tracing(self, context, marker_obj, tvec, rvec, current_frame, tracing_name = "_trace"):
        #print(markerID)
        if bpy.data.objects.get(marker_obj.name +tracing_name) is None:
            mesh = bpy.data.meshes.new(marker_obj.name +tracing_name+"_data")  # add a new mesh
            obj = bpy.data.objects.new(marker_obj.name +tracing_name, mesh)  # add a new object using the mesh
            bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0))
            bpy.context.view_layer.objects.active.name = marker_obj.name + tracing_name + "_tracker"
            keyframe_tracking = bpy.context.view_layer.objects.active
            bpy.context.scene.collection.objects.link(obj)  # put the object into the scene (link)
            bpy.context.scene.collection.objects.link(keyframe_tracking)
            sg = bpy.data.collections.new(marker_obj.name +tracing_name)
            bpy.context.scene.collection.children.link(sg)
            sg.objects.link(obj)
            sg.objects.link(keyframe_tracking)
        else:
            obj = bpy.data.objects.get(marker_obj.name +tracing_name)
            keyframe_tracking = bpy.data.objects.get(marker_obj.name + tracing_name+"_tracker")
        
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        #print(tvec, type(tvec))
        if type(tvec) is not type(Vector()): bm.verts.new((float(tvec[0][0]), float(tvec[1][0]), float(tvec[2][0]))) #single marker -> ((float(tvec[0][0][0]), float(tvec[0][0][1]), float(tvec[0][0][2])))
        else: bm.verts.new(tvec)
        bm.verts.ensure_lookup_table()
        if len(bm.verts) > 1: bm.edges.new((bm.verts[-2], bm.verts[-1]))
        bm.to_mesh(obj.data)
        bm.free()

        if type(tvec) is not type(Vector()): keyframe_tracking.location = (float(tvec[0][0]), float(tvec[1][0]), float(tvec[2][0])) #single marker -> (float(tvec[0][0][0]), float(tvec[0][0][1]), float(tvec[0][0][2]))
        else: keyframe_tracking.location = tvec
        
        dg = context.evaluated_depsgraph_get()
        dg.update()
        #bpy.context.view_layer.update()

        if current_frame is not None:
            keyframe_tracking.keyframe_insert("location", frame = current_frame)

    def update_point_plane(self, context, point_vecs, plane_obj):
        plane_scale = plane_obj.scale.copy()
        plane_loc = 1/3 * (point_vecs[0] + point_vecs[1] + point_vecs[2])
        x = (point_vecs[1] - point_vecs[0]).normalized()
        y = (point_vecs[2] - point_vecs[0]).normalized()
        z = x.cross(y)
        y = z.cross(x)
        mat3x3 = Matrix().to_3x3()
        mat3x3.col[0] = x
        mat3x3.col[1] = y
        mat3x3.col[2] = z
        mat = mat3x3.to_4x4()
        plane_obj.matrix_world = mat
        plane_obj.location = plane_loc
        plane_obj.scale = plane_scale

    def update_vector_mag(self, context, point1, point2):
        vec = point2 - point1
        magnitude = vec.length * 1000.0 #convert from meters to mm
        return magnitude, vec


    def stream_processor(self, context, data_source, debug=False, source='video'):
        # Constant parameters used in Aruco methods
        ARUCO_PARAMETERS = aruco.DetectorParameters_create()
        ARUCO_DICT = context.scene.aruco_dict
        ARUCO_PARAMETERS.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
        ARUCO_PARAMETERS.aprilTagDeglitch = 0           
        ARUCO_PARAMETERS.aprilTagMinWhiteBlackDiff = 30
        ARUCO_PARAMETERS.aprilTagMaxLineFitMse = 20
        ARUCO_PARAMETERS.aprilTagCriticalRad = 0.1745329201221466 *6
        ARUCO_PARAMETERS.aprilTagMinClusterPixels = 5  
        ARUCO_PARAMETERS.maxErroneousBitsInBorderRate = 0.35
        ARUCO_PARAMETERS.errorCorrectionRate = 1.0                    
        ARUCO_PARAMETERS.minMarkerPerimeterRate = 0.05                  
        ARUCO_PARAMETERS.maxMarkerPerimeterRate = 4                  
        ARUCO_PARAMETERS.polygonalApproxAccuracyRate = 0.05
        ARUCO_PARAMETERS.minCornerDistanceRate = 0.05

        # Create vectors we'll be using for rotations and translations for postures
        rvecs, tvecs = None, None
        if source == 'video':
            cap = cv2.VideoCapture(data_source)
            #cam = cv2.VideoCapture("1.mp4")
            # codec = 0x47504A4D # MJPG
            # codec = 844715353.0 # YUY2

            codec = 0x47504A4D # MJPG
            cap.set(cv2.CAP_PROP_FOURCC, codec)

            if data_source == cv2.CAP_DSHOW:
                cap.set(3, context.scene.FRAME_WIDTH) #3. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
                cap.set(4, context.scene.FRAME_HEIGHT) #4. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
                cap.set(10, context.scene.CAMERA_BRIGHTNESS) #10. CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
                cap.set(5, context.scene.VIDEO_FPS) #5. CV_CAP_PROP_FPS Frame rate.
                if context.scene.CAMERA_AUTO_EXPOSURE == False:
                    cap.set(21, 0) #PROP_AUTO_EXPOSURE =21; 0 = OFF/FALSE
                    cap.set(15, context.scene.CAMERA_EXPOSURE_VAL) #CV_CAP_PROP_EXPOSURE
                elif context.scene.CAMERA_AUTO_EXPOSURE == True:
                    cap.set(21, 1)
                if context.scene.CAMERA_AUTO_FOCUS == False:
                    cap.set(39, 0) #CAP_PROP_AUTOFOCUS =39
                    cap.set(28, context.scene.CAMERA_FOCUS_VAL) #CAP_PROP_FOCUS =28
                elif context.scene.CAMERA_AUTO_FOCUS == True:
                    cap.set(39, 1)
            
            '''
            board = aruco.GridBoard_create(
                    markersX=3,
                    markersY=1,
                    markerLength=context.scene.cal_board_markerLength,
                    markerSeparation=0.0093,
                    dictionary=ARUCO_DICT)
            '''
        if context.scene.legacy_file is True:
            f = open(os.path.join(context.scene.legacy_dir, 'data_stream.txt'), "w")

        if  context.scene.use_pose_board is True:
            # create the opencv pose board object by indicating which IDs are grouped together
            anterior_ids = np.array([[0], [1], [2]], dtype=np.int32)
            anterior_board = aruco.Board_create(context.scene.pose_board_local_co, ARUCO_DICT, anterior_ids)
            left_ids = np.array([[3], [4], [5]], dtype=np.int32)
            left_board = aruco.Board_create(context.scene.pose_board_local_co, ARUCO_DICT, left_ids)
            right_ids = np.array([[6], [7], [8]], dtype=np.int32)
            right_board = aruco.Board_create(context.scene.pose_board_local_co, ARUCO_DICT, right_ids)
            MarkersIdCornersDict = dict()

        while True:
            current_frame = None
            if source == 'video':
                success, img = cap.read()
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if source == 'img':
                current_frame = 0
                total_frames = 0
                img = cv2.imread(data_source)
            imgGrey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #imgBlur = cv2.GaussianBlur(imgGrey, (7, 7), cv2.BORDER_DEFAULT)
            #imgPyrDown = cv2.pyrDown(imgGrey)

            # lists of ids and the corners beloning to each id
            corners, ids, rejected_img_points = aruco.detectMarkers(imgGrey, ARUCO_DICT, parameters=ARUCO_PARAMETERS, cameraMatrix=context.scene.cameraMatrix, distCoeff=context.scene.distCoeffs)
            cv2.aruco.drawDetectedMarkers(imgGrey,corners,ids)
            #for one in rejected_img_points:
            #    more_corners, more_ids, rej, recovered_ids = cv2.aruco.refineDetectedMarkers(imgPyrDown, board, corners, ids, one)
            
            # Outline all of the markers detected in our image
            
            if np.all(ids is not None):  # If there are markers found by detector
                if  context.scene.use_pose_board is True:
                    if ids is not None and len(ids) >= 9:
                        for i in range(0, len(ids)):  # Iterate in markers
                            MarkersIdCornersDict[ids[i][0]] = (list(corners))[i]

                        anterior_corners = [
                            MarkersIdCornersDict[0],
                            MarkersIdCornersDict[1],
                            MarkersIdCornersDict[2],
                        ]
                        left_corners = [
                            MarkersIdCornersDict[3],
                            MarkersIdCornersDict[4],
                            MarkersIdCornersDict[5],
                        ]
                        right_corners = [
                            MarkersIdCornersDict[6],
                            MarkersIdCornersDict[7],
                            MarkersIdCornersDict[8],
                        ]

                        # Estimate the posture of the board, which is a construction of 3D space based on the 2D video
                        ant_retval, ant_rvec, ant_tvec = cv2.aruco.estimatePoseBoard(
                            anterior_corners,
                            anterior_ids,
                            anterior_board,
                            context.scene.cameraMatrix,
                            context.scene.distCoeffs,
                            None,
                            None,
                        )
                        if ant_retval:
                            self.queue.put([0, ant_tvec, ant_rvec, current_frame])
                            imgGrey = aruco.drawAxis(imgGrey, context.scene.cameraMatrix, context.scene.distCoeffs, ant_rvec, ant_tvec, 0.05)

                        L_retval, L_rvec, L_tvec = cv2.aruco.estimatePoseBoard(
                            left_corners,
                            left_ids,
                            left_board,
                            context.scene.cameraMatrix,
                            context.scene.distCoeffs,
                            None,
                            None,
                        )
                        if L_retval:
                            self.queue.put([1, L_tvec, L_rvec, current_frame])
                            imgGrey = aruco.drawAxis(imgGrey, context.scene.cameraMatrix, context.scene.distCoeffs, L_rvec, L_tvec, 0.05)

                        R_retval, R_rvec, R_tvec = cv2.aruco.estimatePoseBoard(
                            right_corners,
                            right_ids,
                            right_board,
                            context.scene.cameraMatrix,
                            context.scene.distCoeffs,
                            None,
                            None,
                        )
                        if R_retval:
                            self.queue.put([2, R_tvec, R_rvec, current_frame])
                            imgGrey = aruco.drawAxis(imgGrey, context.scene.cameraMatrix, context.scene.distCoeffs, R_rvec, R_tvec, 0.05)
                        imgGrey = cv2.aruco.drawDetectedMarkers(imgGrey, corners, ids, (0,255,0))    
                        
                else:
                    for i in range(0, len(ids)):  # Iterate in markers
                        # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                        rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], context.scene.cal_board_markerLength, context.scene.cameraMatrix, context.scene.distCoeffs) 
                        (rvec - tvec).any()  # get rid of that nasty numpy value array error
                                    
                        #print(ids[i], tvec[0][0])
                        #print(ids)
                        
                        cv2.aruco.drawDetectedMarkers(imgGrey,corners,ids)
                        aruco.drawAxis(imgGrey, context.scene.cameraMatrix, context.scene.distCoeffs, rvec, tvec, 0.05)  # Draw Axis
                        if context.scene.legacy_file is False: self.queue.put([ids[i][0], tvec, rvec, current_frame])
                        elif context.scene.legacy_file is True:
                            f.write(str([ids[i][0], tvec.tolist(), rvec.tolist(), current_frame])+"\n")
                
            if debug == True:
                cv2.namedWindow("img", cv2.WINDOW_NORMAL)
                #cv2.namedWindow("PyrDown", cv2.WINDOW_NORMAL)

                cv2.imshow("img", imgGrey)
                #cv2.imshow("img", img)

                #cv2.imshow("PyrDown", imgPyrDown)
                if cv2.waitKey(1) & 0xFF ==ord('q'):
                    break
            if current_frame == total_frames:
                cv2.destroyAllWindows()
                break
            if source == 'img': break
            
        if debug == True: cv2.destroyAllWindows()
        if context.scene.legacy_file is True: f.close()
        #self.processor_thread.join()

bl_info = {
    "name": "Open Dental CAD Digital Facebow",
    "author": "Georgi Talmazov, Ilya Fomenko, Patrick R. Moore",
    "version": (0, 1),
    "blender": (2, 83, 0),
    "location": "3D View -> UI SIDE PANEL",
    "description": "Blender add-on utilizing OpenCV aruco markers to capture reference points used in obtaining dental recrods (ie. facebow).",
    "warning": "",
    "wiki_url": "",
    "category": "Dental",
    }

class generate_tracking_marker(bpy.types.Operator):
    bl_idname = "facebow.generate_aruco_marker"
    bl_label = "Generate Tracking Marker"

    def execute(self, context):
        folder = "C:/Users/Mayotic/AppData/Roaming/Blender Foundation/Blender/2.83/scripts/addons/odc_facebow"
        if "markers" not in os.listdir(folder):
            try: os.mkdir(folder+"/markers")
            except FileExistsError as e: print("Marker dir exists!")
        folder += "/markers"
        print(folder)

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        # Create an image from the marker
        # second param is ID number
        # last param is total image size
        for i in range(0, context.scene.manual_marker_num):
            img = aruco.drawMarker(context.scene.aruco_dict, i, context.scene.manual_marker_res)
            cv2.imwrite(os.path.join(folder, str(i)+".jpg"), img)
        return {'FINISHED'}

class generate_aruco_marker_board(bpy.types.Operator):
    bl_idname = "facebow.generate_aruco_appliance_board"
    bl_label = "Generate Appliances' Markers"

    def execute(self, context):
        folder = "C:/Users/talmazovg/AppData/Roaming/Blender Foundation/Blender/2.83/scripts/addons/odc_facebow"
        if "markers" not in os.listdir(folder):
            try: os.mkdir(folder+"/markers")
            except FileExistsError as e: print("Marker dir exists!")
        folder += "/markers"
        print(folder)

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        # Create an image from the marker
        # second param is ID number
        # last param is total image size
        gridboard = aruco.GridBoard_create(
                markersX=3, 
                markersY=3, 
                markerLength=0.04, 
                markerSeparation=0.01, 
                dictionary=context.scene.aruco_dict)

        # Create an image from the gridboard
        img = gridboard.draw(outSize=(context.scene.cal_board_X_res,context.scene.cal_board_Y_res))
        cv2.imwrite(os.path.join(folder,"test_gridboard.jpg"), img)
        return {'FINISHED'}

class generate_calibration_board(bpy.types.Operator):
    bl_idname = "facebow.generate_aruco_board"
    bl_label = "Generate Calibration Board"

    def execute(self, context):
        folder = "C:/Users/Mayotic/AppData/Roaming/Blender Foundation/Blender/2.83/scripts/addons/odc_facebow"
        print(folder)
        #aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000 )
        board = aruco.CharucoBoard_create(
                squaresX=context.scene.cal_board_X_num,
                squaresY=context.scene.cal_board_Y_num,
                squareLength=context.scene.cal_board_squareLength,
                markerLength=context.scene.cal_board_markerLength,
                dictionary=context.scene.aruco_dict)
        print("generated calibration board")
        img = board.draw((context.scene.cal_board_X_res,context.scene.cal_board_Y_res))
        cv2.imwrite(os.path.join(folder,"calibration_board.jpg"), img)
        #cv2.imshow("aruco", img)
        """
        newImage = bpy.data.images.new("aruco_board", 864, 1080, alpha=True, float_buffer=True)
        img = img.astype(float).flatten()
        img = img*1/255
        print(newImage.pixels[0],newImage.pixels[1],newImage.pixels[2],newImage.pixels[3],newImage.pixels[4],newImage.pixels[5],newImage.pixels[6],newImage.pixels[7])
        print(img[0],img[1],img[2],img[3],img[4], img[5], img[6], img[7])
        print(newImage.file_format)

        #newImage.pixels = img
        #newImage.update()
        """
        return {'FINISHED'}

class calibrate(bpy.types.Operator, ImportHelper):
    bl_idname = "facebow.calibrate"
    bl_label = "Select Folder"

    filter_glob: bpy.props.StringProperty(default='*.*', options={'HIDDEN'})

    files = bpy.props.CollectionProperty(
            name="File Path",
            type=bpy.types.OperatorFileListElement,
            )
    directory = bpy.props.StringProperty(
            subtype='DIR_PATH',
            )


    def execute(self, context):
        """Do something with the selected file(s)."""

        # Create constants to be passed into OpenCV and Aruco methods
        CHARUCO_BOARD = aruco.CharucoBoard_create(
                squaresX=context.scene.cal_board_X_num,
                squaresY=context.scene.cal_board_Y_num,
                squareLength=context.scene.cal_board_squareLength,
                markerLength=context.scene.cal_board_markerLength,
                dictionary=context.scene.aruco_dict)
        
        # Create the arrays and variables we'll use to store info like corners and IDs from images processed
        corners_all = [] # Corners discovered in all images processed
        ids_all = [] # Aruco ids corresponding to corners discovered
        image_size = None # Determined at runtime

        directory = self.directory
        for file_elem in self.files:
            filepath = os.path.join(directory, file_elem.name)
            print(filepath)

        images = glob.glob(self.directory+'*.jpg')

        # Loop through images glob'ed
        for iname in images:
            # Open the image
            img = cv2.imread(iname)
            # Grayscale the image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find aruco markers in the query image
            corners, ids, _ = aruco.detectMarkers(
                    image=gray,
                    dictionary=context.scene.aruco_dict)

            # Outline the aruco markers found in our query image
            img = aruco.drawDetectedMarkers(
                    image=img, 
                    corners=corners)

            # Get charuco corners and ids from detected aruco markers
            response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                    markerCorners=corners,
                    markerIds=ids,
                    image=gray,
                    board=CHARUCO_BOARD)

            # If a Charuco board was found, let's collect image/corner points
            # Requiring at least 20 squares
            if response > 10:
                # Add these corners and ids to our calibration arrays
                corners_all.append(charuco_corners)
                ids_all.append(charuco_ids)
                
                # Draw the Charuco board we've detected to show our calibrator the board was properly detected
                img = aruco.drawDetectedCornersCharuco(
                        image=img,
                        charucoCorners=charuco_corners,
                        charucoIds=charuco_ids)
            
                # If our image size is unknown, set it now
                if not image_size:
                    image_size = gray.shape[::-1]
            
                # Reproportion the image, maxing width or height at 1000
                proportion = max(img.shape) / 1000.0
                img = cv2.resize(img, (int(img.shape[1]/proportion), int(img.shape[0]/proportion)))
                # Pause to display each image, waiting for key press
                cv2.imshow('Charuco board', img)
                cv2.waitKey(0)
            else:
                print("Not able to detect a charuco board in image: {}".format(iname))

        # Destroy any open CV windows
        cv2.destroyAllWindows()

        # Make sure at least one image was found
        if len(images) < 1:
            # Calibration failed because there were no images, warn the user
            print("Calibration was unsuccessful. No images of charucoboards were found. Add images of charucoboards and use or alter the naming conventions used in this file.")

        # Make sure we were able to calibrate on at least one charucoboard by checking
        # if we ever determined the image size
        if not image_size:
            # Calibration failed because we didn't see any charucoboards of the PatternSize used
            print("Calibration was unsuccessful. We couldn't detect charucoboards in any of the images supplied. Try changing the patternSize passed into Charucoboard_create(), or try different pictures of charucoboards.")

        # Now that we've seen all of our images, perform the camera calibration
        # based on the set of points we've discovered
        calibration, bpy.types.Scene.cameraMatrix, bpy.types.Scene.distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
                charucoCorners=corners_all,
                charucoIds=ids_all,
                board=CHARUCO_BOARD,
                imageSize=image_size,
                cameraMatrix=None,
                distCoeffs=None)
            
        # Print matrix and distortion coefficient to the console
        print(bpy.types.Scene.cameraMatrix)
        print(bpy.types.Scene.distCoeffs)
            
        # Save values to be used where matrix+dist is required, for instance for posture estimation
        # I save files in a pickle file, but you can use yaml or whatever works for you
        f = open(self.directory+'calibration.pckl', 'wb')
        pickle.dump((bpy.types.Scene.cameraMatrix, bpy.types.Scene.distCoeffs, rvecs, tvecs), f)
        f.close()
            
        # Print to console our success
        print('Calibration successful. Calibration file used: {}'.format(self.directory+'calibration.pckl'))
        return {'FINISHED'}

class load_config(bpy.types.Operator, ImportHelper):
    bl_idname = "facebow.load_config"
    bl_label = "Select File"

    filter_glob: bpy.props.StringProperty(default='*.pckl', options={'HIDDEN'})

    def execute(self, context):
        bpy.context.scene.unit_settings.system = 'METRIC'
        bpy.context.scene.unit_settings.length_unit = 'MILLIMETERS'
        #bpy.context.scene.unit_settings.scale_length = 0.001
        filename, extension = os.path.splitext(self.filepath)
        print('Selected file:', self.filepath)
        print('File name:', filename)
        print('File extension:', extension)
        infile = open(self.filepath,'rb')
        try: 
            calibration_data = pickle.load(infile)
            (bpy.types.Scene.cameraMatrix, bpy.types.Scene.distCoeffs, _, _) = calibration_data
        except: print("Not a python pickled file.")
        infile.close()

        return {'FINISHED'}

class captured_patient_data_images(bpy.types.Operator, ImportHelper):
    bl_idname = "facebow.input_images"
    bl_label = "Select Image Records"

    filter_glob: bpy.props.StringProperty(default='*.*', options={'HIDDEN'})

    files = bpy.props.CollectionProperty(
            name="File Path",
            type=bpy.types.OperatorFileListElement,
            )
    directory = bpy.props.StringProperty(
            subtype='DIR_PATH',
            )


    def execute(self, context):
        """Do something with the selected file(s)."""

        directory = self.directory
        for file_elem in self.files:
            filepath = os.path.join(directory, file_elem.name)
            print(filepath)
        context.scene.pt_record = directory

        images = glob.glob(self.directory+'*.jpg')

        # Loop through images glob'ed
        bpy.types.Scene.tracker_instance = aruco_tracker(context, None, context.scene.debug_cv, True)
        for iname in images:
            bpy.types.Scene.tracker_instance.stream_processor(context, iname, context.scene.debug_cv, 'img')
        bpy.ops.wm.modal_timer_operator("INVOKE_DEFAULT")
        return {'FINISHED'}


class captured_patient_data(bpy.types.Operator, ImportHelper):
    bl_idname = "facebow.input"
    bl_label = "Select Record"

    filter_glob: bpy.props.StringProperty(default='*.mp4;*.jpeg;*.png;*.tif;*.tiff;*.bmp', options={'HIDDEN'})

    def execute(self, context):
        """Do something with the selected file(s)."""
        filename, extension = os.path.splitext(self.filepath)
        print('Selected file:', self.filepath)
        print('File name:', filename)
        print('File extension:', extension)
        context.scene.pt_record = self.filepath
        return {'FINISHED'}

class capture_input_data(bpy.types.Operator):
    bl_idname = "facebow.capture_input"
    bl_label = "Capture"

    def execute(self, context):
        #bpy.context.object.rotation_mode = 'XYZ'
        bpy.context.scene.unit_settings.system = 'METRIC'
        bpy.context.scene.unit_settings.length_unit = 'MILLIMETERS'

        if context.scene.legacy_file is False:
            if context.scene.live_cam == True:
                bpy.types.Scene.tracker_instance = aruco_tracker(context, debug=context.scene.debug_cv)
                bpy.ops.wm.modal_timer_operator("INVOKE_DEFAULT")
            else:
                bpy.types.Scene.tracker_instance = aruco_tracker(context, context.scene.pt_record, context.scene.debug_cv)
                bpy.ops.wm.modal_timer_operator("INVOKE_DEFAULT")
        elif context.scene.legacy_file is True:
            bpy.types.Scene.tracker_instance = aruco_tracker(context, context.scene.pt_record, context.scene.debug_cv, True)
            with open(os.path.join(context.scene.legacy_dir, 'data_stream.txt'), 'r') as reader:
                # Read and print the entire file line by line
                while True:  # The EOF char is an empty string
                    line = reader.readline()
                    if not line:
                        break
                    data = eval(line)
                    print(data)
                    context.scene.tracker_instance.update_tracking_marker(context, bpy.data.objects.get(str(data[0])), data[1], data[2], data[3]) # structure -> [ids[i][0], tvec, rvec, current_frame]
        return {'FINISHED'}

class analyze_data(bpy.types.Operator):
    bl_idname = "facebow.analyze"
    bl_label = "Analyze"

    def execute(self, context):
        context.scene.analyze = True
        return {'FINISHED'}

class analyze_condyles(bpy.types.Operator):
    bl_idname = "facebow.analyze_condyles"
    bl_label = "Analyze Condyles"

    def execute(self, context):
        if context.scene.mandibular_obj is not None and context.scene.frankfort_plane_enable is True:
            r_condyle = context.scene.frankfort_plane_points_post_R.location
            l_condyle = context.scene.frankfort_plane_points_post_L.location
            if bpy.data.objects.get("L_condyle_trace") is None and bpy.data.objects.get("R_condyle_trace") is None:
                bpy.ops.mesh.primitive_uv_sphere_add(radius=0.01, enter_editmode=False, align='WORLD', location=(0, 0, 0))
                context.view_layer.objects.active.name = "L_condyle_trace"
                bpy.data.objects["L_condyle_trace"].location = l_condyle
                if context.scene.L_condyle_offset_axis == 'X':
                    bpy.ops.transform.translate(value=(context.scene.L_condyle_offset/1000,0,0), orient_type='LOCAL')
                elif context.scene.L_condyle_offset_axis == 'Y':
                    bpy.ops.transform.translate(value=(0,context.scene.L_condyle_offset/1000,0), orient_type='LOCAL')
                elif context.scene.L_condyle_offset_axis == 'Z':
                    bpy.ops.transform.translate(value=(0,0,context.scene.L_condyle_offset/1000), orient_type='LOCAL') 
                bpy.data.objects["L_condyle_trace"].constraints.new(type='CHILD_OF')
                bpy.data.objects["L_condyle_trace"].constraints["Child Of"].target = context.scene.mandibular_obj
                bpy.data.objects["L_condyle_trace"].constraints["Child Of"].inverse_matrix = bpy.data.objects["L_condyle_trace"].constraints["Child Of"].target.matrix_world.inverted()
                bpy.ops.object.select_all(action='DESELECT')
                context.view_layer.objects.active = None
                '''
                bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0))
                bpy.context.view_layer.objects.active.name = "L_condyle_tracker"
                bpy.data.objects["L_condyle_tracker"].location = l_condyle
                bpy.ops.object.select_all(action='DESELECT')
                context.view_layer.objects.active = None
                '''
                bpy.ops.mesh.primitive_uv_sphere_add(radius=0.01, enter_editmode=False, align='WORLD', location=(0, 0, 0))
                context.view_layer.objects.active.name = "R_condyle_trace"
                bpy.data.objects["R_condyle_trace"].location = r_condyle
                if context.scene.R_condyle_offset_axis == 'X':
                    bpy.ops.transform.translate(value=(context.scene.R_condyle_offset/1000,0,0), orient_type='LOCAL')
                elif context.scene.R_condyle_offset_axis == 'Y':
                    bpy.ops.transform.translate(value=(0,context.scene.R_condyle_offset/1000,0), orient_type='LOCAL')
                elif context.scene.R_condyle_offset_axis == 'Z':
                    bpy.ops.transform.translate(value=(0,0,context.scene.R_condyle_offset/1000), orient_type='LOCAL') 
                bpy.data.objects["R_condyle_trace"].constraints.new(type='CHILD_OF')
                bpy.data.objects["R_condyle_trace"].constraints["Child Of"].target = context.scene.mandibular_obj
                bpy.data.objects["R_condyle_trace"].constraints["Child Of"].inverse_matrix = bpy.data.objects["R_condyle_trace"].constraints["Child Of"].target.matrix_world.inverted()
                bpy.ops.object.select_all(action='DESELECT')
                context.view_layer.objects.active = None
                '''
                bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0))
                bpy.context.view_layer.objects.active.name = "R_condyle_tracker"
                bpy.data.objects["R_condyle_tracker"].location = r_condyle
                bpy.ops.object.select_all(action='DESELECT')
                context.view_layer.objects.active = None
                '''
            else: pass

        return {'FINISHED'}

# https://github.com/4webmaster4/BD_Jaw_Tracker/blob/9fffdba6190a20f0767b292949e84624d529978c/Operators/BDJawTracker_Operators.py#L829
class smooth_keyframes(bpy.types.Operator):
    """ """

    bl_idname = "facebow.smoothkeyframes"
    bl_label = "Smooth selected object's keyframes"

    def execute(self, context):
        active_obj = bpy.context.view_layer.objects.active

        current_area = bpy.context.area.type
        layer = bpy.context.view_layer

        # change to graph editor
        bpy.context.area.type = "GRAPH_EDITOR"

        # smooth curves of all selected bones
        bpy.ops.graph.decimate(mode='RATIO', remove_ratio=0.25, remove_error_margin=0)
        bpy.ops.graph.smooth()

        # switch back to original area
        bpy.context.area.type = current_area                                    
        bpy.ops.object.select_all(action='DESELECT')

        return {"FINISHED"}


class show_keyframe_path(bpy.types.Operator):
    """ """

    bl_idname = "facebow.keyframe_path"
    bl_label = "Show motion path"

    def execute(self, context):
        scene = bpy.context.scene
        active_object = bpy.context.selected_objects
        bpy.ops.object.paths_calculate(start_frame=scene.frame_start, end_frame=scene.frame_end-5)

        return {"FINISHED"}

class segment_keyframes(bpy.types.Operator):
    """ """

    bl_idname = "facebow.segment_actionpath"
    bl_label = "Segment Path"

    def execute(self, context):
        obj = context.active_object
        curves = context.active_object.animation_data.action.fcurves
        dpath = obj.path_from_id("location")
        fc = curves.find(dpath, index=0)
        for k in fc.keyframe_points:
            fr = k.co[0]
            context.scene.frame_set(fr)
            #pos=obj.location
            #print(pos)
            if context.scene.keyframe_segment_start <= fr and context.scene.keyframe_segment_end >= fr:
                context.scene.tracker_instance.update_marker_tracing(context, obj, obj.matrix_world.translation, None, fr, context.scene.keyframe_segment_name)

        return {"FINISHED"}

class calc_cond_angle(bpy.types.Operator):
    """ """

    bl_idname = "facebow.calc_condyle_angle"
    bl_label = "Calculate"

    def execute(self, context):
        marker_obj = context.active_object
        context.scene.frankfork_plane_obj.data.vertices[0].co
        points = np.array([
            context.scene.frankfork_plane_obj.data.vertices[0].co, #the plane will have at minimum 3 vertices, as that is the geometric limit for plane definition
            context.scene.frankfork_plane_obj.data.vertices[1].co,
            context.scene.frankfork_plane_obj.data.vertices[2].co
            ])
        p0, p1, p2 = points
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        x2, y2, z2 = p2

        print(points)

        u = [x1-x0, y1-y0, z1-z0] #first vector
        v = [x2-x0, y2-y0, z2-z0] #sec vector

        plane_normal_vec = np.cross(u, v)

        print(plane_normal_vec)

        #get first point from marker
        context.scene.frame_set(context.scene.keyframe_condyle_angle_start)
        condyle_pos1 = np.array(marker_obj.location)

        #get second point from marker
        context.scene.frame_set(context.scene.keyframe_condyle_angle_end)
        condyle_pos2 = np.array(marker_obj.location)
        print(condyle_pos1, condyle_pos2)
        condyle_vector = condyle_pos2 - condyle_pos1
        print(condyle_vector)
        # Note: returns angle in radians
        def theta(v, w): return np.arccos(v.dot(w)/(np.linalg.norm(v)*np.linalg.norm(w)))
        context.scene.condylar_angle = 90 - np.degrees(theta(plane_normal_vec, condyle_vector))
        print(context.scene.condylar_angle)
        
        return {"FINISHED"}

class ODC_Facebow_Preferences(bpy.types.AddonPreferences):
    # this must match the add-on name, use '__package__'
    # when defining this in a submodule of a python package.
    bl_idname = __name__

    def draw(self, context):
        layout = self.layout
        #layout.label(text="Open Dental CAD Facebow Preferences:")
        row = layout.row()
        row = layout.row()
        row.label(text="Calibration Board Parameters")
        row = layout.row()
        row.label(text="Grid board resolution:")
        row.prop(context.scene, "cal_board_X_res")
        row.prop(context.scene, "cal_board_Y_res")
        row = layout.row()
        row.label(text="Grid board dimensions:")
        row.prop(context.scene, "cal_board_X_num")
        row.prop(context.scene, "cal_board_Y_num")
        row = layout.row()
        row.prop(context.scene, "cal_board_squareLength")
        row.prop(context.scene, "cal_board_markerLength")
        row = layout.row()
        row.operator("facebow.generate_aruco_board", text="Generate")
        row = layout.row()
        row = layout.row()
        row.label(text="Manual Markers Setup")
        row = layout.row()
        row.prop(context.scene, "manual_marker_num")
        row.prop(context.scene, "manual_marker_res")
        row = layout.row()
        row.operator("facebow.generate_aruco_marker", text="Generate")
        row.label(text="Appliances' Markers Setup")
        row = layout.row()
        row.operator("facebow.generate_aruco_appliance_board", text="Generate")
        row = layout.row()
        row.prop(context.scene, "debug_cv")
        row = layout.row()
        row.label(text="Live Camera Settings")
        row = layout.row()
        row.prop(context.scene, "FRAME_WIDTH")
        row.prop(context.scene, "FRAME_HEIGHT")
        row = layout.row()
        row.prop(context.scene, "VIDEO_FPS")
        row.prop(context.scene, "CAMERA_BRIGHTNESS")
        row = layout.row()
        row.prop(context.scene, "CAMERA_EXPOSURE_VAL")
        row.prop(context.scene, "CAMERA_AUTO_EXPOSURE")
        row = layout.row()
        row.prop(context.scene, "CAMERA_FOCUS_VAL")
        row.prop(context.scene, "CAMERA_AUTO_FOCUS")

class ODC_Facebow_Panel(bpy.types.Panel, ImportHelper):
    """Creates a Panel in the Object properties window"""
    bl_label = "ODC Facebow"
    bl_idname = "SCENE_PT_layout"
    bl_space_type = "VIEW_3D"
    bl_region_type = 'UI'
    bl_category = "Facebow"
    bl_context = ""
    

    def draw(self, context):
        layout = self.layout

        obj = context.object

        row = layout.row()
        row.label(text="Calibrate: ")
        row.operator("facebow.calibrate")
        row = layout.row()
        row.label(text="Configure: ")
        row.operator("facebow.load_config")

        row = layout.row()
        row.label(text="Facebow")
        row = layout.row()
        row.prop(context.scene, "live_cam")
        if context.scene.live_cam == False:
            row = layout.row()
            row.operator("facebow.input")
            row = layout.row()
            row.operator("facebow.input_images")
            row = layout.row()
            row.prop(context.scene, "pt_record")
        row = layout.row()
        row.operator("facebow.capture_input")
        row = layout.row()
        for x in context.scene.marker_obj_list:
            row = layout.row()
            row.label(text="Marker #"+str(x.marker_id)+":")
            row.prop(x, "b_obj")
            row.prop(x, "trace_origin")
            
        row = layout.row()
        row.operator("facebow.smoothkeyframes")
        row = layout.row()
        row.operator("facebow.keyframe_path")
        row = layout.row()
        row.label(text="Frankfort Plane:")
        row.prop(context.scene, "frankfort_plane_enable")
        if context.scene.frankfort_plane_enable == True:
            row = layout.row()
            row.label(text="Posterior Left:")
            row.prop(context.scene, "frankfort_plane_points_post_L")
            row = layout.row()
            row.prop(context.scene, "L_condyle_offset")
            row.prop(context.scene, "L_condyle_offset_axis")
            row = layout.row()
            row.label(text="Posterior Right:")
            row.prop(context.scene, "frankfort_plane_points_post_R")
            row = layout.row()
            row.prop(context.scene, "R_condyle_offset")
            row.prop(context.scene, "R_condyle_offset_axis")
            row = layout.row()
            row.label(text="Anterior:")
            row.prop(context.scene, "frankfort_plane_points_ant")
            row = layout.row()
            row = layout.row()
            row.label(text="Plane Object:")
            row.prop(context.scene, "frankfork_plane_obj")
            row = layout.row()
            row.label(text="Intercondylar Width:")
            row.label(text=str(round(context.scene.condylar_width, 1))+" mm")
        row = layout.row()
        row.label(text="Maxilla:")
        row = layout.row()
        row.prop(context.scene, "maxillary_obj")
        row = layout.row()
        row.label(text="Mandible:")
        row = layout.row()
        row.prop(context.scene, "mandibular_obj")
        row = layout.row()
        row.operator("facebow.analyze")
        if context.scene.frankfort_plane_enable == True and context.scene.analyze == True and context.scene.mandibular_obj is not None and context.scene.frankfort_plane_points_post_L is not None and context.scene.frankfort_plane_points_post_R is not None:
            row = layout.row()
            row.operator("facebow.analyze_condyles")

        if bpy.context.active_object.animation_data is not None and bpy.context.active_object.animation_data.action.fcurves is not None:
            row = layout.row()
            row.label(text="Motion Segmentation")
            row = layout.row()
            row.label(text="Keyframes: ")
            row.prop(context.scene, "keyframe_segment_start")
            row.prop(context.scene, "keyframe_segment_end")
            row = layout.row()
            row.prop(context.scene, "keyframe_segment_name")
            row = layout.row()
            row.operator("facebow.segment_actionpath")

        row = layout.row()
        row.label(text="Estimate Condylar Angle:")
        row = layout.row()
        row.prop(context.scene, "keyframe_condyle_angle_start")
        row.prop(context.scene, "keyframe_condyle_angle_end")
        row = layout.row()
        row.operator("facebow.calc_condyle_angle")
        row.label(text=str(round(context.scene.condylar_angle, 1))+" deg.")



class ModalTimerOperator(bpy.types.Operator):
    """Operator which runs its self from a timer"""
    bl_idname = "wm.modal_timer_operator"
    bl_label = "Modal Timer Operator"

    _timer = None

    def modal(self, context, event):
        if event.type in {'ESC'}:
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            while not context.scene.tracker_instance.queue.empty():
                try:
                    data = context.scene.tracker_instance.queue.get_nowait()
                    markerID = data[0]
                    if markerID not in [x.marker_id for x in context.scene.marker_obj_list]:
                        context.scene.marker_obj_list.add().marker_id = markerID
                    marker = [x for x in context.scene.marker_obj_list if x.marker_id == markerID][0]
                    if context.scene.analyze == True and marker.b_obj is not None:
                        context.scene.tracker_instance.update_tracking_marker(context, marker.b_obj, data[1], data[2], data[3])
                        if marker.trace_origin == True:
                            context.scene.tracker_instance.update_marker_tracing(context, marker.b_obj, data[1], data[2], data[3])
                        if context.scene.frankfort_plane_enable == True:
                            context.scene.tracker_instance.update_point_plane(context, (context.scene.frankfort_plane_points_ant.location, context.scene.frankfort_plane_points_post_R.location, context.scene.frankfort_plane_points_post_L.location), context.scene.frankfork_plane_obj)
                            context.scene.condylar_width, vect = context.scene.tracker_instance.update_vector_mag(context, context.scene.frankfort_plane_points_post_R.location, context.scene.frankfort_plane_points_post_L.location)
                        if context.scene.mandibular_obj is not None and context.scene.frankfort_plane_enable == True and bpy.data.objects.get("R_condyle_trace") is not None and bpy.data.objects.get("L_condyle_trace") is not None:
                            context.scene.tracker_instance.update_marker_tracing(context, bpy.data.objects.get("R_condyle_trace"), bpy.data.objects.get("R_condyle_trace").matrix_world.translation, None, data[3])
                            context.scene.tracker_instance.update_marker_tracing(context, bpy.data.objects.get("L_condyle_trace"), bpy.data.objects.get("L_condyle_trace").matrix_world.translation, None, data[3])
                except queue.Empty: continue
                context.scene.tracker_instance.queue.task_done()

            context.scene.tracker_instance.queue.join()

        return {'PASS_THROUGH'}

    def execute(self, context):
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)

class def_marker_obj_list(bpy.types.PropertyGroup):
    marker_id = bpy.props.IntProperty()
    b_obj = bpy.props.PointerProperty(name = "", type=bpy.types.Object)
    trace_origin = bpy.props.BoolProperty(name="", description="Enable location vertex tracing in separate mesh.", default=False)


def register():
    bpy.types.Scene.debug_cv = bpy.props.BoolProperty(name="Show openCV", description="Shows openCV aruco tracker window. May cause add-on instability.", default=True)
    bpy.types.Scene.legacy_file = bpy.props.BoolProperty(name="Legacy Process", description="Processes data via OpenCV, saved to a file then read into keyframes.", default=False)
    bpy.types.Scene.legacy_dir = os.path.dirname(os.path.abspath(__file__))

    bpy.types.Scene.aruco_dict = aruco.Dictionary_get(aruco.DICT_APRILTAG_36h11) # https://www.pyimagesearch.com/2020/12/14/generating-aruco-markers-with-opencv-and-python/
    bpy.types.Scene.cameraMatrix = None
    bpy.types.Scene.distCoeffs = None
    bpy.types.Scene.tracker_instance = None
    bpy.types.Scene.use_pose_board = bpy.props.BoolProperty(name="Use Pose Board", description="Use a group of markers to estimate position. Use of multiple markers increases accuracy.", default=True) #true during research

    bpy.types.Scene.live_cam = bpy.props.BoolProperty(name="Camera Stream", description="Use system default camera to tracking.", default=False)
    bpy.types.Scene.pt_record = bpy.props.StringProperty(name = "Record File", description = "Patient record containing aruco markers.", default = "")

    bpy.types.Scene.frankfort_plane_enable = bpy.props.BoolProperty(name="", description="Enable the Frankfort Plane tracking.", default=False)
    bpy.types.Scene.frankfort_plane_points_post_L = bpy.props.PointerProperty(name = "", type=bpy.types.Object) #bpy.props.StringProperty(name = "", description = "The 3 markers defining the Frankfort plane. Format ex.: 2,3,1", default = "")
    bpy.types.Scene.L_condyle_offset = bpy.props.FloatProperty(name="Offset", description="Offset amount in mm. Use pre-fix negative sign to invert axis direction", default=0)
    bpy.types.Scene.L_condyle_offset_axis = bpy.props.EnumProperty(name="", items=[ ('X', "X-Axis", ""),('Y', "Y-Axis", ""),('Z', "Z-Axis", "")])
    bpy.types.Scene.frankfort_plane_points_post_R = bpy.props.PointerProperty(name = "", type=bpy.types.Object)
    bpy.types.Scene.R_condyle_offset = bpy.props.FloatProperty(name="Offset", description="Offset amount in mm. Use pre-fix negative sign to invert axis direction", default=0)
    bpy.types.Scene.R_condyle_offset_axis = bpy.props.EnumProperty(name="", items=[ ('X', "X-Axis", ""),('Y', "Y-Axis", ""),('Z', "Z-Axis", "")])
    bpy.types.Scene.frankfort_plane_points_ant = bpy.props.PointerProperty(name = "", type=bpy.types.Object)
    bpy.types.Scene.frankfork_plane_obj = bpy.props.PointerProperty(name = "", type=bpy.types.Object)

    bpy.types.Scene.analyze = bpy.props.BoolProperty(default=False)

    bpy.types.Scene.condylar_width = bpy.props.FloatProperty(name="", description="", default=0, min=0.0)

    bpy.types.Scene.maxillary_obj = bpy.props.PointerProperty(name = "", type=bpy.types.Object)
    bpy.types.Scene.mandibular_obj = bpy.props.PointerProperty(name = "", type=bpy.types.Object)

    bpy.types.Scene.keyframe_segment_start = bpy.props.IntProperty(name="", description="", default=0)
    bpy.types.Scene.keyframe_segment_end = bpy.props.IntProperty(name="", description="", default=250)
    bpy.types.Scene.keyframe_segment_name = bpy.props.StringProperty(name = "Segment Name: ", description = "", default = "")

    bpy.types.Scene.keyframe_condyle_angle_start = bpy.props.IntProperty(name="Start", description="", default=0)
    bpy.types.Scene.keyframe_condyle_angle_end = bpy.props.IntProperty(name="End", description="", default=0)
    bpy.types.Scene.condylar_angle = bpy.props.FloatProperty(name="Condylar Angle", description="", default=0)

    bpy.types.Scene.FRAME_WIDTH = bpy.props.IntProperty(name="Width (px):", description="", default=1920)
    bpy.types.Scene.FRAME_HEIGHT = bpy.props.IntProperty(name="Height (px):", description="", default=1080)
    bpy.types.Scene.VIDEO_FPS = bpy.props.IntProperty(name="Frames/s (FPS):", description="", default=120)
    bpy.types.Scene.CAMERA_BRIGHTNESS = bpy.props.IntProperty(name="Brightness:", description="", default=12)
    bpy.types.Scene.CAMERA_AUTO_EXPOSURE = bpy.props.BoolProperty(name="Auto Exposure", description="", default=False)
    bpy.types.Scene.CAMERA_EXPOSURE_VAL = bpy.props.IntProperty(name="Exposure:", description="", default=-6)
    bpy.types.Scene.CAMERA_AUTO_FOCUS = bpy.props.BoolProperty(name="Auto Focus", description="", default=False)
    bpy.types.Scene.CAMERA_FOCUS_VAL = bpy.props.IntProperty(name="Focal Length (mm):", description="", default=90)


    bpy.types.Scene.cal_board_X_num = bpy.props.IntProperty(name="Width:", description="Number of markers arranged along the width (x-axis).", default=4, min=1)
    bpy.types.Scene.cal_board_Y_num = bpy.props.IntProperty(name="Height:", description="Number of markers arranged along the height (y-axis).", default=6, min=1)
    #Equivalent A4 paper dimensions in pixels at 300 DPI and 72 DPI respectively are: 2480 pixels x 3508 pixels (print resolution)
    bpy.types.Scene.cal_board_X_res = bpy.props.IntProperty(name="Width:", description="Board resolution along the width (for printing only).", default=2480, min=800)
    bpy.types.Scene.cal_board_Y_res = bpy.props.IntProperty(name="Height:", description="Board resolution along the height (for printing only).", default=3508, min=800)
    bpy.types.Scene.cal_board_squareLength = bpy.props.FloatProperty(name="Checkered square length:", description="in meters", soft_min=0.00, default=0.04)
    bpy.types.Scene.cal_board_markerLength = bpy.props.FloatProperty(name="Aruco marker lenght:", description="in meters", soft_min=0.00, default=0.03)
    bpy.types.Scene.facebow_marker_num = bpy.props.IntProperty(name="Number of markers:", description="Total number of markers to be generated for tracking.", default=9, min=1)
    bpy.types.Scene.facebow_marker_res = bpy.props.IntProperty(name="Marker size:", description="Tracking marker resolution.", default=1000, min=100)
    
    bpy.utils.register_class(ODC_Facebow_Preferences)
    bpy.utils.register_class(ODC_Facebow_Panel)
    bpy.utils.register_class(generate_tracking_marker)
    bpy.utils.register_class(generate_aruco_marker_board)
    bpy.utils.register_class(generate_calibration_board)
    bpy.utils.register_class(calibrate)
    bpy.utils.register_class(load_config)
    bpy.utils.register_class(captured_patient_data)
    bpy.utils.register_class(captured_patient_data_images)
    bpy.utils.register_class(capture_input_data)
    bpy.utils.register_class(analyze_data)
    bpy.utils.register_class(analyze_condyles)
    bpy.utils.register_class(smooth_keyframes)
    bpy.utils.register_class(show_keyframe_path)
    bpy.utils.register_class(segment_keyframes)
    bpy.utils.register_class(calc_cond_angle)

    bpy.utils.register_class(ModalTimerOperator)

    bpy.utils.register_class(def_marker_obj_list)
    bpy.types.Scene.marker_obj_list = bpy.props.CollectionProperty(type=def_marker_obj_list)

    ##############################################################################################
    bpy.types.Scene.pose_board_local_co = [
        np.array(
            [
                [0, 0.04125, 0.085417],
                [0, -0.04125, 0.085417],
                [0, -0.04125, 0.05625],
                [0, 0.04125, 0.05625]
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [0, 0.04125, 0.05125],
                [0, -0.04125, 0.05125],
                [0, -0.04125, 0.019583],
                [0, 0.04125, 0.019583]
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [0, 0.04125, 0.014583],
                [0, -0.04125, 0.014583],
                [0, -0.04125, -0.014583],
                [0, 0.04125, -0.014583]
            ],
            dtype=np.float32,
        ),
    ]

def unregister():
    del bpy.types.Scene.debug_cv
    del bpy.types.Scene.legacy_file
    del bpy.types.Scene.legacy_dir

    del bpy.types.Scene.aruco_dict
    del bpy.types.Scene.cameraMatrix
    del bpy.types.Scene.distCoeffs
    del bpy.types.Scene.tracker_instance
    del bpy.types.Scene.use_pose_board

    del bpy.types.Scene.live_cam
    del bpy.types.Scene.pt_record

    del bpy.types.Scene.frankfort_plane_enable
    del bpy.types.Scene.frankfort_plane_points_post_L
    del bpy.types.Scene.L_condyle_offset
    del bpy.types.Scene.L_condyle_offset_axis
    del bpy.types.Scene.frankfort_plane_points_post_R
    del bpy.types.Scene.R_condyle_offset
    del bpy.types.Scene.R_condyle_offset_axis
    del bpy.types.Scene.frankfort_plane_points_ant
    del bpy.types.Scene.frankfork_plane_obj

    del bpy.types.Scene.analyze

    del bpy.types.Scene.condylar_width

    del bpy.types.Scene.maxillary_obj
    del bpy.types.Scene.mandibular_obj

    del bpy.types.Scene.keyframe_segment_start
    del bpy.types.Scene.keyframe_segment_end
    del bpy.types.Scene.keyframe_segment_name

    del bpy.types.Scene.keyframe_condyle_angle_start
    del bpy.types.Scene.keyframe_condyle_angle_end
    del bpy.types.Scene.condylar_angle

    del bpy.types.Scene.FRAME_WIDTH
    del bpy.types.Scene.FRAME_HEIGHT
    del bpy.types.Scene.VIDEO_FPS
    del bpy.types.Scene.CAMERA_BRIGHTNESS
    del bpy.types.Scene.CAMERA_AUTO_EXPOSURE
    del bpy.types.Scene.CAMERA_EXPOSURE_VAL
    del bpy.types.Scene.CAMERA_AUTO_FOCUS
    del bpy.types.Scene.CAMERA_FOCUS_VAL

    del bpy.types.Scene.cal_board_X_num
    del bpy.types.Scene.cal_board_Y_num
    del bpy.types.Scene.cal_board_X_res
    del bpy.types.Scene.cal_board_Y_res
    del bpy.types.Scene.cal_board_squareLength
    del bpy.types.Scene.cal_board_markerLength
    del bpy.types.Scene.manual_marker_num
    del bpy.types.Scene.manual_marker_res
    
    bpy.utils.unregister_class(ODC_Facebow_Preferences)
    bpy.utils.unregister_class(ODC_Facebow_Panel)
    bpy.utils.unregister_class(generate_tracking_marker)
    bpy.utils.unregister_class(generate_aruco_marker_board)
    bpy.utils.unregister_class(generate_calibration_board)
    bpy.utils.unregister_class(calibrate)
    bpy.utils.unregister_class(load_config)
    bpy.utils.unregister_class(captured_patient_data)
    bpy.utils.unregister_class(captured_patient_data_images)
    bpy.utils.unregister_class(capture_input_data)
    bpy.utils.unregister_class(analyze_data)
    bpy.utils.unregister_class(analyze_condyles)
    bpy.utils.unregister_class(smooth_keyframes)
    bpy.utils.unregister_class(show_keyframe_path)
    bpy.utils.unregister_class(segment_keyframes)
    bpy.utils.unregister_class(calc_cond_angle)

    bpy.utils.unregister_class(ModalTimerOperator)

    bpy.utils.unregister_class(def_marker_obj_list)
    del bpy.types.Scene.marker_obj_list

    del bpy.types.Scene.pose_board_local_co


if __name__ == "__main__":
    register()
