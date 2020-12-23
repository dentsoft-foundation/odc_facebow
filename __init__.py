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
    def __init__(self, context, data_source=cv2.CAP_DSHOW, debug=False):
        self.processor_thread = threading.Thread(target=self.stream_processor, args=(context, data_source, debug))
        self.processor_thread.start()
        self.queue = queue.Queue()

    def update_tracking_marker(self, context, markerID, tvec, rvec):
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
        mat_sca = Matrix.Scale(0.01, 4)
        mat_trans = Matrix.Translation((float(tvec[0][0][0]), float(tvec[0][0][1]), float(tvec[0][0][2])))
        # https://devtalk.blender.org/t/understanding-matrix-operations-in-blender/10148
        mat_rot_x = Matrix.Rotation(r_euler[0], 4, 'X')
        mat_rot_y = Matrix.Rotation(r_euler[1], 4, 'Y')
        mat_rot_z = Matrix.Rotation(r_euler[2], 4, 'Z')
        bpy.data.objects[str(markerID)].matrix_world = mat_trans @ mat_rot_x @ mat_rot_y @ mat_rot_z @ mat_sca
        dg = context.evaluated_depsgraph_get()
        dg.update()
        #print(markerID, final)
        if bpy.data.objects.get(str(markerID)+"_trace") is None:
            mesh = bpy.data.meshes.new(str(markerID)+"_trace_data")  # add a new mesh
            obj = bpy.data.objects.new(str(markerID)+"_trace", mesh)  # add a new object using the mesh
            bpy.context.scene.collection.objects.link(obj)  # put the object into the scene (link)
        else:
            obj = bpy.data.objects.get(str(markerID)+"_trace")
        
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.verts.new((float(tvec[0][0][0]), float(tvec[0][0][1]), float(tvec[0][0][2])))
        bm.verts.ensure_lookup_table()
        if len(bm.verts) > 1: bm.edges.new((bm.verts[-2], bm.verts[-1]))
        bm.to_mesh(obj.data)
        bm.free()

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


    def stream_processor(self, context, data_source, debug=False):
        # Constant parameters used in Aruco methods
        ARUCO_PARAMETERS = aruco.DetectorParameters_create()
        ARUCO_DICT = context.scene.aruco_dict
        ARUCO_PARAMETERS.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG

        # Create vectors we'll be using for rotations and translations for postures
        rvecs, tvecs = None, None

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
        while True:
            success, img = cap.read()
            imgGrey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #imgBlur = cv2.GaussianBlur(imgGrey,(11,11),cv2.BORDER_DEFAULT)
            #imgPyrDown = cv2.pyrDown(imgGrey)

            # lists of ids and the corners beloning to each id
            corners, ids, rejected_img_points = aruco.detectMarkers(imgGrey, ARUCO_DICT, parameters=ARUCO_PARAMETERS, cameraMatrix=context.scene.cameraMatrix, distCoeff=context.scene.distCoeffs)
            #for one in rejected_img_points:
            #    more_corners, more_ids, rej, recovered_ids = cv2.aruco.refineDetectedMarkers(imgPyrDown, board, corners, ids, one)
            
            # Outline all of the markers detected in our image
            
            if np.all(ids is not None):  # If there are markers found by detector
                for i in range(0, len(ids)):  # Iterate in markers
                    # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                    rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.021, context.scene.cameraMatrix, context.scene.distCoeffs) 
                    (rvec - tvec).any()  # get rid of that nasty numpy value array error
                                
                    #print(ids[i], tvec[0][0])
                    #print(ids)
                    
                    cv2.aruco.drawDetectedMarkers(img,corners,ids)
                    aruco.drawAxis(img, context.scene.cameraMatrix, context.scene.distCoeffs, rvec, tvec, 0.02)  # Draw Axis

                    self.queue.put([ids[i][0], tvec, rvec])
                
            if debug == True:
                cv2.namedWindow("img", cv2.WINDOW_NORMAL)
                #cv2.namedWindow("PyrDown", cv2.WINDOW_NORMAL)
                cv2.imshow("img", img)
                #cv2.imshow("PyrDown", imgPyrDown)
                if cv2.waitKey(1) & 0xFF ==ord('q'):
                    break
        if debug == True: cv2.destroyAllWindows()
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
        for i in range(0, context.scene.facebow_marker_num):
            img = aruco.drawMarker(context.scene.aruco_dict, i, context.scene.facebow_marker_res)
            cv2.imwrite(os.path.join(folder, str(i)+".jpg"), img)
        return {'FINISHED'}

class generate_calibration_board(bpy.types.Operator):
    bl_idname = "facebow.generate_aruco_board"
    bl_label = "Generate Calibration Board"

    def execute(self, context):
        folder = "C:/Users/talmazovg/AppData/Roaming/Blender Foundation/Blender/2.83/scripts/addons/odc_facebow"
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

class analyze_data(bpy.types.Operator):
    bl_idname = "facebow.analyze"
    bl_label = "Analyze"

    def execute(self, context):
        bpy.context.object.rotation_mode = 'XYZ'
        bpy.context.scene.unit_settings.system = 'METRIC'
        bpy.context.scene.unit_settings.length_unit = 'MILLIMETERS'

        if context.scene.live_cam == True:
            bpy.types.Scene.tracker_instance = aruco_tracker(context, debug=context.scene.debug_cv)
            bpy.ops.wm.modal_timer_operator("INVOKE_DEFAULT")
        else:
            bpy.types.Scene.tracker_instance = aruco_tracker(context, context.scene.pt_record, context.scene.debug_cv)
            bpy.ops.wm.modal_timer_operator("INVOKE_DEFAULT")
        return {'FINISHED'}

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
        row.label(text="Markers Setup")
        row = layout.row()
        row.prop(context.scene, "facebow_marker_num")
        row.prop(context.scene, "facebow_marker_res")
        row = layout.row()
        row.operator("facebow.generate_aruco_marker", text="Generate")
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
            row.prop(context.scene, "pt_record")
        row = layout.row()
        row.label(text="Frankfort Markers:")
        row = layout.row()
        row.label(text="Posterior Left:")
        row.prop(context.scene, "frankfort_plane_points_post_L")
        row = layout.row()
        row.label(text="Posterior Right:")
        row.prop(context.scene, "frankfort_plane_points_post_R")
        row = layout.row()
        row.label(text="Anterior:")
        row.prop(context.scene, "frankfort_plane_points_ant")
        row = layout.row()
        row = layout.row()
        row.label(text="Frankfort Plane:")
        row.prop(context.scene, "frankfork_plane_obj")
        row = layout.row()
        row.operator("facebow.analyze")
        row = layout.row()
        row.label(text="Intercondylar Width:")
        row.label(text=str(round(context.scene.condylar_width, 1))+" mm")

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
                    context.scene.tracker_instance.update_tracking_marker(context, data[0], data[1], data[2])
                    context.scene.tracker_instance.update_point_plane(context, (context.scene.frankfort_plane_points_ant.location, context.scene.frankfort_plane_points_post_R.location, context.scene.frankfort_plane_points_post_L.location), context.scene.frankfork_plane_obj)
                    context.scene.condylar_width, vect = context.scene.tracker_instance.update_vector_mag(context, context.scene.frankfort_plane_points_post_R.location, context.scene.frankfort_plane_points_post_L.location)
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


def register():
    bpy.types.Scene.debug_cv = bpy.props.BoolProperty(name="Show openCV", description="Shows openCV aruco tracker window. May cause add-on instability.", default=True)

    bpy.types.Scene.aruco_dict = aruco.Dictionary_get(aruco.DICT_APRILTAG_36h11)
    bpy.types.Scene.cameraMatrix = None
    bpy.types.Scene.distCoeffs = None
    bpy.types.Scene.tracker_instance = None

    bpy.types.Scene.live_cam = bpy.props.BoolProperty(name="Camera Stream", description="Use system default camera to tracking.", default=False)
    bpy.types.Scene.pt_record = bpy.props.StringProperty(name = "Record File", description = "Patient record containing aruco markers.", default = "")

    bpy.types.Scene.frankfort_plane_points_post_L = bpy.props.PointerProperty(name = "", type=bpy.types.Object) #bpy.props.StringProperty(name = "", description = "The 3 markers defining the Frankfort plane. Format ex.: 2,3,1", default = "")
    bpy.types.Scene.frankfort_plane_points_post_R = bpy.props.PointerProperty(name = "", type=bpy.types.Object)
    bpy.types.Scene.frankfort_plane_points_ant = bpy.props.PointerProperty(name = "", type=bpy.types.Object)
    bpy.types.Scene.frankfork_plane_obj = bpy.props.PointerProperty(name = "", type=bpy.types.Object)

    bpy.types.Scene.condylar_width = bpy.props.FloatProperty(name="", description="", default=0, min=0.0)

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
    bpy.types.Scene.cal_board_X_res = bpy.props.IntProperty(name="Width:", description="Board resolution along the width (for printing only).", default=864, min=800)
    bpy.types.Scene.cal_board_Y_res = bpy.props.IntProperty(name="Height:", description="Board resolution along the height (for printing only).", default=1080, min=800)
    bpy.types.Scene.cal_board_squareLength = bpy.props.FloatProperty(name="Checkered square length:", description="in meters", soft_min=0.00, default=0.044)
    bpy.types.Scene.cal_board_markerLength = bpy.props.FloatProperty(name="Aruco marker lenght:", description="in meters", soft_min=0.00, default=0.02)
    bpy.types.Scene.facebow_marker_num = bpy.props.IntProperty(name="Number of markers:", description="Total number of markers to be generated for tracking.", default=4, min=1)
    bpy.types.Scene.facebow_marker_res = bpy.props.IntProperty(name="Marker size:", description="Tracking marker resolution.", default=700, min=100)
    
    bpy.utils.register_class(ODC_Facebow_Preferences)
    bpy.utils.register_class(ODC_Facebow_Panel)
    bpy.utils.register_class(generate_tracking_marker)
    bpy.utils.register_class(generate_calibration_board)
    bpy.utils.register_class(calibrate)
    bpy.utils.register_class(load_config)
    bpy.utils.register_class(captured_patient_data)
    bpy.utils.register_class(analyze_data)

    bpy.utils.register_class(ModalTimerOperator)


def unregister():
    del bpy.types.Scene.debug_cv

    del bpy.types.Scene.aruco_dict
    del bpy.types.Scene.cameraMatrix
    del bpy.types.Scene.distCoeffs
    del bpy.types.Scene.tracker_instance

    del bpy.types.Scene.live_cam
    del bpy.types.Scene.pt_record

    del bpy.types.Scene.frankfort_plane_points_post_L
    del bpy.types.Scene.frankfort_plane_points_post_R
    del bpy.types.Scene.frankfort_plane_points_ant
    del bpy.types.Scene.frankfork_plane_obj

    del bpy.types.Scene.condylar_width

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
    del bpy.types.Scene.facebow_marker_num
    del bpy.types.Scene.facebow_marker_res
    
    bpy.utils.unregister_class(ODC_Facebow_Preferences)
    bpy.utils.unregister_class(ODC_Facebow_Panel)
    bpy.utils.unregister_class(generate_tracking_marker)
    bpy.utils.unregister_class(generate_calibration_board)
    bpy.utils.unregister_class(calibrate)
    bpy.utils.unregister_class(load_config)
    bpy.utils.unregister_class(captured_patient_data)
    bpy.utils.unregister_class(analyze_data)

    bpy.utils.unregister_class(ModalTimerOperator)


if __name__ == "__main__":
    register()
