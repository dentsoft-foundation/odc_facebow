import os, shutil
import pickle
import numpy as np
import bpy
from bpy_extras.io_utils import ImportHelper

import pip

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


bl_info = {
    "name": "Open Dental CAD Digital Facebow",
    "author": "Patrick R. Moore, Ilya Fomenko, Georgi Talmazov",
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
        board = aruco.GridBoard_create(context.scene.cal_board_X_num, context.scene.cal_board_Y_num, context.scene.cal_board_marker_length, context.scene.cal_board_marker_separation, context.scene.aruco_dict)
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
    bl_label = "Select File"

    filter_glob: bpy.props.StringProperty(default='*.*', options={'HIDDEN'})

    def execute(self, context):
        """Do something with the selected file(s)."""
        filename, extension = os.path.splitext(self.filepath)
        print('Selected file:', self.filepath)
        print('File name:', filename)
        print('File extension:', extension)
        infile = open(self.filepath,'rb')
        try: calibration_data = pickle.load(infile)
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
        row.label(text="Calibration Board Setup")
        row = layout.row()
        row.label(text="Grid board resolution:")
        row.prop(context.scene, "cal_board_X_res")
        row.prop(context.scene, "cal_board_Y_res")
        row = layout.row()
        row.label(text="Grid board dimensions:")
        row.prop(context.scene, "cal_board_X_num")
        row.prop(context.scene, "cal_board_Y_num")
        row = layout.row()
        row.prop(context.scene, "cal_board_marker_length")
        row.prop(context.scene, "cal_board_marker_separation")
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
        row.label(text="Calibration")
        row = layout.row()
        row.operator("facebow.calibrate")

        row = layout.row()
        row.label(text="Facebow")
        row = layout.row()
        row.operator("facebow.input")
        row = layout.row()
        row.prop(obj, "name")


def register():
    bpy.types.Scene.aruco_dict = aruco.Dictionary_get(aruco.DICT_APRILTAG_36h11)
    

    bpy.types.Scene.cal_board_X_num = bpy.props.IntProperty(name="Width:", description="Number of markers arranged along the width.", default=4, min=1)
    bpy.types.Scene.cal_board_Y_num = bpy.props.IntProperty(name="Height:", description="Number of markers arranged along the height.", default=5, min=1)
    bpy.types.Scene.cal_board_X_res = bpy.props.IntProperty(name="Width:", description="Number of markers arranged along the width.", default=864, min=800)
    bpy.types.Scene.cal_board_Y_res = bpy.props.IntProperty(name="Height:", description="Number of markers arranged along the height.", default=1080, min=800)
    bpy.types.Scene.cal_board_marker_length = bpy.props.FloatProperty(name="Marker length:", description="in cm", soft_min=0.00, default=3.75)
    bpy.types.Scene.cal_board_marker_separation = bpy.props.FloatProperty(name="Marker separation:", description="in cm", soft_min=0.00, default=0.50)
    bpy.types.Scene.facebow_marker_num = bpy.props.IntProperty(name="Number of markers:", description="Total number of markers to be generated for tracking.", default=4, min=1)
    bpy.types.Scene.facebow_marker_res = bpy.props.IntProperty(name="Marker size:", description="Tracking marker resolution.", default=700, min=100)
    
    bpy.utils.register_class(ODC_Facebow_Preferences)
    bpy.utils.register_class(ODC_Facebow_Panel)
    bpy.utils.register_class(generate_tracking_marker)
    bpy.utils.register_class(generate_calibration_board)
    bpy.utils.register_class(calibrate)
    bpy.utils.register_class(captured_patient_data)


def unregister():
    del bpy.types.Scene.aruco_dict

    del bpy.types.Scene.cal_board_X_num
    del bpy.types.Scene.cal_board_Y_num
    del bpy.types.Scene.cal_board_X_res
    del bpy.types.Scene.cal_board_Y_res
    del bpy.types.Scene.cal_board_marker_length
    del bpy.types.Scene.cal_board_marker_separation
    del bpy.types.Scene.facebow_marker_num
    del bpy.types.Scene.facebow_marker_res
    
    bpy.utils.unregister_class(ODC_Facebow_Preferences)
    bpy.utils.unregister_class(ODC_Facebow_Panel)
    bpy.utils.unregister_class(generate_tracking_marker)
    bpy.utils.unregister_class(generate_calibration_board)
    bpy.utils.unregister_class(calibrate)
    bpy.utils.unregister_class(captured_patient_data)


if __name__ == "__main__":
    register()
