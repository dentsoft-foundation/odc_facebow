import os
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
    import cv2
    from cv2 import aruco


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

class generate_calibration_board(bpy.types.Operator):
    bl_idname = "facebow.generate_aruco_board"
    bl_label = "Generate Calibration Board"

    def execute(self, context):
        aruco_dict = aruco.getPredefinedDictionary( aruco.DICT_6X6_1000 )
        markerLength = 3.75  # Here, measurement unit is centimetre.
        markerSeparation = 0.5   # Here, measurement unit is centimetre.
        board = aruco.GridBoard_create(4, 5, markerLength, markerSeparation, aruco_dict)
        print("generated calibration board")
        img = board.draw((864,1080))
        cv2.imshow("aruco", img)
        #newImage = bpy.data.images.new("aruco_board", 864, 1080, alpha=True, float_buffer=True)
        #newImage.pixels = np.asarray(img).flatten()
        #newImage.update()
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
        layout.label(text="Open Dental CAD Facebow Preferences:")
        row = layout.row()
        row.operator("facebow.generate_aruco_board")
        
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
    bpy.utils.register_class(ODC_Facebow_Preferences)
    bpy.utils.register_class(ODC_Facebow_Panel)
    bpy.utils.register_class(generate_calibration_board)
    bpy.utils.register_class(calibrate)
    bpy.utils.register_class(captured_patient_data)


def unregister():
    bpy.utils.unregister_class(ODC_Facebow_Preferences)
    bpy.utils.unregister_class(ODC_Facebow_Panel)
    bpy.utils.unregister_class(generate_calibration_board)
    bpy.utils.unregister_class(calibrate)
    bpy.utils.unregister_class(captured_patient_data)


if __name__ == "__main__":
    register()
