import os
import bpy
from bpy_extras.io_utils import ImportHelper

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

class calibrate(bpy.types.Operator, ImportHelper):
    bl_idname = "facebow.calibrate"
    bl_label = "Select File"

    filter_glob: bpy.props.StringProperty(default='*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp', options={'HIDDEN'})

    def execute(self, context):
        """Do something with the selected file(s)."""
        filename, extension = os.path.splitext(self.filepath)
        print('Selected file:', self.filepath)
        print('File name:', filename)
        print('File extension:', extension)
        return {'FINISHED'}

class captured_patient_data(bpy.types.Operator, ImportHelper):
    bl_idname = "facebow.input"
    bl_label = "Select Record"

    filter_glob: bpy.props.StringProperty(default='*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp', options={'HIDDEN'})

    def execute(self, context):
        """Do something with the selected file(s)."""
        filename, extension = os.path.splitext(self.filepath)
        print('Selected file:', self.filepath)
        print('File name:', filename)
        print('File extension:', extension)
        return {'FINISHED'}


class HelloWorldPanel(bpy.types.Panel, ImportHelper):
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
    bpy.utils.register_class(HelloWorldPanel)
    bpy.utils.register_class(calibrate)
    bpy.utils.register_class(captured_patient_data)


def unregister():
    bpy.utils.unregister_class(HelloWorldPanel)
    bpy.utils.unregister_class(calibrate)
    bpy.utils.unregister_class(captured_patient_data)


if __name__ == "__main__":
    register()
