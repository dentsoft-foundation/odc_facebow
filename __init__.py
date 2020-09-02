import bpy

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

class HelloWorldPanel(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "ODC Facebow"
    bl_idname = "odc_facebow"
    bl_space_type = "VIEW_3D"
    bl_region_type = 'UI'
    bl_category = "Facebow"
    bl_context = "object"
    

    def draw(self, context):
        layout = self.layout

        obj = context.object

        row = layout.row()
        row.label(text="Calibration", icon='WORLD_DATA')

        row = layout.row()
        row.label(text="Facebow")
        row = layout.row()
        row.prop(obj, "name")

        row = layout.row()
        row.operator("mesh.primitive_cube_add")


def register():
    bpy.utils.register_class(HelloWorldPanel)


def unregister():
    bpy.utils.unregister_class(HelloWorldPanel)


if __name__ == "__main__":
    register()
