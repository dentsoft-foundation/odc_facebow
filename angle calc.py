        marker_obj = context.active_object

        #get first point from marker
        context.scene.frame_set(context.scene.keyframe_condyle_angle_start)
        condyle_pos1 = marker_obj.location

        #get second point from marker
        context.scene.frame_set(context.scene.keyframe_condyle_angle_end)
        condyle_pos2 = marker_obj.location

        v0 = context.scene.frankfork_plane_obj.data.vertices[0].co
        v1 = context.scene.frankfork_plane_obj.data.vertices[1].co
        v2 =context.scene.frankfork_plane_obj.data.vertices[2].co
        Vec1 = v1 - v0
        Vec2 = v2 - v0
        
        plane_normal_vec = Vec1.cross(Vec2) #no safety on direction here
        plane_normal_vec.normalize() # << important this may be your issue if not normalizing the plane vector
        condyle_vector = condyle_pos2 - condyle_pos1 #yes
        dz = condyle_vector.dot(plane_normal_vec)
        dxy = condyle_vector - condyle_vector.dot(plane_normal_vec) * plane_normal_vec
        print(dxy, condyle_vector, plane_normal_vec)
        theta = math.atan(abs(dz)/dxy.length) #tan(theta) = op/adjacent

        context.scene.condylar_angle = theta
        print(context.scene.condylar_angle)