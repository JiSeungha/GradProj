import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np


############################################################################
class Mesh:
    def __init__(self, parms):
        self._etc = []
        if parms is not None:
            self._vertex = parms[0]
            self._index = parms[1]
            if len(parms) > 2:
                self._normal = parms[2]
        else:
            self._vertex = np.array([])
            self._index = []
            self._normal = np.array([])

    def set_etc(self, parm):
        self._etc = parm

    def get_vertex(self):
        return self._vertex

    def get_index(self):
        return self._index

    def get_index_size(self):
        return len(self._index)

    def get_normal(self):
        return self._normal

    def get_etc(self):
        return self._etc


class NodeBVH:
    def __init__(self):
        self._id = -1
        self._name = ""
        self._offset = np.array([0., 0., 0.], 'float32')
        self._channels_num = 0
        self._channels_str = []
        self._channels_start = 0
        self._children = []
        self._is_end = False

    def set_id(self, parm):
        self._id = parm

    def set_name(self, parm):
        self._name = parm

    def set_offset(self, parm):
        self._offset = np.array(parm, 'float32')

    def set_channels_str(self, start, parm):
        self._channels_start = start
        self._channels_num = int(parm[0])
        self._channels_str = parm[1:]

    def set_end(self, parm):
        self._is_end = True
        self.set_offset(parm)

    def get_id(self):
        return self._id

    def get_name(self):
        return self._name

    def get_offset(self):
        return self._offset

    def get_channels(self):
        return self._channels_start, self._channels_num, self._channels_str

    def get_children(self):
        return self._children

    def get_is_end(self):
        return self._is_end

    def add_child(self, parm):
        self._children.append(parm)


############################################################################
# Global variables
gMousePress = [False, False]        # In orbit and panning, which mouse button is pressed [Left, Right]
gPreMousePos = [0, 0]               # In orbit and panning, previous mouse cursor position
gRotateAngles = [0, 40]             # In orbit, camera angle for [left-right, up-down] (unit : degree)
gMoveDistance = [0, 0]              # In panning, moving distance
gZoomScale = -100.0                 # In zooming, zoom scale

gBVHTree = NodeBVH()                # BVH nodes' tree
gBVHMotion = []                     # loaded motion from BVH file
gBVHVertex = None                   # vertex array of BVH file
gBVHIndex = []                      # index array of BVH file
gBVHDict = []                       # transition information of BVH file.
gMesh = {}                          # obj meshes to draw on BVH model.

gBVHFrameTime = 0                   # frame time of BVH model
gBVHFrameNum = 0                    # frame number of BVH model
gMotionStartTime = 0                # when key "space" is pushed.

gIsPerspective = True               # Whether projection is perspective
gIsAnimateMotion = False            # Whether animate the loaded motion from bvh file.
gIsSampleBVH = False                # Whether an input file is sample file


############################################################################
# Render functions
def render(t):
    global gRotateAngles, gMoveDistance, gZoomScale
    global gBVHTree, gBVHMotion, gBVHDict
    global gBVHFrameTime, gBVHFrameNum, gMotionStartTime
    global gIsAnimateMotion, gIsSampleBVH

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    # Perspective projection or orthogonal projection
    if gIsPerspective:
        gluPerspective(45, 1, 1, 5000)
    else:
        glOrtho(-1, 1, -1, 1, 1, 5000)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glPushMatrix()

    # Viewing transformation
    glTranslatef(gMoveDistance[0],
                 -gMoveDistance[1],
                 gZoomScale)                # for panning and zoom function
    glRotatef(gRotateAngles[1], 1, 0, 0)    # rotate up-down
    glRotatef(gRotateAngles[0], 0, 1, 0)    # rotate left-right

    drawRectangular()               # Draw white rectangular on XZ plane.
    glPushMatrix()
    glLoadIdentity()

    # Get BVH vertex array at that time
    gBVHDict = []
    if gIsAnimateMotion:
        frame = int(int((t - gMotionStartTime) / gBVHFrameTime) % gBVHFrameNum)
        getBVHMotion(gBVHTree, gBVHMotion[frame], np.identity(4), None)
    else:
        getTPose(gBVHTree, np.identity(4), None)

    if gIsSampleBVH:
        glEnable(GL_NORMALIZE)  # Enable normal
        # glEnable(GL_RESCALE_NORMAL)
        setLight()              # Set light position and color, and material color
        glScalef(30, 30, 30)    # Scale

        i = 0
        while i < len(gBVHDict):
            node = gBVHDict[i]
            glPushMatrix()
            glMultMatrixf(node['matrix'].T)
            glPushMatrix()
            drawDetailMesh(node['name'])    # Draw obj files on bvh model
            i += 1
        glPopMatrix()
        glDisable(GL_LIGHTING)
    else:
        glPopMatrix()
        glScalef(.4, .4, .4)  # Scale
        drawLines()  # Draw lines between joint
        glPopMatrix()


def drawLines():
    global gBVHVertex, gBVHIndex

    glColor3f(.5, .5, .7)   # Set line color to purple
    glLineWidth(3.0)        # Set line width

    # Draw the model with lines
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 3 * gBVHVertex.itemsize, gBVHVertex)
    glDrawElements(GL_LINES, len(gBVHIndex), GL_UNSIGNED_INT, gBVHIndex)


def drawDetailMesh(name):
    global gMesh

    glPopMatrix()
    glPushMatrix()

    if name == "Head":
        glTranslatef(0.04, -0.07, 0)
        glScalef(0.06, 0.06, 0.06)
        drawSingleMesh(gMesh['helmet'])
    elif name == "Spine":
        glTranslatef(0.01, 0, 0)
        glScalef(0.07, 0.07, 0.07)
        drawSingleMesh(gMesh["body"])
    elif name == "Hips":
        glTranslatef(0, -0.3, 0.1)
        glScalef(0.06, 0.06, 0.06)
        drawSingleMesh(gMesh["skirt"])

    elif name == "RightForeArm":
        glTranslatef(0.1, 0.08, 0)
        glRotatef(-30, 0, 0, 1)
        glScalef(0.06, 0.06, 0.06)
        drawSingleMesh(gMesh["right_forearm"])
    elif name == "LeftForeArm":
        glTranslatef(-0.12, 0.02, 0.04)
        glRotatef(30, 0, 0, 1)
        glScalef(0.06, 0.06, 0.06)
        drawSingleMesh(gMesh["left_forearm"])

    elif name == "RightArm":
        glTranslatef(-0.35, 0.03, 0.01)
        glRotatef(-30, 0, 0, 1)
        glScalef(0.06, 0.06, 0.06)
        drawSingleMesh(gMesh["right_arm"])
    elif name == "LeftArm":
        glTranslatef(0.35, 0, -0.01)
        glRotatef(30, 0, 0, 1)
        glScalef(0.06, 0.06, 0.06)
        drawSingleMesh(gMesh["left_arm"])

    elif name == "RightHand":
        glTranslate(-0.01, 0, 0.06)
        glRotatef(-90, 0, 0, 1)
        glScalef(0.06, 0.06, 0.06)
        drawSingleMesh(gMesh["right_hand"])
    elif name == "LeftHand":
        glTranslate(0.03, 0, 0.06)
        glRotatef(90, 0, 0, 1)
        glScalef(0.06, 0.06, 0.06)
        drawSingleMesh(gMesh["left_hand"])

    elif name == "LeftUpLeg":
        glTranslatef(-0.04, -0.3, 0)
        glScalef(0.09, 0.09, 0.09)
        drawSingleMesh(gMesh["left_leg_upper"])
    elif name == "RightUpLeg":
        glTranslatef(0.04, -0.3, 0)
        glScalef(0.09, 0.09, 0.09)
        drawSingleMesh(gMesh["right_leg_upper"])

    elif name == "LeftLeg":
        glTranslatef(0.015, -0.15, 0)
        glScalef(0.03, 0.03, 0.03)
        drawSingleMesh(gMesh["calf"])
    elif name == "RightLeg":
        glTranslatef(-0.015, -0.15, 0)
        glScalef(0.03, 0.03, 0.03)
        drawSingleMesh(gMesh["calf"])

    elif name == "end_RightFoot":
        glTranslate(-0.04, 0, 0)
        glScalef(0.006, 0.006, 0.006)
        glRotatef(180, 0, 1, 0)
        drawSingleMesh(gMesh["right_shoe"])
    elif name == "end_LeftFoot":
        glScalef(0.006, 0.006, 0.006)
        glRotatef(180, 0, 1, 0)
        drawSingleMesh(gMesh["left_shoe"])

    glPopMatrix()
    glPopMatrix()


def drawSingleMesh(mesh):
    # Enable using vertex array and normal array.
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)

    # Match normal and vertex array, and vertex index array.
    glNormalPointer(GL_FLOAT, 3*mesh.get_normal().itemsize, mesh.get_normal())
    glVertexPointer(3, GL_FLOAT, 3*mesh.get_vertex().itemsize, mesh.get_vertex())
    glDrawElements(GL_TRIANGLES, mesh.get_index_size(), GL_UNSIGNED_INT, mesh.get_index())


def drawRectangular():
    glLineWidth(1.0)        # set line width
    glBegin(GL_LINES)
    glColor3f(.4, .4, .4)   # gray

    GRID_WIDTH = 1000.0
    GRID_GAP = 10.0

    i = -GRID_WIDTH
    while i <= GRID_WIDTH:
        glVertex3fv(np.array([i, 0, -GRID_WIDTH]))
        glVertex3fv(np.array([i, 0, GRID_WIDTH]))

        glVertex3fv(np.array([-GRID_WIDTH, 0, i]))
        glVertex3fv(np.array([GRID_WIDTH, 0, i]))

        i += GRID_GAP

    glEnd()


def drawFrame():
    glBegin(GL_LINES)
    glColor3ub(255, 0, 0)
    glVertex3fv(np.array([0., 0., 0.]))
    glVertex3fv(np.array([1., 0., 0.]))
    glColor3ub(0, 255, 0)
    glVertex3fv(np.array([0., 0., 0.]))
    glVertex3fv(np.array([0., 1., 0.]))
    glColor3ub(0, 0, 255)
    glVertex3fv(np.array([0., 0., 0]))
    glVertex3fv(np.array([0., 0., 1.]))
    glEnd()


def setLight():
    # Enable light
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHT1)
    glEnable(GL_LIGHT2)
    light = (.2, .2, .2, 1.)
    ambient_light_color = (.1, .1, .1, 1.)

    # Light 0
    glLightfv(GL_LIGHT0, GL_POSITION, (40, 40, 40, 1.))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light)
    glLightfv(GL_LIGHT0, GL_SPECULAR, light)
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient_light_color)

    # Light 1
    glLightfv(GL_LIGHT1, GL_POSITION, (-40, 40, 40, 1.))
    glLightfv(GL_LIGHT1, GL_DIFFUSE, light)
    glLightfv(GL_LIGHT1, GL_SPECULAR, light)
    glLightfv(GL_LIGHT1, GL_AMBIENT, ambient_light_color)

    # Light 2
    glLightfv(GL_LIGHT2, GL_POSITION, (40, 40, -40, 1.))
    glLightfv(GL_LIGHT2, GL_DIFFUSE, light)
    glLightfv(GL_LIGHT2, GL_SPECULAR, light)
    glLightfv(GL_LIGHT2, GL_AMBIENT, ambient_light_color)

    object_color = (.5, .5, .7, 1.)  # purple
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, object_color)
    glMaterialfv(GL_FRONT, GL_SHININESS, 10)
    glMaterialfv(GL_FRONT, GL_SPECULAR, (1., 1., 1., 1.))
    glPopMatrix()


############################################################################
# GLFW Callback functions
def drop_callback(window, path):
    file_name = path[0].split('/')
    file_name = file_name[len(file_name) - 1]
    file_name = file_name[-4:].lower()

    if file_name == '.bvh':
        fetchSkeletonFromBVH(path[0])


def key_callback(window, key, scancode, action, mods):
    global gIsPerspective
    global gIsAnimateMotion, gMotionStartTime

    if action == glfw.PRESS or action == glfw.REPEAT:
        # perspective projection <-> orthogonal projection
        if key == glfw.KEY_V:
            gIsPerspective = not gIsPerspective
            
        elif key == glfw.KEY_SPACE:
            gIsAnimateMotion = not gIsAnimateMotion
            if gIsAnimateMotion:
                gMotionStartTime = glfw.get_time()


def cursor_callback(window, xpos, ypos):
    global gRotateAngles, gMousePress, gPreMousePos, gIsPerspective

    if gMousePress[0]:
        # unit : degree
        gRotateAngles[0] += (xpos - gPreMousePos[0]) / 10
        gRotateAngles[1] += (ypos-gPreMousePos[1]) / 10
        gPreMousePos = [xpos, ypos]

    elif gMousePress[1]:
        gMoveDistance[0] += (xpos - gPreMousePos[0]) / 10   # (1000 if gIsPerspective else 100)
        gMoveDistance[1] += (ypos - gPreMousePos[1]) / 10   # (1000 if gIsPerspective else 100)
        gPreMousePos = [xpos, ypos]


def button_callback(window, button, action, mod):
    global gMousePress, gPreMousePos

    if action == glfw.PRESS:
        gPreMousePos = glfw.get_cursor_pos(window)
        if button == glfw.MOUSE_BUTTON_LEFT:
            gMousePress[0] = True
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            gMousePress[1] = True
    elif action == glfw.RELEASE:
        gMousePress = [False, False]


def scroll_callback(window, xoffset, yoffset):
    global gZoomScale, gRotateAngles
    zoom_scale = 10.
    max_offset = xoffset if np.abs(xoffset) > np.abs(yoffset) else yoffset
    gZoomScale += zoom_scale if max_offset > 0 else -zoom_scale


############################################################################
# Read OBJ files
# return [vertex_arr, index_arr, normal_data]
def fetchSingleMeshFromOBJ(path, print_info=False):
    # Read *.obj file
    file = open(path, "r")      # Open file buffer for reading obj file.
    file.seek(0)                # Change file pointer to first.
    obj = file.readlines()      # Read and push file's each lines into list.
    file.close()                # Close file buffer.

    vertex_arr = []             # vertex array
    index_arr = []              # vertex index array

    # Fetch mesh's information.
    for sentence in obj:
        entry = sentence.split()
        if len(entry) == 0:
            continue

        # Vertices
        if entry[0] == "v":
            vertex_arr += list(map(float, entry[1:]))

        # Face information
        elif entry[0] == "f":
            n = len(entry) - 1              # number of vertices per single face.
            vertex_index_origin = []        # origin tuple of vertex index

            for e in entry[1:]:
                t = e.split('/')            # index tuple : (vertex, texture, normal)
                v = int(t[0]) - 1           # vertex' index
                vertex_index_origin += [v]

            # Triangulation : n-polygon -> triangle
            for i in range(1, n-1):
                index_arr += [vertex_index_origin[0],
                              vertex_index_origin[i],
                              vertex_index_origin[i+1]]

    # Make normal for smooth shading
    # 1. make normal to face
    i = 0
    normal_to_face = [[] for _ in range(0, int(len(vertex_arr)/3))]
    while i < len(index_arr):
        v_i0 = np.array(vertex_arr[index_arr[i]*3:index_arr[i]*3+3])
        v_i1 = np.array(vertex_arr[index_arr[i+1]*3:index_arr[i+1]*3+3])
        v_i2 = np.array(vertex_arr[index_arr[i+2]*3:index_arr[i+2]*3+3])

        v1 = np.array(v_i1 - v_i0)
        v2 = np.array(v_i2 - v_i0)
        n = np.cross(v1, v2)    # normal of single triangle

        normal_to_face[index_arr[i]].append(n)
        normal_to_face[index_arr[i+1]].append(n)
        normal_to_face[index_arr[i+2]].append(n)
        i += 3

    # 2. vertex' normal = unit vector of near faces' normal
    i = 0
    normal_smooth = []
    while i < len(normal_to_face):
        temp = np.array([0., 0., 0.], 'float32')
        for j in normal_to_face[i]:
            temp += j
        if np.linalg.norm(temp) != 0:
            temp = temp / np.linalg.norm(temp)
        normal_smooth.append(temp)
        i += 1

    # Fetch global variables
    vertex_arr = np.array(vertex_arr, 'float32')
    normal_data = np.array(normal_smooth, 'float32')
    return [vertex_arr, index_arr, normal_data]


# Read BVH files
def fetchSkeletonFromBVH(path):
    global gBVHTree, gBVHIndex, gBVHVertex
    global gBVHMotion, gBVHFrameTime, gBVHFrameNum
    global gIsSampleBVH

    gIsSampleBVH = path.lower().find("sample") != -1

    # Read *.bvh file
    file = open(path, "r")      # Open file buffer for reading obj file.
    file.seek(0)                # Change file pointer to first.
    bvh = file.readlines()      # Read and push file's each lines into list.
    bvh = list(map(lambda s: s.strip(), bvh))
    file.close()                # Close file buffer.

    # Fetch skeleton's information.
    def readHierarchy(_i):
        __i = _i
        is_end = False

        stack = []
        tree = NodeBVH()
        parent = tree

        iarr = []
        node_count = 0
        channel_sum = 0

        joint_names = []

        while not is_end:
            sentence = bvh[__i].split()

            if sentence[0] == "ROOT" or sentence[0] == "JOINT":
                current = NodeBVH()
                current.set_id(node_count)                          # id
                current.set_name(sentence[1])                       # name
                current.set_offset(bvh[__i+2].split()[1:])          # offset
                current.set_channels_str(channel_sum,
                                         bvh[__i+3].split()[1:])    # channels

                channel_sum += int(bvh[__i+3].split()[1])
                __i += 4

                # Push to joint names array
                joint_names.append(sentence[1])

                # Push to index array
                node_count += 1
                if node_count > 1:
                    iarr += [parent.get_id(), current.get_id()]

                # Add to parents stack, parent's children stack and change parent <- itself
                stack.append(current)
                parent.add_child(current)
                parent = current

            elif sentence[0].upper() == "END":
                current = NodeBVH()
                current.set_id(node_count)                  # id
                current.set_end(bvh[__i+2].split()[1:])     # offset
                __i += 3

                # Push to index array
                node_count += 1
                iarr += [parent.get_id(), current.get_id()]

                # Add to parent's children stack
                parent.add_child(current)

                while bvh[__i].split()[0] == '}':
                    __i += 1
                    if len(stack) == 0:
                        is_end = True
                        break
                    parent = stack.pop()
                stack.append(parent)

        # Return (file line pointer, all node number, joint names, root, index array)
        return __i, joint_names, node_count, tree.get_children()[0], iarr

    def readMotion(_i):
        __i = _i
        motions = []

        frame_number = int(bvh[__i].split()[1])       # Frame number
        frame_time = float(bvh[__i+1].split()[2])     # Frame time
        __i += 2

        total = _i + frame_number + 2
        while __i < total:
            each_motion = []
            frame = bvh[__i].split()
            for entry in frame:
                each_motion.append(float(entry))
            motions.append(each_motion)
            __i += 1

        return __i, motions, frame_number, frame_time

    bvh_tree = None         # root of bvh hierarchical tree
    bvh_motions = None      # array of each frame's motion
    bvh_all_node_num = 0
    bvh_joint_names = []
    iarr = []             # index array
    bvh_frame_number = bvh_frame_time = None

    i = 0
    length = len(bvh)
    while i < length:
        if bvh[i].upper() == "HIERARCHY":
            point, \
            bvh_joint_names, \
            bvh_all_node_num, \
            bvh_tree, \
            iarr = readHierarchy(i+1)
            i = point

        elif bvh[i].upper() == "MOTION":
            point,\
            bvh_motions, \
            bvh_frame_number,\
            bvh_frame_time = readMotion(i+1)
            i = point

    # Calculate t-pose model.
    gBVHIndex = iarr
    gBVHVertex = np.zeros(bvh_all_node_num*3, 'float32')
    gBVHFrameTime = bvh_frame_time
    gBVHFrameNum = bvh_frame_number

    gBVHTree = bvh_tree
    gBVHMotion = bvh_motions


def getTPose(node, T, parent):
    global gBVHVertex, gBVHDict
    nid = node.get_id()
    offset = node.get_offset()

    T = T @ translate(0, offset[0])
    T = T @ translate(1, offset[1])
    T = T @ translate(2, offset[2])

    if node.get_is_end():
        new = {
            'nid': nid,
            'name': "end_" + parent.get_name(),
            'matrix': T
        }
        gBVHDict.append(new)
        gBVHVertex[nid * 3: nid * 3 + 3] = (T @ np.array([0, 0, 0, 1]))[:3]

    for child in node.get_children():
        getTPose(child, T, node)

    new = {
        'nid': nid,
        'name': node.get_name(),
        'matrix': T
    }
    gBVHDict.append(new)
    gBVHVertex[nid * 3: nid * 3 + 3] = (T @ np.array([0, 0, 0, 1]))[:3]


def getBVHMotion(node, motions, T, parent):
    global gBVHVertex, gBVHDict
    nid = node.get_id()
    start, num, channels = node.get_channels()
    offset = node.get_offset()

    T = T @ translate(0, offset[0])
    T = T @ translate(1, offset[1])
    T = T @ translate(2, offset[2])

    if node.get_is_end():
        new = {
            'nid': nid,
            'name': "end_" + parent.get_name(),
            'matrix': T
        }
        gBVHDict.append(new)
        gBVHVertex[nid * 3: nid * 3 + 3] = (T @ np.array([0, 0, 0, 1]))[:3]

    i = 0
    while i < num:
        if channels[i].upper() == "XROTATION":
            T = T @ rotate(0, motions[start+i])
        elif channels[i].upper() == "YROTATION":
            T = T @ rotate(1, motions[start+i])
        elif channels[i].upper() == "ZROTATION":
            T = T @ rotate(2, motions[start+i])
        elif channels[i].upper() == "XPOSITION":
            T = T @ translate(0, motions[start+i])
        elif channels[i].upper() == "YPOSITION":
            T = T @ translate(1, motions[start+i])
        elif channels[i].upper() == "ZPOSITION":
            T = T @ translate(2, motions[start+i])
        else:
            continue
        i += 1

    for child in node.get_children():
        getBVHMotion(child, motions, T, node)

    new = {
        'nid': nid,
        'name': node.get_name(),
        'matrix': T
    }
    gBVHDict.append(new)
    gBVHVertex[nid * 3: nid * 3 + 3] = (T @ np.array([0, 0, 0, 1]))[:3]


def rotate(direct, motion):
    R = np.identity(4)
    ang = np.deg2rad(motion)
    if direct == 0:     # Rotate X
        R[:3, :3] = np.array([[1, 0, 0],
                              [0, np.cos(ang), -np.sin(ang)],
                              [0, np.sin(ang), np.cos(ang)]])
    elif direct == 1:   # Rotate Y
        R[:3, :3] = np.array([[np.cos(ang), 0, np.sin(ang)],
                              [0, 1, 0],
                              [-np.sin(ang), 0, np.cos(ang)]])
    elif direct == 2:   # Rotate Z
        R[:3, :3] = np.array([[np.cos(ang), -np.sin(ang), 0],
                              [np.sin(ang), np.cos(ang), 0],
                              [0, 0, 1]])
    return R


def translate(direct, motion):
    T = np.identity(4)
    if direct == 0:     # Translate X
        T[:3, 3] = np.array([motion, 0, 0])
    elif direct == 1:   # Translate Y
        T[:3, 3] = np.array([0, motion, 0])
    elif direct == 2:   # Translate Z
        T[:3, 3] = np.array([0, 0, motion])
    return T

def main():
    global gMesh

    if not glfw.init():
        return

    window = glfw.create_window(800, 800, "BVH", None, None)
    if not window:
        glfw.terminate()
        return

    # Set callback functions
    glfw.set_key_callback(window, key_callback)
    glfw.set_cursor_pos_callback(window, cursor_callback)
    glfw.set_mouse_button_callback(window, button_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_drop_callback(window, drop_callback)

    # Make the window's context current
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    fetchSkeletonFromBVH('./work.bvh')

    # Rendering
    t = 0
    while not glfw.window_should_close(window):
        if gIsAnimateMotion:
            t = glfw.get_time()
        render(t)
        glfw.poll_events()
        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()
