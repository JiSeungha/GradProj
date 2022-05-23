import glfw
import time
import os
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *

############################################################################
# Global variables
gMousePress = [False, False]        # In orbit and panning, which mouse button is pressed [Left, Right]
gPreMousePos = [0, 0]               # In orbit and panning, previous mouse cursor position
gRotateAngles = [0, 40]             # In orbit, camera angle for [left-right, up-down] (unit : degree)
gMoveDistance = [0, 0]              # In panning, moving distance
gZoomScale = -100.0                 # In zooming, zoom scale

gIsPerspective = True               # Whether projection is perspective
gIsAnimateMotion = False            # Whether animate the loaded motion from bvh file.

frame_number = 0
bone_number = 0
############################################################################
# Render functions
def render(t, gGlobalOffset, desiredOffset):
    global gRotateAngles, gMoveDistance, gZoomScale
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
    glPointSize(10.)
    # Viewing transformation
    glTranslatef(gMoveDistance[0], -gMoveDistance[1], gZoomScale) # for panning and zoom function
    glRotatef(gRotateAngles[1], 1, 0, 0)    # rotate up-down
    glRotatef(gRotateAngles[0], 0, 1, 0)    # rotate left-right   
    glBegin(GL_POINTS)
    glColor3ub(0, 255, 0)
    glVertex3fv(np.array([0,0,0]))
    
    for i in range(1, bone_number):
        glColor3ub(255, 0, 0)
        current_coordinate = [gGlobalOffset[t][i][0] ,gGlobalOffset[t][i][1] ,gGlobalOffset[t][i][2]]
        glVertex3fv(np.array(current_coordinate))
        glColor3ub(0, 0, 255)
        current_coordinate = [desiredOffset[t][i][0], desiredOffset[t][i][1], desiredOffset[t][i][2]]
        glVertex3fv(np.array(current_coordinate))
    glEnd()
    glPopMatrix()
    glPopMatrix()


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
# Fetch functions
def fetchOutput(dir):
    global frame_number, bone_number
    file = open(dir,"r")
    file.seek(0)
    data = file.readlines()
    file.close()
    
    frame_number = len(data)

    for i in range(frame_number):
        data[i] = data[i].split()
    data = np.array(data, dtype = float)
    bone_number = (len(data[0]) - (36 + 3)) // 12
    globalOffset = [[0 for x in range(bone_number)] for y in range(frame_number)]
    for i in range(frame_number):
        x = 0
        y = 36
        for j in range(bone_number):
            globalOffset[i][x] = [data[i][y],data[i][y+1],data[i][y+2]]
            x = x + 1
            y = y + 12
    return globalOffset
############################################################################
# GLFW Callback functions
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


def main():
    global frame_number
    path_dir = str(os.getcwd()) + '/Export'
    desired_offset   = fetchOutput(path_dir + '/rightKnee_o.txt')
    generated_offset = fetchOutput(path_dir + '/rightKnee_withoutRoot_final_output_format_30.txt')
    
    if not glfw.init():
        return

    window = glfw.create_window(800, 800, "output", None, None)
    if not window:
        glfw.terminate()
        return

    # Set callback functions
    glfw.set_cursor_pos_callback(window, cursor_callback)
    glfw.set_mouse_button_callback(window, button_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    # Make the window's context current
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # Rendering
    t = 0
    while not glfw.window_should_close(window):
        glfw.poll_events()
        render(t, generated_offset, desired_offset)
        glfw.swap_buffers(window)
        if(t<frame_number-1):
            t = t + 1
        else:
            t = 0
        #time.sleep(0.03)
    glfw.terminate()

if __name__ == "__main__":
    main()
