import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw

import Bvh

# global variables
width = 1024
height = 1024
isLeft = False
isRight = False
originX = width / 2.0
originY = height / 2.0

# camera
gAzimuth = 0.0
gElevation = 0.0
moveX = 0.0
moveY = 0.0
gZoom = 50.0

# bvh
gMove = False
bvh = None
gMotionStartTime = 0
gFrame = 0
gOffset = 6


# draw a cube of side 2, centered at the origin.
def drawCube():
    glBegin(GL_QUADS)
    glVertex3f(1.0, 1.0, -1.0)

    glVertex3f(-1.0, 1.0, -1.0)
    glVertex3f(-1.0, 1.0, 1.0)
    glVertex3f(1.0, 1.0, 1.0)

    glVertex3f(1.0, -1.0, 1.0)
    glVertex3f(-1.0, -1.0, 1.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glVertex3f(1.0, -1.0, -1.0)

    glVertex3f(1.0, 1.0, 1.0)
    glVertex3f(-1.0, 1.0, 1.0)
    glVertex3f(-1.0, -1.0, 1.0)
    glVertex3f(1.0, -1.0, 1.0)

    glVertex3f(1.0, -1.0, -1.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glVertex3f(-1.0, 1.0, -1.0)
    glVertex3f(1.0, 1.0, -1.0)

    glVertex3f(-1.0, 1.0, 1.0)
    glVertex3f(-1.0, 1.0, -1.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glVertex3f(-1.0, -1.0, 1.0)

    glVertex3f(1.0, 1.0, -1.0)
    glVertex3f(1.0, 1.0, 1.0)
    glVertex3f(1.0, -1.0, 1.0)
    glVertex3f(1.0, -1.0, -1.0)
    glEnd()


# draw a sphere of radius 1, centered at the origin.
# numLats: number of latitude segments
# numLongs: number of longitude segments
def drawSphere(numLats=12, numLongs=12):
    for i in range(0, numLats + 1):
        lat0 = np.pi * (-0.5 + float(float(i - 1) / float(numLats)))
        z0 = np.sin(lat0)
        zr0 = np.cos(lat0)

        lat1 = np.pi * (-0.5 + float(float(i) / float(numLats)))
        z1 = np.sin(lat1)
        zr1 = np.cos(lat1)

        # Use Quad strips to draw the sphere
        glBegin(GL_QUAD_STRIP)

        for j in range(0, numLongs + 1):
            lng = 2 * np.pi * float(float(j - 1) / float(numLongs))
            x = np.cos(lng)
            y = np.sin(lng)
            glVertex3f(x * zr0, y * zr0, z0)
            glVertex3f(x * zr1, y * zr1, z1)

        glEnd()


def drawFrame(len):
    glDisable(GL_LIGHTING)

    # Start drawing lines.
    glBegin(GL_LINES)
    glColor3d(1, 0, 0)
    glVertex3d(0, 0, 0)
    glVertex3d(len, 0, 0)
    glVertex3d(0, 0, 0)
    glVertex3d(-len, 0, 0)

    # color of y-axis is green.
    glColor3d(0, 1, 0)
    glVertex3d(0, 0, 0)
    glVertex3d(0, len, 0)
    glVertex3d(0, 0, 0)
    glVertex3d(0, -len, 0)

    # color of z-axis is  blue.
    glColor3d(0, 0, 1)
    glVertex3d(0, 0, 0)
    glVertex3d(0, 0, len)
    glVertex3d(0, 0, 0)
    glVertex3d(0, 0, -len)

    glEnd()


def drawGrid(len):
    glDisable(GL_LIGHTING)

    # Start drawing lines.
    glBegin(GL_LINES)
    glColor3d(100, 100, 100)

    # x-grid
    for i in range(1, len):
        glVertex3d(i * 3, 0, 0)
        glVertex3d(i * 3, 0, len)
        glVertex3d(i * 3, 0, 0)
        glVertex3d(i * 3, 0, -len)

        glVertex3d(-i * 3, 0, 0)
        glVertex3d(-i * 3, 0, len)
        glVertex3d(-i * 3, 0, 0)
        glVertex3d(-i * 3, 0, -len)

    # z-grid
    for i in range(1, len):
        glVertex3d(0, 0, i * 3)
        glVertex3d(len, 0, i * 3)
        glVertex3d(0, 0, i * 3)
        glVertex3d(-len, 0, i * 3)

        glVertex3d(0, 0, -i * 3)
        glVertex3d(len, 0, -i * 3)
        glVertex3d(0, 0, -i * 3)
        glVertex3d(-len, 0, -i * 3)

    glEnd()


def drawModel():
    t = 2 * glfw.get_time()

    glColor3d(255, 255, 0)

    # body
    glPushMatrix()
    glScalef(.2, .5, .2)
    drawSphere()
    glPopMatrix()

    # left thigh
    glColor3d(0, 0, 255)
    glPushMatrix()
    glTranslatef(-.2, -.5, 0)
    glRotate(45 * np.sin(t), 1, 0, 0)

    glPushMatrix()
    glScalef(.1, .25, .1)
    drawSphere()
    glPopMatrix()

    # left calf
    glColor3d(0, 255, 255)
    glPushMatrix()
    glTranslatef(0, -.45, 0)
    glRotate(45 * (np.sin(t) * .6), 1, 0, 0)

    glPushMatrix()
    glScalef(.1, .25, .1)
    drawSphere()
    glPopMatrix()

    # left foot
    glColor3d(0, 255, 0)
    glPushMatrix()
    glTranslatef(0, -.3, .05)

    glPushMatrix()
    glScalef(.1, .1, .15)
    drawSphere()
    glPopMatrix()

    glPopMatrix()
    glPopMatrix()
    glPopMatrix()

    # right thigh
    glColor3d(0, 0, 255)
    glPushMatrix()
    glTranslatef(.2, -.5, 0)
    glRotate(45 * np.sin(-t), 1, 0, 0)

    glPushMatrix()
    glScalef(.1, .25, .1)
    drawSphere()
    glPopMatrix()

    # right calf
    glColor3d(0, 255, 255)
    glPushMatrix()
    glTranslatef(0, -.45, 0)
    glRotate(45 * (np.sin(-t) * .6), 1, 0, 0)

    glPushMatrix()
    glScalef(.1, .25, .1)
    drawSphere()
    glPopMatrix()

    # right foot
    glColor3d(0, 255, 0)
    glPushMatrix()
    glTranslatef(0, -.3, .05)

    glPushMatrix()
    glScalef(.1, .1, .15)
    drawSphere()
    glPopMatrix()

    glPopMatrix()
    glPopMatrix()
    glPopMatrix()

    # left shoulder
    glColor3d(255, 0, 255)
    glPushMatrix()
    glTranslatef(-.25, .3, 0)
    glRotatef(-25, 0, 0, 1)
    glRotate(45 * np.sin(-t), 1, 0, 0)

    glPushMatrix()
    glScalef(.07, .2, .07)
    drawSphere()
    glPopMatrix()

    # left forearm
    glColor3d(100, 100, 100)
    glPushMatrix()
    glTranslatef(0, -.35, 0)

    glPushMatrix()
    glScalef(.07, .2, .07)
    drawSphere()
    glPopMatrix()

    glPopMatrix()
    glPopMatrix()

    # right shoulder
    glColor3d(255, 0, 255)
    glPushMatrix()
    glTranslatef(.25, .3, 0)
    glRotatef(25, 0, 0, 1)
    glRotate(45 * np.sin(t), 1, 0, 0)

    glPushMatrix()
    glScalef(.07, .2, .07)
    drawSphere()
    glPopMatrix()

    # right forearm
    glColor3d(100, 100, 100)
    glPushMatrix()
    glTranslatef(0, -.35, 0)

    glPushMatrix()
    glScalef(.07, .2, .07)
    drawSphere()
    glPopMatrix()

    glPopMatrix()
    glPopMatrix()

    # head
    glColor3d(1, 0, 0)
    glPushMatrix()
    glTranslatef(0, .7, 0)
    glScalef(.2, .2, .2)
    drawSphere()
    glPopMatrix()


def openBvh(path):
    global bvh
    with open(path, 'r') as f:
        bvh = Bvh.Tree(f.read())


def drawLine(x, y, z):
    glBegin(GL_LINES)
    glVertex3f(0, 0, 0)
    glVertex3f(x, y, z)
    glEnd()


def extract_position(isRoot, now_node, frame):
    global gOffset
    now_cor = [now_node.offsets[0], now_node.offsets[1], now_node.offsets[2]]
    now_rotate = [[], [], []]
    
    if isRoot:
        for idx, channel in enumerate(now_node.channels):
            angle = float(bvh.frames[frame][idx])

            if channel == 'Xposition':
                now_cor[0] += angle
            elif channel == 'Yposition':
                now_cor[1] += angle
            elif channel == 'Zposition':
                now_cor[2] += angle
            else:
                if channel == 'Xrotation':
                    now_rotate[idx - 3] = [angle, 1, 0, 0]
                elif channel == 'Yrotation':
                    now_rotate[idx - 3] = [angle, 0, 1, 0]
                elif channel == 'Zrotation':
                    now_rotate[idx - 3] = [angle, 0, 0, 1]
    else:
        for idx, channel in enumerate(now_node.channels):
            angle = float(bvh.frames[frame][gOffset])

            if channel == 'Xrotation':
                now_rotate[idx] = [angle, 1, 0, 0]
            elif channel == 'Yrotation':
                now_rotate[idx] = [angle, 0, 1, 0]
            elif channel == 'Zrotation':
                now_rotate[idx] = [angle, 0, 0, 1]

            gOffset += 1

    return now_cor, now_rotate


def drawBvh(isRoot, now_node, frame):
    global bvh, gOffset, gMove
    glColor3d(255, 255, 0)

    if gMove:
        now_cor, now_rotate = extract_position(isRoot, now_node, frame)
        if not isRoot:
            drawLine(now_cor[0], now_cor[1], now_cor[2])

        glPushMatrix()
        glTranslatef(now_cor[0], now_cor[1], now_cor[2])

        for rotate in now_rotate:
            if len(rotate) == 0: continue
            glRotate(rotate[0], rotate[1], rotate[2], rotate[3])

        for child in now_node.children:
            drawBvh(False, child, frame)
            glPopMatrix()
    else:
        drawLine(now_node.offsets[0], now_node.offsets[1], now_node.offsets[2])
        glPushMatrix()
        glTranslatef(now_node.offsets[0], now_node.offsets[1], now_node.offsets[2])

        for child in now_node.children:
            drawBvh(False, child, frame)
            glPopMatrix()


def onMouseButton(window, button, state, mods):
    global isLeft, isRight, originX, originY
    print(f"button : {button}, state : {state}")

    originX, originY = glfw.get_cursor_pos(window)
    if button == glfw.MOUSE_BUTTON_LEFT:
        if state == 1:
            isLeft = True
            print(f"clicked left mouse button at ({originX}, {originY})")
        elif state == 0:
            isLeft = False
            print(f"off left mouse click at ({originX}, {originY})")
    if button == glfw.MOUSE_BUTTON_RIGHT:
        if state == 1:
            isRight = True
            print(f"clicked right mouse button at ({originX}, {originY})")
        elif state == 0:
            isRight = False
            print(f"off right mouse click at ({originX}, {originY})")


def onMouseDrag(window, xPos, yPos):
    global isLeft, isRight, gAzimuth, gElevation, originX, originY, moveX, moveY
    if isLeft:
        gElevation += (xPos - originX) * 0.1
        gAzimuth += (yPos - originY) * 0.1
        print(f"gAzimuth : {gAzimuth}, gElevation : {gElevation}")

    if isRight:
        moveX += (xPos - originX) * 0.01
        moveY += (yPos - originY) * 0.01
        print(f"moveX : {moveX}, moveY : {moveY}")

    originX = xPos
    originY = yPos


def onMouseScroll(window, xOffset, yOffset):
    global gZoom
    gZoom -= yOffset

    print(f"gZoom : {gZoom}")


def onKeyButton(window, key, scancode, action, mods):
    global gMove, gMotionStartTime
    if action == glfw.PRESS or action == glfw.REPEAT:
        if key == glfw.KEY_SPACE:
            gMove = not gMove
            if gMove:
                gMotionStartTime = glfw.get_time()


def onDropFile(window, path):
    file_name = path[0].split('\\')
    file_name = file_name[- 1]
    print(f"\n************* OPEN NEW FILE : {file_name} *************")
    file_name = file_name[-4:].lower()

    if file_name == '.bvh':
        openBvh(path[0])


def render():
    global width, height, gAzimuth, gElevation, moveX, moveY, gZoom, gMotionStartTime, bvh, gFrame, gOffset
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    glLoadIdentity()
    gluPerspective(45, float(width) / float(height), 0.1, 100)  # 원금감에 관련된 함수

    gluLookAt(0, 0, gZoom, 0, 0, 0, 0, 1, 0)

    # panning
    glTranslatef(moveX, moveY, 0)

    # orbit
    glRotatef(gAzimuth, 1, 0, 0)
    glRotatef(gElevation, 0, 1, 0)

    drawFrame(100)
    drawGrid(100)

    glScalef(0.1, 0.1, 0.1)

    # drawSphere()
    # drawCube()
    # drawModel()
    if bvh is None:
        return

    frame = int(int((glfw.get_time() - gMotionStartTime) / bvh.frame_time) % bvh.num_frames)
    gOffset = 6
    drawBvh(True, bvh.root.children[0], frame)
    glPopMatrix()


def main():
    global width, height, gMotionStartTime, gFrame, gMove
    # initialize glfw
    if not glfw.init():
        return

    window = glfw.create_window(width, height, "My OpenGL window", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_mouse_button_callback(window, onMouseButton)
    glfw.set_cursor_pos_callback(window, onMouseDrag)
    glfw.set_scroll_callback(window, onMouseScroll)
    glfw.set_key_callback(window, onKeyButton)
    glfw.set_drop_callback(window, onDropFile)

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        render()
        gFrame += 1
        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()
