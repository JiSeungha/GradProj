import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw

# global variables
gLMousePress = False
gRMousePress = False
gPreMousePos = [0, 0]

gRotateAngles = [-0, 0]
gMoveDistance = [0, 0]          
gZoomScale = -5.0

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


def drawPerson():
    glColor3f(255,255,255)
    t = 2*glfw.get_time()
    
    # translate body
    glPushMatrix()
    glTranslatef(0,1.25 + 0.03*np.sin(t), 0)
    
    # draw body
    glPushMatrix()
    glScalef(.2,.4,.2)
    drawSphere()
    glPopMatrix()

    glColor3ub(0,255,0)
    # rotate&translate left thigh
    glPushMatrix()
    glTranslatef(-0.1,-.2,0)
    glRotatef(45*(-0.5+np.sin(t)), 1, 0, 0)
    glTranslatef(0,-.25,0)

    # draw left thigh
    glPushMatrix()
    glScalef(.1,.25,.1)
    drawSphere()
    glPopMatrix()

    # rotate&translate left calf
    glPushMatrix()
    glTranslatef(0,-.25,0)
    glRotatef(45-25*np.sin(t), 1, 0, 0)
    glTranslatef(0,-.25,0)

    # draw left calf
    glPushMatrix()
    glScalef(.08,.25,.08)
    drawSphere()
    glPopMatrix()

    # translate left foot
    glPushMatrix()
    glTranslatef(0,-.3,.125)
    glScalef(.1,.05,.2)
    drawSphere()
    glPopMatrix()
    glPopMatrix()
    glPopMatrix()

    # rotate&translate right thigh
    glPushMatrix()
    glTranslatef(0.1,-.2,0)
    glRotatef(45*(-0.5+np.sin(np.pi+t)), 1, 0, 0)
    glTranslatef(0,-.25,0)

    # draw right thigh
    glPushMatrix()
    glScalef(.1,.25,.1)
    drawSphere()
    glPopMatrix()

    # rotate&translate right calf
    glPushMatrix()
    glTranslatef(0,-.25,0)
    glRotatef(45-25*np.sin(np.pi+t), 1, 0, 0)
    glTranslatef(0,-.25,0)

    # draw right calf
    glPushMatrix()
    glScalef(.08,.25,.08)
    drawSphere()
    glPopMatrix()

    # translate right foot
    glPushMatrix()
    glTranslatef(0,-.3,.125)
    glScalef(.1,.05,.2)
    drawSphere()
    glPopMatrix()
    glPopMatrix()
    glPopMatrix()

    glColor3ub(255,255,0)
    # rotate&translate left upper arm
    glPushMatrix()
    glTranslatef(-0.25,.35,0)
    glRotatef(10-25*np.sin(t), 1, 0, 0)
    glTranslatef(0,-.25,0)

    # draw left upper arm
    glPushMatrix()
    glScalef(.07,.2,.07)
    drawSphere()
    glPopMatrix()

    # rotate&translate left lower arm
    glPushMatrix()
    glTranslatef(0,-.2,0)
    glRotatef(-45-10*np.sin(t), 1, 0, 0)
    glTranslatef(0,-.2,0)

    # draw left lower arm
    glPushMatrix()
    glScalef(.07,.2,.07)
    drawSphere()
    glPopMatrix()

    # translate left hand
    glPushMatrix()
    glTranslatef(0,-.25,0)
    glScalef(.08,.08,.08)
    drawSphere()
    glPopMatrix()
    glPopMatrix()
    glPopMatrix()

    # rotate&translate right upper arm
    glPushMatrix()
    glTranslatef(0.25,.35,0)
    glRotatef(10-25*np.sin(np.pi+t), 1, 0, 0)
    glTranslatef(0,-.25,0)

    # draw right upper arm
    glPushMatrix()
    glScalef(.07,.2,.07)
    drawSphere()
    glPopMatrix()

    # rotate&translate right lower arm
    glPushMatrix()
    glTranslatef(0,-.2,0)
    glRotatef(-45-10*np.sin(np.pi+t), 1, 0, 0)
    glTranslatef(0,-.2,0)

    # draw right lower arm
    glPushMatrix()
    glScalef(.07,.2,.07)
    drawSphere()
    glPopMatrix()

    # translate right hand
    glPushMatrix()
    glTranslatef(0,-.25,0)
    glScalef(.08,.08,.08)
    drawSphere()
    glPopMatrix()
    glPopMatrix()
    glPopMatrix()

    glColor3ub(255,0,0)
    # translate&draw head
    glPushMatrix()
    glTranslatef(0,.5,0)
    glScalef(.15,.15,.15)
    drawSphere()
    glPopMatrix()
    glPopMatrix()
    

def drawGrid(len):
    glDisable(GL_LIGHTING)

    # Start drawing lines.
    glBegin(GL_LINES)
    glColor3d(100, 100, 100)

    # x-grid
    for i in range(1, len):
        glVertex3d(i / 2, 0, 0)
        glVertex3d(i / 2, 0, len)
        glVertex3d(i / 2, 0, 0)
        glVertex3d(i / 2, 0, -len)

        glVertex3d(-i / 2, 0, 0)
        glVertex3d(-i / 2, 0, len)
        glVertex3d(-i / 2, 0, 0)
        glVertex3d(-i / 2, 0, -len)

    # z-grid
    for i in range(1, len):
        glVertex3d(0, 0, i / 2)
        glVertex3d(len, 0, i / 2)
        glVertex3d(0, 0, i / 2)
        glVertex3d(-len, 0, i / 2)

        glVertex3d(0, 0, -i / 2)
        glVertex3d(len, 0, -i / 2)
        glVertex3d(0, 0, -i / 2)
        glVertex3d(-len, 0, -i / 2)

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


def button_callback(window, button, state, mods):
    global gLMousePress, gRMousePress, gPreMousePos

    if state == glfw.PRESS:
        gPreMousePos = glfw.get_cursor_pos(window)
        if button == glfw.MOUSE_BUTTON_LEFT:
            gLMousePress = True
            #print('left click : (%d, %d)' % glfw.get_cursor_pos(window))

        elif button == glfw.MOUSE_BUTTON_RIGHT:
            gRMousePress = True
            #print('right click : (%d, %d)' % glfw.get_cursor_pos(window))

    elif state == glfw.RELEASE:
        gLMousePress = False
        gRMousePress = False
        #print('release : (%d, %d)' % glfw.get_cursor_pos(window))


def drag_callback(window, x, y):
    global gRotateAngles, gMoveDistance, gLMousePress, gRMousePress, gPreMousePos

    if gLMousePress:
        gRotateAngles[0] += (x - gPreMousePos[0]) / 10
        gRotateAngles[1] += (y - gPreMousePos[1]) / 10
        gPreMousePos = [x, y]

    elif gRMousePress:
        gMoveDistance[0] += (x - gPreMousePos[0]) / 100
        gMoveDistance[1] += (y - gPreMousePos[1]) / 100
        gPreMousePos = [x, y]

def scroll_callback(window, xoffset, yoffset):
    global gZoomScale, gRotateAngles
    max_offset = xoffset if np.abs(xoffset) > np.abs(yoffset) else yoffset

    # Zoom in
    if max_offset > 0:
        gZoomScale += .1
        # Limit zoom in scale when seeing origin(0,0) in world space.
        if gMoveDistance == [0, 0] and gZoomScale >= -1:
            gZoomScale = -10/9
            print("**Limited zoom in scale!!**")
    # Zoom out
    else:
        gZoomScale -= .1


def render():
    global gRotateAngles, gMoveDistance
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glEnable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1, 0.1, 100)
    gluLookAt(0,0,5,0,0,0,0,1,0)
    #Rotate and Translate

    glTranslatef(gMoveDistance[0], gMoveDistance[1], gZoomScale)    
    glRotatef(gRotateAngles[1], 1, 0, 0)
    glRotatef(gRotateAngles[0], 0, 1, 0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    drawFrame(100)
    drawGrid(100)
    drawPerson()

def main():
    # initialize glfw
    if not glfw.init():
        return

    window = glfw.create_window(800, 600, "My OpenGL window", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.set_cursor_pos_callback(window, drag_callback)
    glfw.set_mouse_button_callback(window, button_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    
    glfw.make_context_current(window)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        render()
        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()
