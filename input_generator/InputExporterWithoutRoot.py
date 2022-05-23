import numpy as np
import os
import math

class Data:
    def __init__(self, file_name):
        self.file_name = file_name
        self.file_stream = open(str(os.getcwd()) + '/Export/' + file_name + '.txt' , 'w')
        self.value_list = []

    def feed(self, value):
        self.value_list.append(str(value))

    def store(self):
        self.file_stream.write(" ".join(self.value_list))
        self.file_stream.write('\n')
        self.value_list = []

    def finish(self):
        self.file_stream.close()


class State:
    def __init__(self, frame_idx):
        global matrixes
        self.frame_idx = frame_idx
        self.bone_num = len(matrixes[frame_idx])
        self.root = self.getRootWorldMatrix()
        self.posture = self.getPostures()
        self.up = self.getUp()
        self.velocities = self.getVelocites()

    def getRootWorldMatrix(self):
        # root의 world matrix
        global matrixes
        return matrixes[self.frame_idx][0]

    def getVelocites(self):
        velocities = []
        for i in range(self.bone_num):
            velocities.append(self.getBoneVelocity(i))

        return velocities

    def getPostures(self):
        global matrixes
        postures = []
        for i in range(self.bone_num):
            postures.append(matrixes[self.frame_idx][i])
        return postures

    def getUp(self):
        up = np.array([self.posture[0][1], self.posture[1][1], self.posture[2][1]])
        normalized_up = np.round((up / np.sqrt(np.sum(up**2))).tolist(),3)
        return normalized_up

    def getBoneVelocity(self, bone_idx):
        global gBVHFrameTime, gGlobalOffset

        # first frame_idx : 0
        frame_idx = self.frame_idx
        if self.frame_idx == 0:
            frame_idx = 0 + 1
        previous_frame_position = gGlobalOffset[frame_idx-1][bone_idx]
        current_frame_position = gGlobalOffset[frame_idx][bone_idx]
        framerate = round(1.0/gBVHFrameTime)
        return ((np.array(previous_frame_position) - np.array(current_frame_position)) * framerate).tolist()


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

    def get_channelstr(self):
        return self._channels_str

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

    def get_start(self):
        return self._channels_start

    def get_is_end(self):
        return self._is_end

    def add_child(self, parm):
        self._children.append(parm)


############################################################################
# Read BVH files
def fetchSkeletonFromBVH(path):
    global gBVHTree, gBVHIndex, gBVHVertex
    global gBVHMotion, gBVHFrameTime, gBVHFrameNum
    global stack, joint_names, gGlobalOffset

    # Read *.bvh file
    file = open(path, "r")      # Open file buffer for reading obj file.
    file.seek(0)                # Change file pointer to first.
    bvh = file.readlines()      # Read and push file's each lines into list.
    bvh = list(map(lambda s: s.strip(), bvh))
    file.close()                # Close file buffer.

    
    def readHierarchy(_i):
        global glChannel, glName, iarr, joints, stack, joint_names, matrixes

        glChannel = []
        glName = []
        joints = []

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
                current.set_channels_str(channel_sum, bvh[__i+3].split()[1:])   # channels

                channel_sum += int(bvh[__i+3].split()[1])
                __i += 4

                # Push to joint names array
                joint_names.append(sentence[1])

                # Push to index array
                node_count += 1
                if node_count > 1:
                    iarr += [parent.get_id(), current.get_id()]
                # Add to parents stack, parent's children stack and change parent <- itself
                joints.append(current)
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

                glName.append('End')
                glChannel.append(current.get_start())

                # Add to parent's children stack
                joints.append(current)
                parent.add_child(current)

                while bvh[__i].split()[0] == '}':
                    __i += 1
                    if len(stack) == 0:
                        is_end = True
                        break
                    parent = stack.pop()
                stack.append(parent)
        return __i, joint_names, node_count, tree.get_children()[0], iarr


    def readMotion(_i):
        global frame_number, motions, matrixes, joints, joint_names, gGlobalOffset
        
        __i = _i
        motions = []
        frame_number = int(bvh[__i].split()[1])      # Frame number
        frame_time = float(bvh[__i+1].split()[2])    # Frame time
        __i += 2

        total = _i + frame_number + 2
        current_frame = 0
        joint_num = 0
        for i in range(len(iarr)):
            if(iarr[i] > joint_num):
                joint_num = iarr[i]

        matrixes = [[np.identity(4) for x in range(joint_num+1)] for y in range(frame_number)]
        gGlobalOffset = [[0 for x in range(joint_num+1)] for y in range(frame_number)]
        p = np.asmatrix([[0],[0],[0],[1]])

        while __i < total:
            each_motion = []
            frame = bvh[__i].split()
            
            for entry in frame:
                each_motion.append(float(entry))
            motions.append(each_motion)
            
            N = idx = idx_ = 0
            #idx = current index
            #idx_ = parent's index

            while N < len(each_motion) :
                idx_ = findParent(idx)
                while(joints[idx].get_name() == ''):
                    T = np.identity(4)
                    T[:3,3] = joints[idx].get_offset()
                    matrixes[current_frame][idx] = matrixes[current_frame][idx_] @ T
                    coordinate = matrixes[current_frame][idx] @ p
                    gGlobalOffset[current_frame][idx] = coordinate[0:3]
                    idx += 1                
                idx_ = findParent(idx)
                M = matrixes[current_frame][idx_]

                R = np.identity(4)
                T = np.identity(4)

                if(idx == 0):
                    channel = each_motion[0:6]
                else:
                    channel = each_motion[N:N+3]
                
                for num in range(N, N + len(channel), 1):
                    if   joints[idx].get_channelstr()[num - N].upper() == "XROTATION":
                        R = R @ rotate(0, each_motion[num])
                    elif joints[idx].get_channelstr()[num - N].upper() == "YROTATION":
                        R = R @ rotate(1, each_motion[num])
                    elif joints[idx].get_channelstr()[num - N].upper() == "ZROTATION":
                        R = R @ rotate(2, each_motion[num])

                ####Trajectory Editor####
                #LKnee : 1    RKnee : 6
                #LShol : 17   Rshol : 22
                if(idx == 17):
                    R = R @ rotate(2, 20)
                #########################

                if(idx == 0) :
                    T[:3, 3] = np.array([each_motion[0],each_motion[1],each_motion[2]])
                    N += 3
                
                else:
                    T[:3,3] = joints[idx].get_offset()
                A = T@R

                matrixes[current_frame][idx] = M @ A

                p = np.asmatrix([[0],[0],[0],[1]])
                coordinate = matrixes[current_frame][idx] @ p
                gGlobalOffset[current_frame][idx] = coordinate[0:3]
                N += 3
                idx += 1

            idx_ = findParent(joint_num)
            M = matrixes[current_frame][idx_]
            T = np.identity(4)
            T[:3,3] = joints[idx].get_offset()
            matrixes[current_frame][joint_num] = matrixes[current_frame][idx_] @ T
            coordinate = matrixes[current_frame][joint_num] @ p
            gGlobalOffset[current_frame][joint_num] = coordinate[0:3]

            current_frame += 1
            __i += 1

        return __i, motions, frame_number, frame_time


    bvh_tree = None         # root of bvh hierarchical tree
    bvh_motions = None      # array of each frame's motion
    bvh_all_node_num = 0
    bvh_joint_names = []
    iarr = []               # index array
    bvh_frame_number = bvh_frame_time = None

    i = 0
    length = len(bvh)

    while i < length:
        if bvh[i].upper() == "MOTION":
            point,\
            bvh_motions, \
            bvh_frame_number,\
            bvh_frame_time = readMotion(i+1)
            i = point

        elif bvh[i].upper() == "HIERARCHY":
            point, \
            bvh_joint_names, \
            bvh_all_node_num, \
            bvh_tree, \
            iarr = readHierarchy(i+1)
            i = point

    # Calculate t-pose model.
    gBVHIndex = iarr
    gBVHVertex = np.zeros(bvh_all_node_num*3, 'float32')
    gBVHFrameTime = bvh_frame_time
    gBVHFrameNum = bvh_frame_number

    gBVHTree = bvh_tree
    gBVHMotion = bvh_motions

############################################################################

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

def findParent(idx):
    global iarr
    for i in range(1,len(iarr),2):
        if(idx == iarr[i]):
            return iarr[i-1]
    return 0

# generate data
def getAngle(v):
    #https://github.com/Unity-Technologies/UnityCsReference/blob/master/Runtime/Export/Math/Vector3.cs#L309
    v = v/np.linalg.norm(v)
    vForward = np.array([0,0,1])
    vUp = np.array([0,1,0])

    angle = math.degrees(np.dot(vForward,v))
    x = vForward[1]*v[2] - vForward[2]*v[1]
    y = vForward[2]*v[0] - vForward[0]*v[2]
    z = vForward[0]*v[1] - vForward[1]*v[0]
    vv = np.array([x,y,z])

    signed = 1
    if(np.dot(vv,vUp)<0):
        signed = -1
    
    return signed*angle

def exportData(states):
    global matrixes, gGlobalOffset, frame_number, gBVHFrameTime, filename
    x = Data(filename[:-4] + "_i")
    y = Data(filename[:-4] + "_o")
    for i in range(1, len(states)-12): # Except the case that occurs overflow & underflow
        Previous = states[i-1]
        Current = states[i]
        Next = states[i+1]

        def multiplyVector(m, d):
            '''
            # Get Relative postion
            m_ = np.linalg.inv(np.array(m))
            d_ = np.array(d)
            return [d_[0] * m_[0][0] + d_[1] * m_[0][1] + d_[2] * m_[0][2],
                    d_[0] * m_[1][0] + d_[1] * m_[1][1] + d_[2] * m_[1][2],
                    d_[0] * m_[2][0] + d_[1] * m_[2][1] + d_[2] * m_[2][2]]
            '''
            return d

        def clamp(num, low, hi):
            if (num < low):
                return low
            elif (num > hi):
                return hi
            else:
                return num

        # INPUT ---
        for j in range(12):                # Trajectory (of root of each frame)
            cPoints = states[i + j].getPostures()[0].tolist()
            position = multiplyVector(Current.getRootWorldMatrix(), [cPoints[0][3], cPoints[1][3], cPoints[2][3]])
            direction = multiplyVector(Current.getRootWorldMatrix(), [cPoints[0][2], cPoints[1][2], cPoints[2][2]])
            velocity = multiplyVector(Current.getRootWorldMatrix(),
                                      [Current.getVelocites()[j][0][0], Current.getVelocites()[j][1][0],
                                       Current.getVelocites()[j][2][0]])
            speed = 0
            for k in range(6):
                index_ = clamp(i + j + k, 0, frame_number - 2)
                a = np.array([gGlobalOffset[index_][0][0], gGlobalOffset[index_][0][2]])
                b = np.array([gGlobalOffset[index_ + 1][0][0], gGlobalOffset[index_ + 1][0][2]])
                length = np.linalg.norm(a - b)
                speed = speed + length

            x.feed(f"{position[0]:.5f}")   # position x of Trajectory j+1
            x.feed(f"{position[2]:.5f}")   # position z of Trajectory j+1
            x.feed(f"{direction[0]:.5f}")  # direction x of Trajectory j+1
            x.feed(f"{direction[2]:.5f}")  # direction z of Trajectory j+1
            x.feed(velocity[0])     # velocity x of Trajectory j+1
            x.feed(velocity[2])    # velocity z of Trajectory j+1
            x.feed(f"{speed:.5f}")         # speed of Trajectory j+1

        for j in range(len(matrixes[0])):  # bone
            pPosture = Previous.getPostures()[j]
            position = multiplyVector(Current.getRootWorldMatrix(), [pPosture[0][3], pPosture[1][3], pPosture[2][3]])
            forward = multiplyVector(Current.getRootWorldMatrix(), [pPosture[0][2], pPosture[1][2], pPosture[2][2]])
            up = multiplyVector(Current.getRootWorldMatrix(), [pPosture[0][1], pPosture[1][1], pPosture[2][1]])
            velocity = multiplyVector(Current.getRootWorldMatrix(),
                                      [Previous.getVelocites()[j][0][0], Previous.getVelocites()[j][1][0],
                                       Previous.getVelocites()[j][2][0]])
            if(j == 0):
                x.feed(f"{position[0]:.5f}")   # position x of bone j+1
                x.feed(f"{position[1]:.5f}")   # position y of bone j+1
                x.feed(f"{position[2]:.5f}")   # position z of bone j+1
            else:
                x.feed(f"{position[0] - Current.getRootWorldMatrix()[0][3]:.5f}")   # position x of bone j+1
                x.feed(f"{position[1] - Current.getRootWorldMatrix()[1][3]:.5f}")   # position y of bone j+1
                x.feed(f"{position[2] - Current.getRootWorldMatrix()[2][3]:.5f}")   # position z of bone j+1
            x.feed(f"{forward[0]:.5f}")    # forward x of bone j+1
            x.feed(f"{forward[1]:.5f}")    # forward y of bone j+1
            x.feed(f"{forward[2]:.5f}")    # forward z of bone j+1
            x.feed(f"{up[0]:.5f}")         # up x of bone j+1
            x.feed(f"{up[1]:.5f}")         # up y of bone j+1
            x.feed(f"{up[2]:.5f}")         # up z of bone j+1
            x.feed(velocity[0])     # velocity x of bone j+1
            x.feed(velocity[1])     # velocity y of bone j+1
            x.feed(velocity[2])     # velocity z of bone j+1
        x.store()
        # OUTPUT ---
        for j in range(6, 12):  # Trajectory (of root of each frame)
            nPoints = np.round(states[i + j + 1].getPostures()[0].tolist(), 3)
            position = multiplyVector(Next.getRootWorldMatrix(), [nPoints[0][3], nPoints[1][3], nPoints[2][3]])
            direction = multiplyVector(Next.getRootWorldMatrix(), [nPoints[0][2], nPoints[1][2], nPoints[2][2]])
            velocity = multiplyVector(Next.getRootWorldMatrix(),
                                      [Next.getVelocites()[j][0][0], Next.getVelocites()[j][1][0], Next.getVelocites()[j][2][0]])
            y.feed(f"{position[0]:.5f}")   # position x of Trajectory j+1
            y.feed(f"{position[2]:.5f}")   # position z of Trajectory j+1
            y.feed(f"{direction[0]:.5f}")  # direction x of Trajectory j+1
            y.feed(f"{direction[2]:.5f}")  # direction z of Trajectory j+1
            y.feed(velocity[0])     # velocity x of Trajectory j+1
            y.feed(velocity[2])     # velocity z of Trajectory j+1

        for j in range(len(matrixes[0])):  # bone
            cPosture = np.round(Current.getPostures()[j].tolist(), 3)
            position = multiplyVector(Next.getRootWorldMatrix(), [cPosture[0][3], cPosture[1][3], cPosture[2][3]])
            forward = multiplyVector(Next.getRootWorldMatrix(), [cPosture[0][2], cPosture[1][2], cPosture[2][2]])
            up = multiplyVector(Next.getRootWorldMatrix(), [cPosture[0][1], cPosture[1][1], cPosture[2][1]])
            velocity = multiplyVector(Next.getRootWorldMatrix(),
                                      [Previous.getVelocites()[j][0][0], Previous.getVelocites()[j][1][0],
                                       Previous.getVelocites()[j][2][0]])
            if(j == 0 ):
                y.feed(f"{position[0]:.5f}")  # position x of bone j+1
                y.feed(f"{position[1]:.5f}")  # position y of bone j+1
                y.feed(f"{position[2]:.5f}")  # position z of bone j+1
            else: 
                y.feed(f"{position[0] - Next.getRootWorldMatrix()[0][3]:.5f}")  # position x of bone j+1
                y.feed(f"{position[1] - Next.getRootWorldMatrix()[1][3]:.5f}")  # position y of bone j+1
                y.feed(f"{position[2] - Next.getRootWorldMatrix()[2][3]:.5f}")  # position z of bone j+1
            y.feed(f"{forward[0]:.5f}")   # forward x of bone j+1
            y.feed(f"{forward[1]:.5f}")   # forward y of bone j+1
            y.feed(f"{forward[2]:.5f}")   # forward z of bone j+1
            y.feed(f"{up[0]:.5f}")        # up x of bone j+1
            y.feed(f"{up[1]:.5f}")        # up y of bone j+1
            y.feed(f"{up[2]:.5f}")        # up z of bone j+1
            y.feed(velocity[0])    # velocity x of bone j+1
            y.feed(velocity[1])    # velocity y of bone j+1
            y.feed(velocity[2])    # velocity z of bone j+1
        cRoot = np.linalg.inv(np.array(matrixes[i][0]))
        nRoot = np.array(matrixes[i + 1][0])
        delta = np.matmul(cRoot, nRoot)
        angle = getAngle(np.array([delta[0][2], delta[1][2], delta[2][2]]))
        delta_ = np.array([delta[0][3], delta[1][3], delta[2][3]]) * gBVHFrameTime
        y.feed(f"{delta_[0]:.5f}")        # root translation x
        y.feed(f"{delta_[1]:.5f}")        # root rotation y
        y.feed(f"{delta_[2]:.5f}")        # root translation z
        y.store()
    x.finish()
    y.finish()


def main():
    global frame_number, gGlobalOffset, filename
    
    ########set file name here########
    path_dir = str(os.getcwd()) + '/MotionCapture/lafan'
    file_list = os.listdir(path_dir)
    ##################################
    for num in range(len(file_list)):
        print("Exporting file...  " + str(num) + "/" + str(len(file_list)))
        print("Fetching BVH file... ")
        filename = file_list[num]
        fetchSkeletonFromBVH(path_dir + "/"+ filename)
        states = []
        for i in range(frame_number):
            new_state = State(i)
            states.append(new_state)
        print("Generating input/output file... ")
        exportData(states)
    
if __name__ == "__main__":
   main()
