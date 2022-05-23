import numpy as np
import os

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

    def get_start(self):
        return self._channels_start

    def get_is_end(self):
        return self._is_end

    def add_child(self, parm):
        self._children.append(parm)

# Read BVH files
def fetchSkeletonFromBVH(path):
    global gBVHTree, gBVHIndex, gBVHVertex
    global gBVHMotion, gBVHFrameTime, gBVHFrameNum

    # Read *.bvh file
    file = open(path, "r")      # Open file buffer for reading obj file.
    file.seek(0)                # Change file pointer to first.
    bvh = file.readlines()      # Read and push file's each lines into list.
    bvh = list(map(lambda s: s.strip(), bvh))
    file.close()                # Close file buffer.

    def readMotion(_i):
        global frame_number, motions
        
        __i = _i
        motions = []
        frame_number = int(bvh[__i].split()[1])      # Frame number
        frame_time = float(bvh[__i+1].split()[2])    # Frame time
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

    def readHierarchy(_i):
        global glTransOffset, glChannel, glName, iarr

        glTransOffset = []
        glChannel = []
        glName = []

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

                tmp_offset = current.get_offset().tolist()
                for i in range (3):
                    tmp_offset[i] = tmp_offset[i]
                glTransOffset.append(tmp_offset)
                glName.append(current.get_name())
                glChannel.append(current.get_start())

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
                current.set_end(bvh[__i+2].split()[1:])  # offset
                __i += 3

                # Push to index array
                node_count += 1
                iarr += [parent.get_id(), current.get_id()]

                glTransOffset.append(current.get_offset().tolist())
                glName.append('End')
                glChannel.append(current.get_start())

                # Add to parent's children stack
                parent.add_child(current)

                while bvh[__i].split()[0] == '}':
                    __i += 1
                    if len(stack) == 0:
                        is_end = True
                        break
                    parent = stack.pop()
                stack.append(parent)

        return __i, joint_names, node_count, tree.get_children()[0], iarr

    bvh_tree = None      # root of bvh hierarchical tree
    bvh_motions = None      # array of each frame's motion
    bvh_all_node_num = 0
    bvh_joint_names = []
    iarr = []            # index array
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

def trans(matrixA, Toffset, Roffset):
    global matrixes

    matrix_ = np.asmatrix(matrixA)

    T = np.identity(4)
    for i in range(3):
        Toffset[i] = Toffset[i]
    T[:3, -1] = Toffset
    
    R = np.identity(4)  
    Rx = np.array([[1, 0, 0, 0],
                   [0, round(np.cos(Roffset[1]* (np.pi / 180)),3), -round(np.sin(Roffset[1]* (np.pi / 180)),3), 0],
                   [0, round(np.sin(Roffset[1]* (np.pi / 180)),3), round(np.cos(Roffset[1]* (np.pi / 180)),3),0],
                   [0,0,0,1]])
    Ry = np.array([[round(np.cos(Roffset[2]* (np.pi / 180)),3), 0, round(np.sin(Roffset[2]* (np.pi / 180)),3),0],
                   [0, 1, 0,0],
                   [-round(np.sin(Roffset[2]* (np.pi / 180)),3), 0, round(np.cos(Roffset[2]* (np.pi / 180)),3),0],
                   [0,0,0,1]])
    Rz = np.array([[round(np.cos(Roffset[0]* (np.pi / 180)),3), -round(np.sin(Roffset[0]* (np.pi / 180)),3), 0,0],
                   [round(np.sin(Roffset[0]* (np.pi / 180)),3), round(np.cos(Roffset[0]* (np.pi / 180)),3), 0,0],
                   [0, 0, 1,0],
                   [0,0,0,1]])
    R = Rz@Rx@Ry

    a = T@R
    
    matrix_ = matrix_@a
    return matrix_.A

def getCoordinate(matrix, coordinate):
    matrix_ = np.asmatrix(matrix)
    coordinate_ = [[0],[0],[0],[1]]
    coordinate_ = np.asmatrix(coordinate_)
    matrix_ = matrix_@coordinate_
    matrix_ = matrix_.A
    result = []
    for i in range (3):
        result.append(matrix_[i][0]+coordinate[i])
    return result


def main():
    global glTransOffset, glChannel, glName, iarr, motions, frame_number, matrixes
    path_dir = str(os.getcwd()) + '/MotionCapture'
    file_list = os.listdir(path_dir)
    
    fetchSkeletonFromBVH(file_list[0]) #<- change file name here
    
    coordinates = [[0 for x in range(len(glName))] for y in range(frame_number)] 
    matrixes = [[0 for x in range(len(glName))] for y in range(frame_number)]

    for i in range(frame_number):
        for j in range(len(glName)) :
            if(j == 0) :
                root_matrix = np.identity(4)
                matrixes[i][j] = trans(root_matrix,glTransOffset[0], motions[i][3:6])
                coordinates[i][j] = getCoordinate(matrixes[i][j], motions[i][0:3])
            else :
                for k in range (1, len(iarr), 2):
                    if(iarr[k] == j):
                        matrixes[i][j] = trans(matrixes[i][iarr[k-1]], glTransOffset[j],motions[i][glChannel[j]:glChannel[j]+3])
                        coordinates[i][j] = getCoordinate(matrixes[i][iarr[k-1]], motions[i][0:3])
                        break

    return coordinates

if __name__ == "__main__":
    print(main())
