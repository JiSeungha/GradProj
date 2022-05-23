# 참고
# https://github.com/20tab/bvh-python/blob/master/bvh.py

import re

class Node:
    def __init__(self, name=""):
        self.name = name
        self.offsets = []
        self.channels = []
        self.children = []
        self.parent = None

    def set_value(self, values):
        if values[0] == 'OFFSET':
            self.offsets = [float(values[1]), float(values[2]), float(values[3])]
        elif values[0] == 'CHANNELS':
            self.channels = values[2:]

    def add_child(self, item):
        item.parent = self
        self.children.append(item)


class Tree:
    def __init__(self, file=None):
        self.file = file
        self.root = Node()
        self.frames = []
        self.num_frames = 0
        self.frame_time = 0.0
        self.read_file()

    def read_file(self):
        split_list = []
        for line in self.file.split('\n'):
            split_list.append(re.split('\\s+', line.strip()))

        is_hierarchy = False
        node_stack = [self.root]
        node = None
        for item in split_list:
            key = item[0]
            if key == 'HIERARCHY':
                is_hierarchy = True
                continue
            if key == 'Frames:':
                self.num_frames = int(item[-1])
            if item[0] == 'Frame' and item[1] == 'Time:':
                is_hierarchy = False
                self.frame_time = float(item[-1])
                continue

            if is_hierarchy:
                if key == 'ROOT' or key == 'JOINT' or key == 'End':
                    node = Node(item[1])
                elif key == '{':
                    node_stack[-1].add_child(node)
                    node_stack.append(node)
                elif key == '}':
                    node = node_stack.pop()
                else:
                    node_stack[-1].set_value(item)
            else:
                self.frames.append(item)
