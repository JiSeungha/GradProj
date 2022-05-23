# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from MANN import MANN

num_joints = 26
num_styles = 0
dataPath = '../data/human_data'
num_experts = 8
index_gatingIn = [93, 94, 95, # hip Velocity X,Y,Z
                  105, 106, 107, # left upper leg Velocity X,Y,Z
                  117, 118, 119, # left lower leg Velocity X,Y,Z
                  165, 166, 167, # right upper leg Velocity X,Y,Z
                  177, 178, 179, # left right leg Velocity X,Y,Z
                  309, 310, 311, # left upper arm Velocity X,Y,Z
                  321, 322, 323, # left lower arm Velocity X,Y,Z
                  369, 370, 371, # right upper arm Velocity X,Y,Z
                  381, 382, 383, # right lower arm Velocity X,Y,Z
                  255, 226, 227, # spine Velocity X,Y,Z
                  273, 274, 275, # head Velocity X,Y,Z
                  48,  # Trajectory7 Speed
                  ]

def main():
    rng = np.random.RandomState(23456)
    sess = tf.compat.v1.Session()
    mann = MANN(num_joints, num_styles,
                rng, sess,
                dataPath, num_experts,
                gating_hidden_size=32, hidden_size=512,
                gating_index=index_gatingIn,
                batch_size=32, epoch=50, Te=10, Tmult=2,
                init_learning_rate=0.0001, init_weightDecay=0.0025, init_keep_prob=0.7)

    mann.build_model()
    mann.train()


if __name__ == '__main__':
    main()