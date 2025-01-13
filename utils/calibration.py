import numpy as np
import os

def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype = np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype = np.float32)
    obj = lines[4].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype = np.float32)
    Tr_velo_to_cam = np.vstack((Tr_velo_to_cam.reshape(3, 4), np.array([0, 0, 0, 1])))
    # print(Tr_velo_to_cam)

    return { 'P2'         : P2.reshape(3, 4),
             'P3'         : P3.reshape(3, 4),
             'Tr_velo2cam': Tr_velo_to_cam }