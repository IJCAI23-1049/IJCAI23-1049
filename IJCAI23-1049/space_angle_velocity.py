import os
import yaml
import math
import numpy as np
import torch
import torch.nn.functional as F
import data_utils


def angle(v1):
    x = torch.FloatTensor([1, 0, 0])
    y = torch.FloatTensor([0, 1, 0])
    z = torch.FloatTensor([0, 0, 1])

    L = torch.sqrt(v1.dot(v1))
    Lx = torch.sqrt(x.dot(x))
    Ly = torch.sqrt(y.dot(y))
    Lz = torch.sqrt(z.dot(z))

    cos_angle_x = v1.dot(x) / (L * Lx)
    cos_angle_y = v1.dot(y) / (L * Ly)
    cos_angle_z = v1.dot(z) / (L * Lz)

    return cos_angle_x, cos_angle_y, cos_angle_z


def space_angle(previous_one_frame, current_frame):
    space_angle = torch.FloatTensor([])
    A = current_frame - previous_one_frame
    angle_x, angle_y, angle_z = angle(A)
    one_joint_space_angle = torch.tensor([angle_x, angle_y, angle_z], dtype=torch.float32)
    space_angle = torch.cat((space_angle, one_joint_space_angle), dim=0)
    space_angle = space_angle.view(1, previous_one_frame.shape[0])

    return space_angle


def space_distance(previous_one_frame, current_frame):
    space_velicity = torch.FloatTensor([])

    dist = torch.sqrt(torch.sum((current_frame - previous_one_frame) ** 2))
    dist = torch.unsqueeze(dist, 0)
    space_velicity = torch.cat((space_velicity, dist), dim=0)
    space_velicity = space_velicity.view(1, 1)

    return space_velicity


def loc_exchange(input):
    '''b:betach size; f:frames; n:nodes_num*3'''
    bs = input.shape[0]  # batch_size
    fr = input.shape[1]  # frame_number

    input = input.reshape(bs, fr, -1, 3)
    nd = input.shape[2]  # node_num
    input = torch.tensor(input, dtype=torch.float32)

    angle_velocity = torch.FloatTensor([])
    for i in range(input.shape[0]):
        one_sequence = torch.FloatTensor([])
        for a in range(input.shape[1] - 1):
            one_frame = torch.FloatTensor([])
            for b in range(input.shape[2]):
                space_angles = space_angle(input[i, a, b], input[i, a + 1, b])
                space_velocity = space_distance(input[i, a, b], input[i, a + 1, b])
                space_angles = torch.where(torch.isnan(space_angles), torch.full_like(space_angles, 0), space_angles)
                one_frame = torch.cat([one_frame, space_angles, space_velocity], dim=1)
            one_sequence = torch.cat([one_sequence, one_frame], dim=0)
        angle_velocity = torch.cat([angle_velocity, one_sequence], dim=0)
    angle_velocity = angle_velocity.view(bs, fr - 1, nd, 4)
    return angle_velocity


def reconstruction_motion(prediction_distance, prediction_angle, current_frame, node_num):
    reconstruction_coordinate = torch.zeros([node_num, 3], dtype=torch.float32)
    # print('node_num:\n',node_num)
    for i in range(current_frame.shape[0]):
        x = current_frame[i, 0] + prediction_distance[i] * prediction_angle[i, 0]
        y = current_frame[i, 1] + prediction_distance[i] * prediction_angle[i, 1]
        z = current_frame[i, 2] + prediction_distance[i] * prediction_angle[i, 2]
        current_joint_coordinates = torch.tensor([x, y, z])
        reconstruction_coordinate[i] = current_joint_coordinates

    return reconstruction_coordinate


def mpjpe(input, target):
    return torch.mean(torch.norm(input - target, 2, 1))
