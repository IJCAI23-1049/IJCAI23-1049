import yaml
import os
import numpy as np
import h5py
import math
import torch
import torch.nn.functional as F

config = yaml.load(open('config.yml'), Loader=yaml.FullLoader)
print(config)
"""
{'batch_size': 16, 'dataset': 'H36M', 'epsilon': 0.01, 'gradient_clip': 5.0, 'hidden_size': 128, 
'in_features': 9, 'input_n': 10, 'learning_rate': 0.001, 
'node_num': 25, 'out_features': 25, 'output_n': 25, 'train_epoches': 20}
"""
# build path
base_path = ''
move_joint = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21, 22, 25, 26, 27, 29, 30])
print(len(move_joint))
train_dataset = []
train_data_path = open(r'')
train_save_path = os.path.join(base_path, 'train.npy')
train_save_path = train_save_path.replace("\\", "/")


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
    fr = input.shape[0]  # frame_number

    input = input.reshape(fr, -1, 3) # 固定3列 行数自动调节
    nd = input.shape[1]  # node_num
    input = torch.tensor(input, dtype=torch.float32)

    angle_velocity = torch.FloatTensor([])
    one_sequence = torch.FloatTensor([])
    for a in range(input.shape[0] - 1):
        one_frame = torch.FloatTensor([])
        for b in range(input.shape[1]):
            space_angles = space_angle(input[a, b], input[a + 1, b])
            space_velocity = space_distance(input[a, b], input[a + 1, b])
            space_angles = torch.where(torch.isnan(space_angles), torch.full_like(space_angles, 0), space_angles)
            one_frame = torch.cat([one_frame, space_angles, space_velocity], dim=1)
        one_sequence = torch.cat([one_sequence, one_frame], dim=0)
    angle_velocity = torch.cat([angle_velocity, one_sequence], dim=0)
    print('angle_velocity:\n', angle_velocity.shape)
    angle_velocity = angle_velocity.view(fr - 1, nd, 4)
    print('angle_velocity:\n', angle_velocity.shape)
    return angle_velocity

for train_one_data_path in train_data_path:

    keyword = 'Eating'
    if keyword in train_one_data_path:

        print(train_one_data_path)
        train_one_data_path = train_one_data_path.strip('\n')
        # load train data
        train_data = h5py.File(train_one_data_path, 'r')
        coordinate_normalize_joint = train_data['coordinate_normalize_joint'][:, move_joint, :]
        train_num = int(coordinate_normalize_joint.shape[0])
        print('coordinate_normalize_joint.shape:\n', coordinate_normalize_joint.shape)
        coordinate_normalize_joint = torch.tensor(coordinate_normalize_joint)
        for i in range(2):
            even_list = [x for x in range(i, coordinate_normalize_joint.shape[0], 2)]
            train = coordinate_normalize_joint[even_list, :, :]
            print('train.shape:\n', train.shape)
            angle_velocity = loc_exchange(train)
            position_set = train[1:]
            position_set = torch.tensor(position_set, dtype=torch.float32)
            angle_velocity = torch.tensor(angle_velocity, dtype=torch.float32)

            angle_position = torch.cat([angle_velocity, position_set], 2)
            print(angle_position.size())

            train_one_dataset = []
            for i in range(angle_position.shape[0]):
                train_data = angle_position[i]
                train_data = np.array(train_data)
                train_one_dataset.append(train_data)
            train_one_dataset = np.array(train_one_dataset)
            print('train_one_dataset:\n', train_one_dataset.shape)
            train_dataset.append(train_one_dataset)
    else:
        continue


train_dataset = np.array(train_dataset)
# save data
np.save(train_save_path, train_dataset)


test_data_path = open(
    r'/home/lv/MM/H3.6M/H36M/test.txt').readline()
print('test_data_path:\n', test_data_path)
test_data_path = test_data_path[:-1]
test_save_path = os.path.join(base_path, 'test.npy')
test_save_path = test_save_path.replace("\\", "/")

# load test data
test_data = h5py.File(test_data_path, 'r')
test_coordinate_normalize_joint = test_data['coordinate_normalize_joint'][:, move_joint, :] # (2984, 25, 3)
test_num = int(test_coordinate_normalize_joint.shape[0])
test_coordinate_normalize_joint = torch.tensor(test_coordinate_normalize_joint)
print('test_coordinate_normalize_joint.shape:\n', test_coordinate_normalize_joint.shape)
even_list = [x for x in range(0, test_coordinate_normalize_joint.shape[0], 2)]
test = test_coordinate_normalize_joint[even_list, :, :]

test_angle_velocity = loc_exchange(test)
test_position_set = test[1:] #第i帧的位置和第i-1帧的角度，速度拼接
test_position_set = torch.tensor(test_position_set, dtype=torch.float32)
test_angle_velocity = torch.tensor(test_angle_velocity, dtype=torch.float32)
test_angle_position = torch.cat([test_angle_velocity, test_position_set], 2)

print('test_dataset:\n', test_angle_position.shape)
# save data
np.save(test_save_path, test_angle_position)


test_data_path = open(
    r'').readline()
print('test_data_path:\n', test_data_path)
test_data_path = test_data_path[:-1]
test_save_path = os.path.join(base_path, 'test1.npy')
test_save_path = test_save_path.replace("\\", "/")

# load test data
test_data = h5py.File(test_data_path, 'r')
test_coordinate_normalize_joint = test_data['coordinate_normalize_joint'][:, move_joint, :]
test_num = int(test_coordinate_normalize_joint.shape[0])
test_coordinate_normalize_joint = torch.tensor(test_coordinate_normalize_joint)
print('test_coordinate_normalize_joint.shape:\n', test_coordinate_normalize_joint.shape)
even_list = [x for x in range(0, test_coordinate_normalize_joint.shape[0], 2)]
test = test_coordinate_normalize_joint[even_list, :, :]

test_angle_velocity = loc_exchange(test)
test_position_set = test[1:]
test_position_set = torch.tensor(test_position_set, dtype=torch.float32)
test_angle_velocity = torch.tensor(test_angle_velocity, dtype=torch.float32)
test_angle_position = torch.cat([test_angle_velocity, test_position_set], 2)
print('test_dataset:\n', test_angle_position.shape)
# save data
np.save(test_save_path, test_angle_position)