import yaml
import h5py
import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from torch.utils.data.dataloader import DataLoader
import data_utils
import space_angle_velocity
import bone_length_loss
import model_4GRU
import ctypes
import time
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')
import model_TCN
def find_indices_srnn(frame_num1, frame_num2, seq_len, input_n=10):
    SEED = 1234567890
    rng = np.random.RandomState(SEED)

    T1 = frame_num1 - 150
    T2 = frame_num2 - 150  # seq_len
    idxo1 = None
    idxo2 = None
    for _ in np.arange(0, 4):
        idx_ran1 = rng.randint(16, T1)
        idx_ran2 = rng.randint(16, T2)
        idxs1 = np.arange(idx_ran1 + 50 - input_n, idx_ran1 + 50 + seq_len)
        idxs2 = np.arange(idx_ran2 + 50 - input_n, idx_ran2 + 50 + seq_len)
        if idxo1 is None:
            idxo1 = idxs1
            idxo2 = idxs2
        else:
            idxo1 = np.vstack((idxo1, idxs1))
            idxo2 = np.vstack((idxo2, idxs2))
    return idxo1, idxo2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
config = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

node_num = 1
input_n = 10
output_n = 5
base_path = ''
input_size = config['in_features']
hidden_size = config['hidden_size']
output_size = 5
batch_size = config['batch_size']

test_save_path = os.path.join(base_path, 'test.npy')
test_save_path = test_save_path.replace("\\", "/")
dataset1 = np.load(test_save_path, allow_pickle=True)
dataset1 = torch.tensor(dataset1, dtype=torch.float32, requires_grad=False)

test_save_path = os.path.join(base_path, 'test1.npy')
test_save_path = test_save_path.replace("\\", "/")
dataset2 = np.load(test_save_path, allow_pickle=True)
dataset2 = torch.tensor(dataset2, dtype=torch.float32, requires_grad=False)


move_joint = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21, 22, 25, 26, 27, 29, 30])
no_move_joint = np.array([4,5,9,10,19,21,29,30])
use_node = np.array([0, 1, 2, 3, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22])

idx1, idx2 = find_indices_srnn(dataset1.shape[0], dataset2.shape[0], 5, input_n)
dataset1 = dataset1[idx1, :, :]
dataset2 = dataset2[idx2, :, :]

dataset = torch.cat([dataset1, dataset2], dim=0)
print(dataset.shape)
print(idx1)
print(idx2)

total_loss = [0.0] * output_n

dataset = dataset.to(device)
for i in range(dataset.shape[0]):

    input_dataset = dataset[i]
    output_dataset = dataset[i]

    input_dataset = input_dataset[0:input_n]
    output_dataset = output_dataset[input_n:input_n + output_n]

    input_dataset = input_dataset.expand(batch_size, input_dataset.shape[0], input_dataset.shape[1],
                                         input_dataset.shape[2])
    output_dataset = output_dataset.expand(batch_size, output_dataset.shape[0], output_dataset.shape[1],
                                           output_dataset.shape[2])

    joints_re_data = torch.FloatTensor([]).to(device)
    joints_target_3d_data = torch.FloatTensor([]).to(device)

    for k in use_node:
        # if not k in use_node:
        #         continue
        print(k)
        model_x = model_TCN.TCN(1, [1, 1, 1], kernel_size=3, dropout=0.05, block_num=3, input_n=input_n - 1,
                                output_n=output_n)
        model_y = model_TCN.TCN(1, [1, 1, 1], kernel_size=3, dropout=0.05, block_num=3, input_n=input_n - 1,
                                output_n=output_n)
        model_z = model_TCN.TCN(1, [1, 1, 1], kernel_size=3, dropout=0.05, block_num=3, input_n=input_n - 1,
                                output_n=output_n)
        model_v = model_TCN.TCN(1, [1, 1, 1], kernel_size=3, dropout=0.05, block_num=3, input_n=input_n - 1,
                                output_n=output_n)

        model_x.load_state_dict(torch.load(os.path.join(base_path,
                                                        'generator_x_' + str(move_joint[k]) + ".pkl")), strict=False)
        model_y.load_state_dict(torch.load(os.path.join(base_path,
                                                        'generator_y_' + str(move_joint[k]) + ".pkl")), strict=False)
        model_z.load_state_dict(torch.load(os.path.join(base_path,
                                                        'generator_z_' + str(move_joint[k]) + ".pkl")), strict=False)
        model_v.load_state_dict(torch.load(os.path.join(base_path,
                                                        'generator_v_' + str(move_joint[k]) + ".pkl")), strict=False)

        model_x.eval()
        model_y.eval()
        model_z.eval()
        model_v.eval()

        input_angle = input_dataset[:, 1:, k, :3].unsqueeze(dim=2)
        input_velocity = input_dataset[:, 1:, k, 3].unsqueeze(dim=2).permute(0, 2, 1)

        target_angle = output_dataset[:, :, k, :3].unsqueeze(dim=2)
        target_velocity = output_dataset[:, :, k, 3].unsqueeze(dim=2)

        # read velocity
        input_velocity = input_velocity.float()
        target_velocity = target_velocity.float()

        # read angle_x
        input_angle_x = input_angle[:, :, :, 0].permute(0, 2, 1).float()
        target_angle_x = target_angle[:, :, :, 0].float()

        # read angle_y
        input_angle_y = input_angle[:, :, :, 1].permute(0, 2, 1).float()
        target_angle_y = target_angle[:, :, :, 1].float()

        # read angle_z
        input_angle_z = input_angle[:, :, :, 2].permute(0, 2, 1).float()
        target_angle_z = target_angle[:, :, :, 2].float()

        # read 3D data
        input_3d_data = input_dataset[:, :, k, 4:].unsqueeze(dim=2)
        target_3d_data = output_dataset[:, :, k, 4:].unsqueeze(dim=2)
        joints_target_3d_data = torch.cat([joints_target_3d_data, target_3d_data], dim=2)

        output_v = model_v(input_velocity)
        output_v = output_v.view(target_velocity.shape[0], target_velocity.shape[2], output_size)

        output_x = model_x(input_angle_x)
        output_x = output_x.view(target_angle_x.shape[0], target_angle_x.shape[2], output_size)

        output_y = model_y(input_angle_y)
        output_y = output_y.view(target_angle_y.shape[0], target_angle_y.shape[2], output_size)

        output_z = model_z(input_angle_z)
        output_z = output_z.view(target_angle_z.shape[0], target_angle_z.shape[2], output_size)

        angle_x = output_x.permute(0, 2, 1)
        angle_y = output_y.permute(0, 2, 1)
        angle_z = output_z.permute(0, 2, 1)
        pred_v = output_v.permute(0, 2, 1)

        pred_angle_set = torch.stack((angle_x, angle_y, angle_z), 3)
        pred_angle_set = pred_angle_set.reshape(pred_angle_set.shape[0], pred_angle_set.shape[1], -1, 3)

        # reconstruction_loss
        input_pose = torch.zeros((target_velocity.shape[0], output_n, input_3d_data.shape[-2], input_3d_data.shape[-1]))

        for a in range(input_pose.shape[0]):
            input_pose[a, 0, :, :] = input_3d_data[a, input_n - 1, :, :]

        re_data = torch.FloatTensor([]).to(device)
        for b in range(target_3d_data.shape[0]):
            for c in range(target_3d_data.shape[1]):
                reconstruction_coordinate = space_angle_velocity.reconstruction_motion(pred_v[b, c, :, ],
                                                                                       pred_angle_set[b, c, :, :],
                                                                                       input_pose[b, c, :, :], node_num)

                reconstruction_coordinate = reconstruction_coordinate.to(device)
                re_data = torch.cat([re_data, reconstruction_coordinate], dim=0)
                reconstruction_coordinate = reconstruction_coordinate
                if c + 1 < target_3d_data.shape[1]:
                    input_pose[b, c + 1, :, :] = reconstruction_coordinate
                else:
                    continue

        re_data = re_data.view(target_3d_data.shape[0], -1, node_num, 3)
        joints_re_data = torch.cat([joints_re_data, re_data], dim=2)

    frame_re_data = joints_re_data[0]
    frame_target_3d_data = joints_target_3d_data[0]
    mpjpe_set = []

    for j in range(frame_re_data.shape[0]):
        frame_re_data = frame_re_data.to(device)
        frame_target_3d_data = frame_target_3d_data.to(device)
        frame_rec_loss = torch.mean(torch.norm(frame_re_data[j] - frame_target_3d_data[j], 2, 1))
        mpjpe_set.append(frame_rec_loss)

    for j in range(len(total_loss)):
        total_loss[j] += mpjpe_set[j]
    print("total_loss: ", total_loss)

    # save vis data
    print("i: ", i)
    if i == 0:
        frame_target_3d_data = frame_target_3d_data.cpu()
        frame_re_data = frame_re_data.cpu()
        frame_target_3d_data = np.array(frame_target_3d_data[0])
        mpjpe_set = np.array(mpjpe_set)

        print(frame_re_data[:, 1, :])
        move_loss = 0.0
        for m in range(1, 5):
            gap = frame_re_data[m, 1, 1] - frame_re_data[m - 1, 1, 1]
            gap = abs(gap)
            move_loss += gap
        move_loss = move_loss.numpy().sum()
        print(move_loss)

        vis_save_path = os.path.join(base_path, 'vis.npy')
        vis_mpjpe_save_path = os.path.join(base_path, 'vis_mpjpe.npy')
        np.save(vis_save_path, frame_re_data)
        exit()

    print('-------------------')
    print('mpjpe_set', mpjpe_set)
    print('frame_re_data.shape:\n', frame_re_data.shape)

    print('40ms:\n', mpjpe_set[0])
    print('80ms:\n', mpjpe_set[1])
    print('120ms:\n', mpjpe_set[2])
    print('160ms:\n', mpjpe_set[3])
    print('200ms:\n', mpjpe_set[4])

avg_loss = []
for i in range(len(total_loss)):
    avg_loss.append(total_loss[i]/8)
    avg_loss[i] = avg_loss[i].tolist()
print('avg_loss', avg_loss)

print('40ms:\n', avg_loss[0])
print('80ms:\n', avg_loss[1])
print('120ms:\n', avg_loss[2])
print('160ms:\n', avg_loss[3])
print('200ms:\n', avg_loss[4])