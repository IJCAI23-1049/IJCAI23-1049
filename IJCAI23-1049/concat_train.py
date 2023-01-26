from __future__ import print_function, absolute_import, division
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
import model_enhance_each_joint
import model_TCN
import model_GAN
import torchsnooper
import matplotlib.pyplot as plt
import scipy.io as io
import math
import ctypes
import time
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

node_num = config['node_num']
input_n = 25
output_n = 5
path_to_data = ''
base_path = ''
input_size = 25
hidden_size = config['hidden_size']
output_size = 50
lr = config['learning_rate']
batch_size = config['batch_size']
train_save_path = os.path.join(base_path, 'train.npy')
train_save_path = train_save_path.replace("\\", "/")
dataset = np.load(train_save_path, allow_pickle=True)

print(device)
move_joint = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21, 22, 25, 26, 27, 29, 30])
use_node = np.array([0, 1, 2, 3, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22])
frame_weight = torch.tensor([3, 2, 1.5, 1.5, 1, 0.5, 0.2, 0.2, 0.1, 0.1,
                             0.06, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.02, 0.02,
                             0.02, 0.02, 0.02, 0.02])

frame_weight_50 = []
for i in range(25):
    frame_weight_50.append(frame_weight[i])
    frame_weight_50.append(frame_weight[i])

frame_weight_50 = torch.tensor(frame_weight_50)
frame_weight = frame_weight_50.to(device)

time_start = time.time()
for k in use_node:

    model_x = model_TCN.TCN(1, [1, 1, 1, 1], kernel_size=3, dropout=0.05, block_num=3, input_n=input_n - 1, output_n=output_n)
    model_y = model_TCN.TCN(1, [1, 1, 1, 1], kernel_size=3, dropout=0.05, block_num=3, input_n=input_n - 1, output_n=output_n)
    model_z = model_TCN.TCN(1, [1, 1, 1, 1], kernel_size=3, dropout=0.05, block_num=3, input_n=input_n - 1, output_n=output_n)
    model_v = model_TCN.TCN(1, [1, 1, 1, 1], kernel_size=3, dropout=0.05, block_num=3, input_n=input_n - 1, output_n=output_n)

    Disc_x = model_GAN.Discriminator(input_n=output_size)
    Disc_y = model_GAN.Discriminator(input_n=output_size)
    Disc_z = model_GAN.Discriminator(input_n=output_size)

    criterion = nn.BCELoss()

    optimizer_x = optim.Adam(model_x.parameters(), lr)
    optimizer_y = optim.Adam(model_y.parameters(), lr)
    optimizer_z = optim.Adam(model_z.parameters(), lr)
    optimizer_v = optim.Adam(model_v.parameters(), lr)

    d_optimizer_x = torch.optim.Adam(Disc_x.parameters(), lr)
    d_optimizer_y = torch.optim.Adam(Disc_y.parameters(), lr)
    d_optimizer_z = torch.optim.Adam(Disc_z.parameters(), lr)

    scheduler_x = torch.optim.lr_scheduler.ExponentialLR(optimizer_x, gamma=0.98)
    scheduler_y = torch.optim.lr_scheduler.ExponentialLR(optimizer_y, gamma=0.98)
    scheduler_z = torch.optim.lr_scheduler.ExponentialLR(optimizer_z, gamma=0.98)
    scheduler_v = torch.optim.lr_scheduler.ExponentialLR(optimizer_v, gamma=0.98)

    d_scheduler_x = torch.optim.lr_scheduler.ExponentialLR(d_optimizer_x, gamma=0.95)
    d_scheduler_y = torch.optim.lr_scheduler.ExponentialLR(d_optimizer_y, gamma=0.95)
    d_scheduler_z = torch.optim.lr_scheduler.ExponentialLR(d_optimizer_z, gamma=0.95)

    for epoch in range(config['train_epoches']):

        for i in range(dataset.shape[0]):
            data = dataset[i]

            train_data = data_utils.LPDataset(data, input_size, output_size)
            train_loader = DataLoader(
                dataset=train_data,
                batch_size=config['batch_size'],
                shuffle=True,
                pin_memory=True,
                drop_last=True
            )

            if epoch == 0 and i == 0:
                # they must exist because pretrain
                model_x.load_state_dict(torch.load(os.path.join(base_path, 'generator_x_' + str(move_joint[k]) + ".pkl")), strict=False)
                model_y.load_state_dict(torch.load(os.path.join(base_path, 'generator_y_' + str(move_joint[k]) + ".pkl")), strict=False)
                model_z.load_state_dict(torch.load(os.path.join(base_path, 'generator_z_' + str(move_joint[k]) + ".pkl")), strict=False)
                model_v.load_state_dict(torch.load(os.path.join(base_path, 'generator_v_' + str(move_joint[k]) + ".pkl")), strict=False)

            else:

                model_x.load_state_dict(torch.load(os.path.join(base_path, 'ft_generator_x_' + str(move_joint[k]) + ".pkl")),strict=False)
                model_y.load_state_dict(torch.load(os.path.join(base_path, 'ft_generator_y_' + str(move_joint[k]) + ".pkl")),strict=False)
                model_z.load_state_dict(torch.load(os.path.join(base_path, 'ft_generator_z_' + str(move_joint[k]) + ".pkl")),strict=False)
                model_v.load_state_dict(torch.load(os.path.join(base_path, 'ft_generator_v_' + str(move_joint[k]) + ".pkl")),strict=False)

                Disc_x.load_state_dict(torch.load(os.path.join(base_path, 'ft_Disc_x_' + str(move_joint[k]) + ".pkl")), strict=False)
                Disc_y.load_state_dict(torch.load(os.path.join(base_path, 'ft_Disc_y_' + str(move_joint[k]) + ".pkl")), strict=False)
                Disc_z.load_state_dict(torch.load(os.path.join(base_path, 'ft_Disc_z_' + str(move_joint[k]) + ".pkl")), strict=False)

            model_x.to(device)
            model_y.to(device)
            model_z.to(device)
            model_v.to(device)

            Disc_x.to(device)
            Disc_y.to(device)
            Disc_z.to(device)

            for i, data in enumerate(train_loader):
                print("i:", i)

                optimizer_x.zero_grad()
                optimizer_y.zero_grad()
                optimizer_z.zero_grad()
                optimizer_v.zero_grad()

                d_optimizer_x.zero_grad()
                d_optimizer_y.zero_grad()
                d_optimizer_z.zero_grad()

                in_shots, out_shot = data
                in_shots = in_shots.to(device)
                out_shot = out_shot.to(device)

                gt_25frame = out_shot[:, :, :, :4]
                real_label = torch.ones(gt_25frame.size(0))  # 定义真实的图片label为1
                fake_label = torch.zeros(gt_25frame.size(0))  # 定义假的图片的label为0

                real_z = torch.rand(gt_25frame.size(0)) / 5
                real_label = real_label - real_z
                fake_z = torch.rand(gt_25frame.size(0)) / 5
                fake_label = fake_label + fake_z

                real_label = real_label.to(device)
                fake_label = fake_label.to(device)

                input_angle = in_shots[:, 1:, k, :3].unsqueeze(dim=2)
                input_velocity = in_shots[:, 1:, k, 3].unsqueeze(dim=2).permute(0, 2, 1)
                target_angle = out_shot[:, :, k, :3].unsqueeze(dim=2)
                target_velocity = out_shot[:, :, k, 3].unsqueeze(dim=2)

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
                input_3d_data = in_shots[:, :, k, 4:].unsqueeze(dim=2)
                target_3d_data = out_shot[:, :, k, 4:].unsqueeze(dim=2)
                input_3d_data = input_3d_data.to(device)
                target_3d_data = target_3d_data.to(device)

                loss_v = 0
                loss_x = 0
                loss_y = 0
                loss_z = 0

                input_velocity_10 = input_velocity
                input_angle_x_10 = input_angle_x
                input_angle_y_10 = input_angle_y
                input_angle_z_10 = input_angle_z

                output_v_5 = torch.FloatTensor([])
                output_x_5 = torch.FloatTensor([])
                output_y_5 = torch.FloatTensor([])
                output_z_5 = torch.FloatTensor([])

                angle_x = torch.FloatTensor([]).to((device))
                angle_y = torch.FloatTensor([]).to((device))
                angle_z = torch.FloatTensor([]).to((device))
                pred_v = torch.FloatTensor([]).to((device))

                # 拼接出25帧
                m = output_size // output_n
                for j in range(m):

                    output_v_5 = model_v(input_velocity_10)
                    output_v_5 = output_v_5.view(target_velocity.shape[0], target_velocity.shape[2], output_n)
                    target_velocity_5 = target_velocity[:, j * output_n: j * output_n + 5, :]
                    target_velocity_loss = target_velocity_5.permute(0, 2, 1)
                    loss_v += torch.mean(
                        torch.norm((output_v_5 - target_velocity_loss) * frame_weight[j * output_n: j * output_n + 5], 2, 1))

                    output_x_5 = model_x(input_angle_x_10)
                    output_x_5 = output_x_5.view(target_angle_x.shape[0], target_angle_x.shape[2], output_n)
                    target_angle_x_5 = target_angle_x[:, j * output_n: j * output_n + 5, :]
                    target_angle_x_loss = target_angle_x_5.permute(0, 2, 1)
                    loss_x += torch.mean(
                        torch.norm((output_x_5 - target_angle_x_loss) * frame_weight[j * output_n: j * output_n + 5], 2, 1))

                    output_y_5 = model_y(input_angle_y_10)
                    output_y_5 = output_y_5.view(target_angle_y.shape[0], target_angle_y.shape[2], output_n)
                    target_angle_y_5 = target_angle_y[:, j * output_n: j * output_n + 5, :]
                    target_angle_y_loss = target_angle_y_5.permute(0, 2, 1)
                    loss_y += torch.mean(
                        torch.norm((output_y_5 - target_angle_y_loss) * frame_weight[j * output_n: j * output_n + 5], 2, 1))

                    output_z_5 = model_z(input_angle_z_10)
                    output_z_5 = output_z_5.view(target_angle_z.shape[0], target_angle_z.shape[2], output_n)
                    target_angle_z_5 = target_angle_z[:, j * output_n: j * output_n + 5, :]
                    target_angle_z_loss = target_angle_z_5.permute(0, 2, 1)
                    loss_z += torch.mean(
                        torch.norm((output_z_5 - target_angle_z_loss) * frame_weight[j * output_n: j * output_n + 5], 2, 1))

                    # 迭代处理
                    input_velocity_10 = torch.cat([input_velocity_10[:, :, 5:], output_v_5[:, :, :]], dim=2)
                    input_angle_x_10 = torch.cat([input_angle_x_10[:, :, 5:], output_x_5[:, :, :]], dim=2)
                    input_angle_y_10 = torch.cat([input_angle_y_10[:, :, 5:], output_y_5[:, :, :]], dim=2)
                    input_angle_z_10 = torch.cat([input_angle_z_10[:, :, 5:], output_z_5[:, :, :]], dim=2)

                    # 拼接处理
                    angle_x = torch.cat([angle_x, output_x_5], dim=2)
                    angle_y = torch.cat([angle_y, output_y_5], dim=2)
                    angle_z = torch.cat([angle_z, output_z_5], dim=2)
                    pred_v = torch.cat([pred_v, output_v_5], dim=2)

                angle_x = angle_x.permute(0, 2, 1)
                angle_y = angle_y.permute(0, 2, 1)
                angle_z = angle_z.permute(0, 2, 1)
                pred_v = pred_v.permute(0, 2, 1)

                # total_loss = loss_v + loss_x + loss_y + loss_z

                # For　d_optimizer_x and optimizer_x
                # For discriminator
                real_out_x = Disc_x(target_angle_x)
                d_loss_real_x = criterion(real_out_x, real_label)  # 得到判别真实动作序列的loss
                real_scores_x = real_out_x  # 得到真实图片的判别值，输出的值越接近1越好

                fake_out_x = Disc_x(angle_x)  # 判别器判断假的图片，
                d_loss_fake_x = criterion(fake_out_x, fake_label)  # 得到假的动作序列的loss
                fake_scores_x = fake_out_x  # 对于判别器来说，假动作序列的损失越接近0越好

                d_loss_x = d_loss_real_x + d_loss_fake_x
                d_loss_x.backward(retain_graph=True)
                d_optimizer_x.step()
                scheduler_x

                # For generator
                # get fake data
                D_result_x = Disc_x(angle_x)
                g_loss_x = criterion(D_result_x, real_label) + loss_x
                g_loss_x.backward()
                nn.utils.clip_grad_norm_(model_x.parameters(), config['gradient_clip'])
                optimizer_x.step()

                # For　d_optimizer_y and optimizer_y
                # For discriminator
                real_out_y = Disc_y(target_angle_y)
                d_loss_real_y = criterion(real_out_y, real_label)  # 得到判别真实动作序列的loss
                real_scores_y = real_out_y  # 得到真实图片的判别值，输出的值越接近1越好

                fake_out_y = Disc_y(angle_y)  # 判别器判断假的图片，
                d_loss_fake_y = criterion(fake_out_y, fake_label)  # 得到假的动作序列的loss
                fake_scores_y = fake_out_y  # 对于判别器来说，假动作序列的损失越接近0越好

                d_loss_y = d_loss_real_y + d_loss_fake_y
                d_loss_y.backward(retain_graph=True)
                d_optimizer_y.step()

                # For generator
                # get fake data
                D_result_y = Disc_y(angle_y)
                g_loss_y = criterion(D_result_y, real_label) + loss_y
                g_loss_y.backward()
                nn.utils.clip_grad_norm_(model_y.parameters(), config['gradient_clip'])
                optimizer_y.step()

                # For　d_optimizer_z and optimizer_z
                # For discriminator
                real_out_z = Disc_z(target_angle_z)
                d_loss_real_z = criterion(real_out_z, real_label)  # 得到判别真实动作序列的loss
                real_scores_z = real_out_z  # 得到真实图片的判别值，输出的值越接近1越好

                fake_out_z = Disc_z(angle_z)  # 判别器判断假的图片，
                d_loss_fake_z = criterion(fake_out_z, fake_label)  # 得到假的动作序列的loss
                fake_scores_z = fake_out_z  # 对于判别器来说，假动作序列的损失越接近0越好

                d_loss_z = d_loss_real_z + d_loss_fake_z
                d_loss_z.backward(retain_graph=True)
                d_optimizer_z.step()

                # For generator
                # get fake data
                D_result_z = Disc_z(angle_z)
                g_loss_z = criterion(D_result_z, real_label) + loss_z
                g_loss_z.backward()
                nn.utils.clip_grad_norm_(model_z.parameters(), config['gradient_clip'])
                optimizer_z.step()

                loss_v.backward()
                nn.utils.clip_grad_norm_(model_v.parameters(), config['gradient_clip'])
                optimizer_v.step()

                print('[epoch %d] [step %d] [loss_x %.4f] [loss_y %.4f] [loss_z %.4f] [loss_v %.4f]' %
                      (epoch, i, loss_x.item(), loss_y.item(), loss_z.item(), loss_v.item()))

            torch.save(model_x.state_dict(), os.path.join(base_path, 'ft_generator_x_' + str(move_joint[k]) + ".pkl"))
            torch.save(model_y.state_dict(), os.path.join(base_path, 'ft_generator_y_' + str(move_joint[k]) + ".pkl"))
            torch.save(model_z.state_dict(), os.path.join(base_path, 'ft_generator_z_' + str(move_joint[k]) + ".pkl"))
            torch.save(model_v.state_dict(), os.path.join(base_path, 'ft_generator_v_' + str(move_joint[k]) + ".pkl"))

            torch.save(Disc_x.state_dict(), os.path.join(base_path, 'ft_Disc_x_' + str(move_joint[k]) + ".pkl"))
            torch.save(Disc_y.state_dict(), os.path.join(base_path, 'ft_Disc_y_' + str(move_joint[k]) + ".pkl"))
            torch.save(Disc_z.state_dict(), os.path.join(base_path, 'ft_Disc_z_' + str(move_joint[k]) + ".pkl"))

        scheduler_x.step()
        scheduler_y.step()
        scheduler_z.step()
        scheduler_v.step()

        d_scheduler_x.step()
        d_scheduler_y.step()
        d_scheduler_z.step()

    torch.save(model_x, os.path.join(base_path, 'ft_generator_x_' + str(move_joint[k]) + ".pkl"))
    torch.save(model_y, os.path.join(base_path, 'ft_generator_y_' + str(move_joint[k]) + ".pkl"))
    torch.save(model_z, os.path.join(base_path, 'ft_generator_z_' + str(move_joint[k]) + ".pkl"))
    torch.save(model_v, os.path.join(base_path, 'ft_generator_v_' + str(move_joint[k]) + ".pkl"))

    torch.save(Disc_x, os.path.join(base_path, 'ft_Disc_x_' + str(move_joint[k]) + ".pkl"))
    torch.save(Disc_y, os.path.join(base_path, 'ft_Disc_y_' + str(move_joint[k]) + ".pkl"))
    torch.save(Disc_z, os.path.join(base_path, 'ft_Disc_z_' + str(move_joint[k]) + ".pkl"))

time_end = time.time()
time_sum = time_end - time_start
print(time_sum)