# Written by Tiankai Yang

import os
import time
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import cv2
from sklearn.model_selection import train_test_split
from kornia.color import rgb_to_lab
import gc
from config import DefaultConfig

config = DefaultConfig()
device = DefaultConfig.device
# torch.set_default_dtype(torch.float32)


# Modified from https://github.com/zohrehazizi/torch_SSL by Tiankai Yang
class TimerClass():
    def __init__(self, report_timing=True):
        self.events = ['start']
        self.times = [time.time()]
        self.report_timing = report_timing

    def register(self, eventname):
        self.events.append(eventname)
        self.times.append(time.time())
        print(f"{eventname}: {self.times[-1]-self.times[-2]}")

    def print(self):
        if self.report_timing:
            totalTime = self.times[-1]-self.times[0]
            # save to file in config.temp_dir
            with open(os.path.join(DefaultConfig.temp_dir, "timing.txt"), "w") as f:
                f.write(f"total time: {totalTime}\n")
                for i in range(1, len(self.events)):
                    event = self.events[i]
                    time = self.times[i]
                    time_prev = self.times[i-1]
                    f.write(
                        f"{event}: {time-time_prev}, percentage: {(time-time_prev)/(totalTime)*100}\n")


def convert_to_torch(X):
    if type(X)!=torch.Tensor:
        return torch.tensor(X, dtype=torch.float32, device=device)
    if X.device != device:
        return X.to(device)
    return X


def convert_to_numpy(x):
    if type(x)==torch.Tensor:
        return x.detach().cpu().numpy()
    else:
        return x
    

def upsample_x2_lanczos(input_set):
    output_set = np.zeros((input_set.shape[0], input_set.shape[1], input_set.shape[2]*2, input_set.shape[3]*2))
    for i in range(input_set.shape[0]):
        for j in range(input_set.shape[1]):
            output_set[i,j,:,:] = cv2.resize(src=input_set[i,j,:,:], dsize=None, 
                                             fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
            # output_set[i,j,:,:] = cv2.resize(src=input_set[i,j,:,:], dsize=None, 
            #                                  fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    return output_set


def RGB2YUV(image_set):
    # image_set: Nx3xHxW
    Y_trans_matrix = torch.tensor([0.299, 0.587, 0.114])
    U_trans_matrix = torch.tensor([-0.14713, -0.28886, 0.436])
    V_trans_matrix = torch.tensor([0.615, -0.51499, -0.10001])
    h, w = image_set.shape[2], image_set.shape[3]
    image_set = image_set.view(image_set.shape[0], 3, -1)
    Y_set = torch.matmul(Y_trans_matrix, image_set).view(image_set.shape[0], 1, h, w)
    U_set = torch.matmul(U_trans_matrix, image_set).view(image_set.shape[0], 1, h, w)
    V_set = torch.matmul(V_trans_matrix, image_set).view(image_set.shape[0], 1, h, w)
    del image_set
    del Y_trans_matrix
    del U_trans_matrix
    del V_trans_matrix
    gc.collect()
    return Y_set, U_set, V_set
    # return torch.cat([Y_set, U_set, V_set], dim=1)


def RGB2LAB(image_set):
    # image_set: Nx3xHxW
    lab_set = rgb_to_lab(image_set)
    return lab_set


def load_casia_v2_lab(index=None):
    name_list = os.listdir(config.casia_v2_image_dir)
    # only keep .png files and remove extension
    name_list = [name for name in name_list if name.endswith(".png")]
    print("number of images: ", len(name_list))
    name_list.sort()

    # load images, surfaces and edges
    image_set = []
    surface_set = []
    edge_set = []

    if index is not None:
        # if index is a list
        if type(index) == list:
            name_list = [name_list[i] for i in index]
        # if index is a single number
        else:
            name_list = [name_list[index]]
        print(name_list)

    for i in range(len(name_list)):
        image = Image.open(os.path.join(
            config.casia_v2_image_dir, name_list[i]))
        surface = Image.open(os.path.join(
            config.casia_v2_surface_dir, name_list[i]))
        # edge = Image.open(os.path.join(config.casia_v2_edge_dir, name_list[i]))

        # [0, 1]
        image = transforms.ToTensor()(image)
        surface = transforms.ToTensor()(surface)
        # edge = transforms.ToTensor()(edge)

        ###!!!!!!! handle edge labels

        image_set.append(image)
        surface_set.append(surface)

    if index is not None:
        # to tensor
        image_set = torch.stack(image_set)
        surface_set = torch.stack(surface_set)
        # edge_set = torch.stack(edge_set)

        # turn to lab
        lab_set = RGB2LAB(image_set)
        del image_set
        gc.collect()
        return lab_set, surface_set
    
    test_image_set, image_set, test_surface_set, surface_set = \
        train_test_split(image_set, surface_set, test_size=0.8, shuffle=False)
    gc.collect()

    # to tensor
    image_set = torch.stack(image_set)
    surface_set = torch.stack(surface_set)
    test_image_set = torch.stack(test_image_set)
    test_surface_set = torch.stack(test_surface_set)

    # turn to lab
    lab_set = RGB2LAB(image_set)
    test_lab_set = RGB2LAB(test_image_set)
    del image_set
    del test_image_set
    gc.collect()

    return lab_set, surface_set, test_lab_set, test_surface_set


def load_casia_v2_yuv(index=None):
    name_list = os.listdir(config.casia_v2_image_dir)
    # only keep .png files and remove extension
    name_list = [name for name in name_list if name.endswith(".png")]
    print("number of images: ", len(name_list))
    name_list.sort()

    # load images, surfaces and edges
    image_set = []
    surface_set = []
    edge_set = []

    if index is not None:
        # if index is a list
        if type(index) == list:
            name_list = [name_list[i] for i in index]
        # if index is a single number
        else:
            name_list = [name_list[index]]
        print(name_list)

    # name_list = name_list[:100]

    for i in range(len(name_list)):
        image = Image.open(os.path.join(
            config.casia_v2_image_dir, name_list[i]))
        surface = Image.open(os.path.join(
            config.casia_v2_surface_dir, name_list[i]))
        # edge = Image.open(os.path.join(config.casia_v2_edge_dir, name_list[i]))

        # [0, 255]
        image = transforms.ToTensor()(image)
        image = image * 255
        # [0, 1]
        surface = transforms.ToTensor()(surface)
        # edge = transforms.ToTensor()(edge)

        ###!!!!!!! handle edge labels

        image_set.append(image)
        surface_set.append(surface)

    if index is not None:
        # to tensor
        image_set = torch.stack(image_set)
        surface_set = torch.stack(surface_set)
        y_set, u_set, v_set = RGB2YUV(image_set)
        return y_set, u_set, v_set, surface_set
    
    test_images_set, image_set, test_surfaces_set, surface_set = \
        train_test_split(image_set, surface_set, test_size=0.8, shuffle=False)
    # _, image_set, _, surface_set = \
    #     train_test_split(image_set, surface_set, test_size=0.5, shuffle=True)
    # test_name_list, name_list, _, _ = \
    #     train_test_split(name_list, name_list, test_size=0.8, shuffle=False)
    # print(test_name_list)
    # print(len(test_name_list))
    # print(name_list[0])
    # print(test_name_list[0])
    # print(name_list[-1])
    # print(test_name_list[-1])
    # print("number of training images: ", len(image_set))
    # print("number of testing images: ", len(test_images_set))
    gc.collect()

    # to tensor
    image_set = torch.stack(image_set)
    surface_set = torch.stack(surface_set)
    test_image_set = torch.stack(test_images_set)
    test_surface_set = torch.stack(test_surfaces_set)

    # to YUV
    y_set, u_set, v_set = RGB2YUV(image_set)
    test_y_set, test_u_set, test_v_set = RGB2YUV(test_image_set)

    return y_set, u_set, v_set, surface_set, \
              test_y_set, test_u_set, test_v_set, test_surface_set


def load_columbia_yuv(index=None):
    name_list = os.listdir(config.columbia_image_dir)
    name_list = [name for name in name_list if name.endswith(".png")]
    print("number of images: ", len(name_list))
    name_list.sort()

    # load images, surfaces and edges
    image_set = []
    surface_set = []
    edge_set = []

    if index is not None:
        # if index is a list
        if type(index) == list:
            name_list = [name_list[i] for i in index]
        # if index is a single number
        else:
            name_list = [name_list[index]]

    for i in range(len(name_list)):
        image = Image.open(os.path.join(
            config.columbia_image_dir, name_list[i]))
        surface = Image.open(os.path.join(
            config.columbia_surface_dir, name_list[i]))
        # edge = Image.open(os.path.join(config.casia_v2_edge_dir, name_list[i]))

        # [0, 255]
        image = transforms.ToTensor()(image)
        image = image * 255
        # [0, 1]
        surface = transforms.ToTensor()(surface)
        # edge = transforms.ToTensor()(edge)

        ###!!!!!!! handle edge labels

        image_set.append(image)
        surface_set.append(surface)
        
    # to tensor
    image_set = torch.stack(image_set)
    surface_set = torch.stack(surface_set)

    # to YUV
    y_set, u_set, v_set = RGB2YUV(image_set)
    if index is not None:
        print(name_list[0])
    
    return y_set, u_set, v_set, surface_set

