# Modified from https://github.com/zohrehazizi/torch_SSL by Tiankai Yang
# The "mode" is not used. Omit it.

import os
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc
from utils import convert_to_torch, convert_to_numpy
from config import DefaultConfig

device = DefaultConfig.device
prefix = DefaultConfig.pre_fix


def remove_mean(features, dim):
    feature_mean = torch.mean(features, dim=dim, keepdims=True)
    feature_remove_mean = features - feature_mean
    return feature_remove_mean, feature_mean


def select_features(out_channel_num, kernel_size,
                    preserved_channel_num):
    channel_list = [i for i in range(out_channel_num)
                    if i % (kernel_size * kernel_size)
                    < preserved_channel_num]
    return channel_list


class TorchSaab(nn.Module):
    def __init__(self, kernel_size, stride, padding, channelwise):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.feature_expectation = torch.nn.parameter.Parameter(
            data=None, requires_grad=False)
        self.ac_kernels = torch.nn.parameter.Parameter(
            data=None, requires_grad=False)
        self.compute_mode = 'saab'
        self.bias = torch.nn.parameter.Parameter(
            data=None, requires_grad=False)
        self.has_bias = False
        self.channelwise = channelwise
        self.in_w = None
        self.in_h = None
        self.in_channels = None
        self.w = None
        self.h = None
        self.out_channels = None
        self.padding = padding

    # def set_param_sizes(self, input_size):
    #     self.in_channels, self.in_w, self.in_h = input_size
    #     self.w = (self.in_w-self.kernel_size[0])//self.stride[0]
    #     self.h = (self.in_h-self.kernel_size[1])//self.stride[1]
    #     self.out_channels = self.in_channels * \
    #         self.kernel_size[0]*self.kernel_size[1]

    def fit(self, x, bias=None, subsample=False):
        if self.channelwise:
            return self.fit_channelwise(x, bias, subsample)
        # x = convert_to_torch(x)

        self.in_w = x.size(2)
        self.in_h = x.size(3)
        self.in_channels = x.size(1)
        self.w = (self.in_w + 2 * self.padding -
                  self.kernel_size[0]) // self.stride[0] + 1
        self.h = (self.in_h + 2 * self.padding -
                  self.kernel_size[1]) // self.stride[1] + 1
        # padding
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding,
                      self.padding, self.padding), mode='reflect')
            
        # # compute current memory usage
        # need_memory = torch.cuda.memory_allocated() / 1024 / 1024 / 1024 * 4
        # if need_memory > max_memory:
        #     print('Subsampling...')
        #     print('Current memory usage: {} GB'.format(need_memory))
        #     # print(x.size())
        #     # subsample x randomly, reduce computation
        #     # subsample_num = max_memory / need_memory * x.size(0)
        #     # subsample_num = int(subsample_num)
        #     subsample_num = x.size(0) // 3 * 2
        #     print('Subsample number: {}'.format(subsample_num))
        #     index = np.random.choice(x.size(0), subsample_num, replace=False)
        #     x = x[index]
        #     gc.collect()
        #     # clear memory
        #     torch.cuda.empty_cache()
            
        sample_patches = F.unfold(
            x, self.kernel_size, stride=self.stride)  # <n, dim, windows>
        #print(f"GB: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024}")
        
        n, dim, windows = sample_patches.size()
        sample_patches = sample_patches.permute(0, 2, 1)  # <n, windows, dim>
        gc.collect()
        sample_patches = sample_patches.reshape(n*windows, dim)  # <n.windows, dim>

        # print(f"sample_patches: {sample_patches.size()}")
        if subsample and sample_patches.size(0) > 50000000:
            # subsample x randomly, reduce computation
            subsample_num = 50000000 
            index = np.random.choice(sample_patches.size(0), subsample_num, replace=False)
            sample_patches = sample_patches[index]
            gc.collect()

        rank_ac = dim-1

        sample_patches_ac, dc = remove_mean(
            sample_patches, dim=1)  # Remove patch mean
        del sample_patches
        gc.collect()

        training_data, feature_expectation = remove_mean(
            sample_patches_ac, dim=0)
        
        self.feature_expectation.data = feature_expectation
        print(f"training_data: {training_data.size()}")
        U, s, V = torch.linalg.svd(training_data, full_matrices=False)
        self.ac_kernels.data = V[:rank_ac]
        del U, s, V
        gc.collect()
        num_channels = self.ac_kernels.size(-1)
        dc = dc * num_channels * 1.0 / np.sqrt(num_channels)
        ac = torch.matmul(
            training_data, torch.transpose(self.ac_kernels, 1, 0))
        transformed = torch.cat([dc, ac], dim=1)
        del training_data, dc, ac
        gc.collect()
        bias = torch.max(torch.norm(transformed, dim=1))
        # if subsample:
        #     feature_layer = self.saab(x)
        # else:
        #     feature_layer = transformed.reshape(
        #         n, self.w, self.h, -1)  # -> [num, w, h, c]
        #     feature_layer = feature_layer.permute(0, 3, 1, 2)  # -> [num, c, w, h]
        del transformed
        gc.collect()
        # self.out_channels = feature_layer.size(1)
        # return feature_layer, bias
        return bias

    def fit_channelwise(self, x, bias, subsample=False):
        # x = convert_to_torch(x)
        if bias is not None:
            if bias.ndim != 0:
                bias = bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            self.bias.data = bias
            self.has_bias = True
            x = x+bias  # <n,c,w,h>
        n, self.in_channels, self.in_w, self.in_h = x.size()
        self.w = (self.in_w + 2 * self.padding -
                  self.kernel_size[0]) // self.stride[0] + 1
        self.h = (self.in_h + 2 * self.padding -
                  self.kernel_size[1]) // self.stride[1] + 1
        # padding
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding,
                      self.padding, self.padding), mode='reflect')
        # <n, dim, windows>, where dim=c.kernelsize[0].kernelsize[1]
        sample_patches = F.unfold(x, self.kernel_size, stride=self.stride)
        sample_patches = sample_patches.permute(0, 2, 1)  # <n, windows, dim>
        # <n, w, h, c, k0k1>dim = self.kernel_size[0]*self.kernel_size[1]
        sample_patches = sample_patches.reshape(
            n, self.w, self.h, self.in_channels, -1)
        sample_patches = sample_patches.permute(
            3, 0, 1, 2, 4)  # <c, n, w, h, k.k1>
        sample_patches = sample_patches.reshape(
            self.in_channels, -1, self.kernel_size[0]*self.kernel_size[1])  # <c, nwh, k0k1>
        
        if subsample and self.in_channels * sample_patches.size(1) > 50000000:
            # some problems here!!
            subsample_num = 50000000 // self.in_channels
            subsample_num = subsample_num if subsample_num > 100000 else 100000
            print(f"subsample_num: {subsample_num}")
            # subsample_num = 50000000
            new_sample_patches = torch.zeros(
                self.in_channels, subsample_num, self.kernel_size[0]*self.kernel_size[1], device=sample_patches.device)
            for c in range(self.in_channels):
                index = np.random.choice(sample_patches.size(1), subsample_num, replace=False)
                new_sample_patches[c] = sample_patches[c][index]
                gc.collect()
            sample_patches = new_sample_patches
            del new_sample_patches
            gc.collect()
            
                
        sample_patches_ac, dc = remove_mean(sample_patches, dim=2)
        del sample_patches
        gc.collect()
        training_data, feature_expectation = remove_mean(
            sample_patches_ac, dim=1)
        del sample_patches_ac
        gc.collect()
        self.feature_expectation.data = feature_expectation

        rank_ac = self.kernel_size[0]*self.kernel_size[1]-1

        U, s, V = torch.linalg.svd(training_data, full_matrices=False)
        
        self.ac_kernels.data = V[:, :rank_ac, :]
        # print(f"ac_kernels shape: {self.ac_kernels.shape}")
        del U, s, V
        gc.collect()
        num_channels = self.ac_kernels.shape[-1]
        dc = dc * num_channels * 1.0 / np.sqrt(num_channels)
        ac = torch.matmul(training_data, self.ac_kernels.permute(0, 2, 1))
        # print(f"ac shape: {ac.shape}")
        del training_data
        gc.collect()
        transformed = torch.cat([dc, ac], dim=2)  # <c, n.windows, k0.k1>
        del dc, ac
        gc.collect()
        next_bias = torch.norm(transformed, dim=2)
        next_bias, _ = torch.max(next_bias, dim=1)
        # if subsample:
        #     transformed = self.saab_channelwise(x)
        # else:
        #     # transformed = transformed.reshape(
        #     #     self.in_channels, n, self.w, self.h, self.kernel_size[0]*self.kernel_size[1])
        #     transformed = transformed.reshape(
        #         self.in_channels, n, self.w, self.h, rank_ac + 1)
        #     transformed = transformed.permute(1, 0, 4, 2, 3)
        #     transformed = transformed.reshape(n, -1, self.w, self.h)
        # next_bias = next_bias.repeat(
        #     self.kernel_size[0]*self.kernel_size[1], 1)
        next_bias = next_bias.repeat(rank_ac + 1, 1)
        next_bias = next_bias.transpose(1, 0)
        next_bias = next_bias.ravel()
        # self.out_channels = transformed.size(1)
        # return transformed, next_bias
        return next_bias

    def forward(self, x):
        if self.compute_mode == 'saab':
            if self.channelwise:
                return self.saab_channelwise(x)
            else:
                return self.saab(x)
        elif self.compute_mode == 'inverse_saab':
            if self.channelwise:
                return self.inverse_saab_channelwise(x)
            else:
                return self.inverse_saab(x)
        else:
            assert False

    def saab(self, x):
        print(f"x shape: {x.shape}")
        if self.channelwise:
            return self.saab_channelwise(x)
        # x = convert_to_torch(x)
        assert x.size(1) == self.in_channels
        assert x.size(2) == self.in_w
        assert x.size(3) == self.in_h
        # padding
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding,
                      self.padding, self.padding), mode='reflect')
        sample_patches = F.unfold(
            x, self.kernel_size, stride=self.stride)  # <n, dim, windows>
        n, dim, windows = sample_patches.size()
        sample_patches = sample_patches.permute(0, 2, 1)  # <n, windows, dim>
        sample_patches = sample_patches.reshape(
            n*windows, dim)  # <n.windows, dim>
        sample_patches_ac, dc = remove_mean(
            sample_patches, dim=1)  # Remove patch mean
        del sample_patches
        gc.collect()
        sample_patches_centered_ac = sample_patches_ac - self.feature_expectation
        del sample_patches_ac
        gc.collect()
        num_channels = self.ac_kernels.size(-1)
        dc = dc * num_channels * 1.0 / np.sqrt(num_channels)
        ac = torch.matmul(sample_patches_centered_ac,
                          self.ac_kernels.transpose(1, 0))
        transformed = torch.cat([dc, ac], dim=1)
        del dc, ac, sample_patches_centered_ac
        gc.collect()
        feature_layer = transformed.reshape(
            n, self.w, self.h, -1)  # -> [num, w, h, c]
        del transformed
        gc.collect()
        feature_layer = feature_layer.permute(0, 3, 1, 2)  # -> [num, c, w, h]
        return feature_layer

    def saab_channelwise(self, x):
        # x = convert_to_torch(x)
        assert x.size(1) == self.in_channels
        assert x.size(2) == self.in_w
        assert x.size(3) == self.in_h
        if self.has_bias:
            x = x+self.bias  # <n,c,w,h>
        n, self.in_channels, self.in_w, self.in_h = x.size()
        self.w = (self.in_w + 2 * self.padding -
                  self.kernel_size[0]) // self.stride[0] + 1
        self.h = (self.in_h + 2 * self.padding -
                  self.kernel_size[1]) // self.stride[1] + 1
        # padding
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding,
                      self.padding, self.padding), mode='reflect')
        # <n, dim, windows>, where dim=c.kernelsize[0].kernelsize[1]
        sample_patches = F.unfold(x, self.kernel_size, stride=self.stride)
        sample_patches = sample_patches.permute(0, 2, 1)  # <n, windows, dim>
        # <n, w, h, c, k0k1>dim = self.kernel_size[0]*self.kernel_size[1]
        sample_patches = sample_patches.reshape(
            n, self.w, self.h, self.in_channels, -1)
        sample_patches = sample_patches.permute(
            3, 0, 1, 2, 4)  # <c, n, w, h, k.k1>
        sample_patches = sample_patches.reshape(
            self.in_channels, -1, self.kernel_size[0]*self.kernel_size[1])  # <c, nwh, k0k1>
        sample_patches_ac, dc = remove_mean(sample_patches, dim=2)
        del sample_patches
        gc.collect()
        sample_patches_ac = sample_patches_ac-self.feature_expectation
        num_channels = self.ac_kernels.shape[-1]
        dc = dc * num_channels * 1.0 / np.sqrt(num_channels)
        ac = torch.matmul(sample_patches_ac, self.ac_kernels.permute(0, 2, 1))
        transformed = torch.cat([dc, ac], dim=2)  # <c, n.windows, k0.k1>
        del dc, ac, sample_patches_ac
        gc.collect()
        # transformed = transformed.reshape(
        #     self.in_channels, n, self.w, self.h, self.kernel_size[0]*self.kernel_size[1])
        transformed = transformed.reshape(self.in_channels, n, self.w, self.h, -1)
        transformed = transformed.permute(1, 0, 4, 2, 3)
        transformed = transformed.reshape(n, -1, self.w, self.h)
        return transformed

    def inverse_saab(self, x):
        if self.channelwise:
            return self.inverse_saab_channelwise(x)
        # x = convert_to_torch(x)
        assert x.size(2) == self.w
        assert x.size(3) == self.h
        sample_images = x.permute(0, 2, 3, 1)  # <n, w, h, c>
        n, w, h, c = sample_images.size()
        sample_patches = sample_images.reshape(n*w*h, c)
        del sample_images
        gc.collect()
        dc_comp = sample_patches[:, 0:1] * 1.0 / \
            np.sqrt(self.ac_kernels.shape[-1])
        needed_ac_num = c - 1
        print(f"self.ac_kernels.shape: {self.ac_kernels.shape}")
        ac_comp = torch.matmul(
            sample_patches[:, 1:], self.ac_kernels[:needed_ac_num, :])
        del sample_patches
        gc.collect()
        sample_rec = dc_comp + (ac_comp + self.feature_expectation)
        del dc_comp, ac_comp
        gc.collect()
        sample_rec = sample_rec.reshape(n, -1, sample_rec.size(1))
        sample_rec = sample_rec.permute(0, 2, 1)
        out = F.fold(sample_rec, output_size=(self.in_w, self.in_h),
                     kernel_size=self.kernel_size, stride=self.stride)
        del sample_rec
        gc.collect()
        # cut padding
        if self.padding > 0:
            out = out[:, :, self.padding:-self.padding,
                      self.padding:-self.padding]
            gc.collect()
        return out

    def inverse_saab_channelwise(self, x):
        # x = convert_to_torch(x)
        assert x.size(2) == self.w
        assert x.size(3) == self.h
        n = x.size(0)
        x = x.reshape(x.size(0), self.in_channels, -1, self.w, self.h)
        x = x.permute(1, 0, 3, 4, 2)
        x = x.reshape(self.in_channels, -1, x.size(-1))
        dc_comp = x[:, :, 0:1] * 1.0 / np.sqrt(self.ac_kernels.size(-1))
        need_ac_num = x.size(-1) - 1
        print(f"self.ac_kernels.shape: {self.ac_kernels.shape}")
        ac_comp = torch.matmul(
            x[:, :, 1:], self.ac_kernels[:, :need_ac_num, :])
        sample_rec = dc_comp + (ac_comp + self.feature_expectation)
        sample_rec = sample_rec.reshape(
            self.in_channels, n, -1, sample_rec.size(2))
        sample_rec = sample_rec.permute(1, 2, 0, 3)  # ??????????????????
        sample_rec = sample_rec.reshape(
            sample_rec.size(0), sample_rec.size(1), -1)
        sample_rec = sample_rec.permute(0, 2, 1)
        del dc_comp, ac_comp
        gc.collect()
        out = F.fold(sample_rec, output_size=(self.in_w, self.in_h),
                     kernel_size=self.kernel_size, stride=self.stride)
        del sample_rec
        gc.collect()
        # cut padding
        if self.padding > 0:
            out = out[:, :, self.padding:-self.padding,
                      self.padding:-self.padding]
        # print(self.bias)
        return out-self.bias

    def get_kernel_size(self):
        return self.kernel_size[0]
    

class SurfaceSSLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = TorchSaab(kernel_size=(3, 3), stride=(1, 1), padding=1, channelwise=False)
        self.layer_2 = TorchSaab(kernel_size=(3, 3), stride=(1, 1), padding=1, channelwise=True)
        self.layer_3 = TorchSaab(kernel_size=(3, 3), stride=(1, 1), padding=1, channelwise=True)
        self.layer_4 = TorchSaab(kernel_size=(3, 3), stride=(1, 1), padding=1, channelwise=True)
        self.layer_5 = TorchSaab(kernel_size=(3, 3), stride=(1, 1), padding=1, channelwise=True)
        layer_set = [self.layer_1, self.layer_2, self.layer_3, self.layer_4, self.layer_5]
        self.layer_set = nn.ModuleList(layer_set)

    def forward(self, x, index_set=None, need_save=False, name=prefix):
        import matplotlib.pyplot as plt
        from PIL import Image
        import os
        from config import DefaultConfig
        config = DefaultConfig()
        x = convert_to_torch(x)
        feature_set = []
        for i, layer in enumerate(self.layer_set):
            x = layer(x)
            # output_num = x.size(1)
            # # x = x[:, [i for i in range(output_num) if i % 9 < 3], :, :]
            # channel_list = [i for i in range(output_num) if i % 9 < 3]
            # # insert 1, 2 to channel_list
            # # channel_list.insert(1, 1)
            # # channel_list.insert(2, 2)
            # channel_list.insert(4, 4)
            # channel_list.insert(5, 5)
            # channel_list.insert(6, 6)
            # x = x[:, channel_list, :, :]
            x = x[:, :3, :, :]
            # insert 1, 2
            # max pooling
            x = F.max_pool2d(x, kernel_size=2, stride=2)

            # for j in range(x.size(1)):
            #     # save image
            #     # min-max normalization 
            #     x_norm = x[0, j, :, :].detach().numpy()
            #     x_norm = (x_norm - np.min(x_norm)) / (np.max(x_norm) - np.min(x_norm)) * 255
            #     x_norm = x_norm.astype(np.uint8)
            #     x_norm = Image.fromarray(x_norm, mode="L")
            #     x_norm.save(os.path.join(config.temp_dir, f"layer_{i}_channel_{j}.png"))

            if need_save:
                temp_x = convert_to_numpy(x)
                np.save(os.path.join(config.model_dir, f"{name}_level_{i+1}.npy"), temp_x)
                del temp_x
                gc.collect()

            if index_set is not None and i + 1 in index_set:
                temp_x = convert_to_numpy(x)
                feature_set.append(deepcopy(temp_x))
                del temp_x
                gc.collect()

            # drop AC1 and AC2, preserve DC only
            x = x[:, :1, :, :]

        if index_set is not None:
            return feature_set if len(feature_set) > 1 else feature_set[0]
        return x
    
    def fit(self, x):
        x = convert_to_torch(x)
        bias = None
        for i, layer in enumerate(self.layer_set):
            layer.compute_mode = "saab"
            # if i == 0 or i == 1:
            #     _, bias = layer.fit(x, bias, subsample=True)
            #     x = layer(x)
            # else:
            #     x, bias = layer.fit(x, bias)
            bias = layer.fit(x, bias, subsample=True)
            x = layer(x)
            # output_num = x.size(1)
            # x = x[:, [i for i in range(output_num) if i % 9 < 3], :, :]
            # channel_list = [i for i in range(output_num) if i % 9 < 3]
            # # insert 1, 2 to channel_list
            # # channel_list.insert(1, 1)
            # # channel_list.insert(2, 2)
            # channel_list.insert(4, 4)
            # channel_list.insert(5, 5)
            # channel_list.insert(6, 6)
            # x = x[:, channel_list, :, :]
            # x = x[:, :3, :, :]
            # drop AC1 and AC2, preserve DC only
            x = x[:, :1, :, :]
            # if bias has no dimension
            if bias.shape != torch.Size([]):
                # bias = bias[[i for i in range(output_num) if i % 9 < 3]]
                # bias = bias[channel_list]
                # bias = bias[:3]
                # drop AC1 and AC2, preserve DC only
                bias = bias[:1]
            # max pooling
            # if i != len(self.layer_set) - 1:
            #     x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            
            print(f"learned parameters for layer {i}, output size was {x.size()}")

    def level_inverse(self, x, level_index):
        x = convert_to_torch(x)
        x = self.layer_set[level_index-1].inverse_saab(x)
        return convert_to_numpy(x)
    

class LABSSLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = TorchSaab(kernel_size=(3, 3), stride=(1, 1), padding=1, channelwise=False)
        self.layer_2 = TorchSaab(kernel_size=(3, 3), stride=(1, 1), padding=1, channelwise=True)
        self.layer_3 = TorchSaab(kernel_size=(3, 3), stride=(1, 1), padding=1, channelwise=True)
        self.layer_4 = TorchSaab(kernel_size=(3, 3), stride=(1, 1), padding=1, channelwise=True)
        self.layer_5 = TorchSaab(kernel_size=(3, 3), stride=(1, 1), padding=1, channelwise=True)
        layer_set = [self.layer_1, self.layer_2, self.layer_3, self.layer_4, self.layer_5]
        self.layer_set = nn.ModuleList(layer_set)

    def forward(self, x, index_set=None, need_save=False, name=prefix):
        x = convert_to_torch(x)
        feature_set = []
        for i, layer in enumerate(self.layer_set):
            x = layer(x)
            if i >= 2 - 1:
                output_num = x.size(1)
                x = x[:, [i for i in range(output_num) if i % 9 < 3], :, :]
            # max pooling
            x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
            if need_save:
                height, width = x.size(2), x.size(3)
                # scan each channel with 3x3 kernel, use F.unfold
                temp_x = F.pad(x, (1, 1, 1, 1), mode='reflect')
                temp_x = F.unfold(temp_x, kernel_size=(3, 3), stride=(1, 1))
                temp_x = temp_x.view(temp_x.size(0), -1, height, width)
                print(f"temp_x size is {temp_x.size()}")
                temp_x = convert_to_numpy(temp_x)
                
                np.save(os.path.join(DefaultConfig.model_dir, f"{name}_level_{i+1}.npy"), temp_x)
                del temp_x
                gc.collect()

            if index_set is not None and i + 1 in index_set:
                height, width = x.size(2), x.size(3)
                # scan each channel with 3x3 kernel, use F.unfold
                temp_x = F.pad(x, (1, 1, 1, 1), mode='reflect')
                temp_x = F.unfold(temp_x, kernel_size=(3, 3), stride=(1, 1))
                temp_x = temp_x.view(temp_x.size(0), -1, height, width)
                print(f"temp_x size is {temp_x.size()}")
                temp_x = convert_to_numpy(temp_x)
                feature_set.append(deepcopy(temp_x))
                del temp_x
                gc.collect()
        if index_set is not None:
            return feature_set if len(feature_set) > 1 else feature_set[0]
        return x

    
    def fit(self, x):
        x = convert_to_torch(x)
        bias = None
        for i, layer in enumerate(self.layer_set):
            bias = layer.fit(x, bias, subsample=True)
            x = layer(x)
            if i >= 2 - 1:
                output_num = x.size(1)
                channel_list = [i for i in range(output_num) if i % 9 < 3]
                x = x[:, channel_list, :, :]
                if bias.shape != torch.Size([]):
                    bias = bias[channel_list]

            x = F.max_pool2d(x, kernel_size=2, stride=2)
            print(f"learned parameters for layer {i}, output size was {x.size()}")


class YUVSSLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = TorchSaab(kernel_size=(3, 3), stride=(1, 1), padding=1, channelwise=False)
        self.layer_2 = TorchSaab(kernel_size=(3, 3), stride=(1, 1), padding=1, channelwise=True)
        self.layer_3 = TorchSaab(kernel_size=(3, 3), stride=(1, 1), padding=1, channelwise=True)
        self.layer_4 = TorchSaab(kernel_size=(3, 3), stride=(1, 1), padding=1, channelwise=True)
        layer_set = [self.layer_1, self.layer_2, self.layer_3, self.layer_4]
        self.layer_set = nn.ModuleList(layer_set)

    def forward(self, x, index_set=None, need_save=False, name="y"):
        x = convert_to_torch(x)
        feature_set = []
        for i, layer in enumerate(self.layer_set):
            x = layer(x)
            # if i >= 3 - 1:
            if i >= 2 - 1:
                output_num = x.size(1)
                x = x[:, [i for i in range(output_num) if i % 9 < 3], :, :]
            # max pooling
            x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
            if need_save:
                height, width = x.size(2), x.size(3)
                # scan each channel with 3x3 kernel, use F.unfold
                temp_x = F.pad(x, (1, 1, 1, 1), mode='reflect')
                temp_x = F.unfold(temp_x, kernel_size=(3, 3), stride=(1, 1))
                print(f"temp_x size is {temp_x.size()}")
                temp_x = temp_x.view(temp_x.size(0), -1, height, width)
                print(f"temp_x size is {temp_x.size()}")
                temp_x = convert_to_numpy(temp_x)
                
                np.save(os.path.join(DefaultConfig.model_dir, f"33_{name}_level_{i+1}.npy"), temp_x)
                del temp_x
                gc.collect()

            if index_set is not None and i + 1 in index_set:
                height, width = x.size(2), x.size(3)
                # scan each channel with 3x3 kernel, use F.unfold
                temp_x = F.pad(x, (1, 1, 1, 1), mode='reflect')
                temp_x = F.unfold(temp_x, kernel_size=(3, 3), stride=(1, 1))
                print(f"temp_x size is {temp_x.size()}")
                temp_x = temp_x.view(temp_x.size(0), -1, height, width)
                print(f"temp_x size is {temp_x.size()}")
                # n, dim, windows = temp_x.size()
                # temp_x = temp_x.permute(0, 2, 1).contiguous().view(n*windows, dim)
                temp_x = convert_to_numpy(temp_x)
                feature_set.append(temp_x)
        if index_set is not None:
            return feature_set if len(feature_set) > 1 else feature_set[0]
        return x

    
    def fit(self, x):
        x = convert_to_torch(x)
        bias = None
        for i, layer in enumerate(self.layer_set):
            bias = layer.fit(x, bias, subsample=True)
            x = layer(x)
            # if i >= 3 - 1:
            if i >= 2 - 1:
                output_num = x.size(1)
                channel_list = [i for i in range(output_num) if i % 9 < 3]
                x = x[:, channel_list, :, :]
                if bias.shape != torch.Size([]):
                    bias = bias[channel_list]

            # if i != len(self.layer_set) - 1:
            #     x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            print(f"learned parameters for layer {i}, output size was {x.size()}")
