# Written by Tiankai Yang

import os
import gc
import pickle
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from matplotlib import pyplot as plt

from torch_ssl import *
from rft import *
from utils import *
from config import DefaultConfig

config = DefaultConfig()
timer = TimerClass(report_timing=True)
pre_fix = config.pre_fix
test_index = 0


def prepare_xgb_data(input_feature_set, coarse_surface_prediction_set=[]):
    # coarse_surface_prediction_set is a list of numpy arrays
    if coarse_surface_prediction_set:
        for i in range(len(coarse_surface_prediction_set)):
            print(f"coarse_surface_prediction_set[{i}].shape: "
                    f"{coarse_surface_prediction_set[i].shape}")
            print(f"input_feature_set.shape: {input_feature_set.shape}")
            input_feature_set = np.concatenate(
                (input_feature_set, coarse_surface_prediction_set[i]), axis=1)
    gc.collect()
    
    sample_num, input_channel, height, width = input_feature_set.shape
    print(f"sample_num: {sample_num}, height: {height}, width: {width}, "
          f"input_channel: {input_channel}")
    input_feature_set = np.transpose(input_feature_set, (0, 2, 3, 1))
    input_feature_set = input_feature_set.reshape(
        (sample_num * height * width, input_channel))
    return input_feature_set, sample_num, height, width


def predict_surface_level(lab_feature_level_set, level_index, 
                                   coarse_surface_prediction_set=[]):
    input_feature_set, sample_num, height, width = \
        prepare_xgb_data(lab_feature_level_set, coarse_surface_prediction_set)
    timer.register("feature set prepared")
    
    # xgboost predict
    surface_prediction_set = []
    for i in range(config.target_channel):
        # RFT
        with open(os.path.join(config.model_dir, 
                               f'{pre_fix}_surface_rft_level_{level_index}_{i}.pkl'), 'rb'
                               ) as f:
            rft = pickle.load(f)
        selected_num = config.selected_num_list[level_index - 1]
        selected_input_feature_set = rft.transform(input_feature_set, n_selected=selected_num)
        timer.register("rft transformed")

        # xgboost
        xgb_model = xgb.XGBRegressor(tree_method='gpu_hist', n_jobs=3, gpu_id=config.gpu_id,
                                     objective='reg:squarederror', 
                                     max_depth=6, n_estimators=2000, subsample=0.8, 
                                     early_stopping_rounds=20, learning_rate=0.15, 
                                     gamma=2, min_child_weight=2, 
                                     colsample_bytree=0.8)
        
        xgb_model.load_model(os.path.join(
            config.model_dir, 
            f'{pre_fix}_surface_xgb_model_level_{level_index}_{i}.json'))
        surface_prediction = xgb_model.predict(selected_input_feature_set)
        surface_prediction = surface_prediction.reshape(sample_num, height, width, 1)
        surface_prediction_set.append(surface_prediction)
        del rft, xgb_model, selected_input_feature_set
        gc.collect()
        timer.register(f"xgb model level {level_index} channel {i} predicted")
        
    del input_feature_set
    gc.collect()

    surface_prediction_set = np.concatenate(surface_prediction_set, axis=-1)
    surface_prediction_set = np.transpose(surface_prediction_set, (0, 3, 1, 2))
    timer.register(f"surface prediction set level {level_index} prepared")

    # save intermediate results
    for i in range(config.target_channel):
        surface_prediction = surface_prediction_set[test_index, i, :, :]
        surface_prediction = np.squeeze(surface_prediction)
        # save .txt
        np.savetxt(os.path.join(config.temp_dir, 
                                f'{pre_fix}_before_upsampling_surface_prediction_{level_index}_{i}.txt'), 
                                surface_prediction, fmt='%.2f')
        # min-max normalization
        surface_prediction = (surface_prediction - np.min(surface_prediction)) \
            / (np.max(surface_prediction) - np.min(surface_prediction)) * 255
        surface_prediction = surface_prediction.astype(np.uint8)
        surface = Image.fromarray(surface_prediction, mode='L')
        surface.save(os.path.join(config.temp_dir, 
                                  f'{pre_fix}_before_upsampling_surface_prediction_{level_index}_{i}.png'))
        timer.register(f"surface prediction level {level_index} channel {i} saved")

    return surface_prediction_set


def predict_surface_all(lab_set, surface_set=None):
    # load models
    lab_ssl_model = torch.load(
        os.path.join(config.model_dir, f'{pre_fix}_lab_ssl_model.pth'))
    surface_ssl_model = torch.load(
        os.path.join(config.model_dir, f'{pre_fix}_surface_ssl_model.pth'))
    timer.register("models loaded")

    # code below is just for visualization of each level's targets
    # comment it in final experiments
    surface_feature_set = surface_ssl_model(surface_set, [1, 2, 3, 4, 5])
    timer.register("surface feature set extracted")
    for level_index in range(config.level_num, 0, -1):
        surface_feature_level_set = surface_feature_set[level_index - 1]
        for i in range(config.target_channel):
            surface_feature = surface_feature_level_set[test_index, i, :, :]
            surface_feature = np.squeeze(surface_feature)
            # save .txt
            np.savetxt(os.path.join(config.temp_dir,
                                    f'{pre_fix}_surface_feature_level_{level_index}_{i}.txt'),
                                    surface_feature, fmt='%.2f')
            # min-max normalization
            surface_feature = (surface_feature - np.min(surface_feature)) \
                / (np.max(surface_feature) - np.min(surface_feature)) * 255
            surface_feature = surface_feature.astype(np.uint8)
            surface = Image.fromarray(surface_feature, mode='L')
            surface.save(os.path.join(config.temp_dir,
                                        f'{pre_fix}_surface_feature_level_{level_index}_{i}.png'))
            timer.register(f"surface feature level {level_index} channel {i} saved")

    # get features
    lab_feature_set = lab_ssl_model(lab_set, [1, 2, 3, 4, 5])
    timer.register("lab feature set extracted")

    coarse_surface_prediction_set = []
    test_coarse_surface_prediction_set = []
    for level_index in range(config.level_num, 0, -1):
        # !记得删除RFT文件
        # # upsample the previous level coarse surface prediction
        # for i in range(config.level_num - level_index):
        #     coarse_surface_prediction_set[i] = \
        #         upsample_x2_lanczos(coarse_surface_prediction_set[i])
        #     test_coarse_surface_prediction_set[i] = \
        #         upsample_x2_lanczos(test_coarse_surface_prediction_set[i])
        # gc.collect()

        lab_feature_level_set = lab_feature_set[level_index - 1]
        surface_prediction_set = \
            predict_surface_level(lab_feature_level_set, 
                                  level_index, 
                                  coarse_surface_prediction_set)
        
        # coarse_surface_prediction_set.append(deepcopy(surface_prediction_set))

        del lab_feature_level_set, surface_prediction_set
        gc.collect()
        timer.register(f"level {level_index} predicted")

    # !if surface_set is not none, predict final probability map and do evaluation
    # refer to the code in predict_yuv.py


if __name__ == '__main__':
    print(config.device)
    # load data (if provide index for load function, only load the specified data)
    # index could be a list or a number or just one number
    # test_index = 0 means only save the first sample's intermediate results
    lab_set, surface_set = load_casia_v2_lab(4)
    timer.register("data loaded")

    # predict surface
    predict_surface_all(lab_set, surface_set)
