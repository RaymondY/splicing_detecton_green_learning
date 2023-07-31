# Written by Tiankai Yang

import os
import gc
import pickle
import numpy as np
from PIL import Image
import torch
import xgboost as xgb
from sklearn.metrics import f1_score, matthews_corrcoef

from torch_ssl import *
from rft import *
from utils import *
from config import DefaultConfig

config = DefaultConfig()
timer = TimerClass()
pre_fix = "33"


def prepare_xgb_data(input_feature_set, coarse_surface_prediction=None):
    if coarse_surface_prediction is not None:
        input_feature_set = np.concatenate(
            (input_feature_set, coarse_surface_prediction), axis=1)
    
    sample_num, input_channel, height, width = input_feature_set.shape
    print(f"sample_num: {sample_num}, height: {height}, \
          width: {width}, input_channel: {input_channel}")
    input_feature_set = np.transpose(input_feature_set, (0, 2, 3, 1))
    input_feature_set = input_feature_set.reshape(
        (sample_num * height * width, input_channel))
    return input_feature_set, input_channel, sample_num, height, width


def predict_surface_decision_level(yuv_feature_level_set, level_index, 
                                   coarse_surface_prediction=None,):
    test_index = 0
    input_feature_set = yuv_feature_level_set
    print(f"input_feature_set.shape: {input_feature_set.shape}")
    if level_index == 4:
        test_feature = input_feature_set[test_index, 0, :, :]
        test_feature = np.squeeze(test_feature)
        print("test feature shape: ", test_feature.shape)
        # print(type(test_feature))
        # print(np.max(test_feature), np.min(test_feature))
        # save .txt
        np.savetxt(os.path.join(config.temp_dir, '{pre_fix}_test_feature.txt'), test_feature)
        # convert to 0-255 by min-max normalization
        test_feature = (test_feature - np.min(test_feature)) \
            / (np.max(test_feature) - np.min(test_feature)) * 255
        test_feature = test_feature.astype(np.uint8)
        test_feature = Image.fromarray(test_feature, mode='L')
        test_feature.save(os.path.join(config.temp_dir, '{pre_fix}_test_feature.png'))
        timer.register("test feature saved")

    # prepare data
    input_feature_set, input_channel, sample_num, height, width = \
        prepare_xgb_data(input_feature_set, coarse_surface_prediction)
    timer.register("feature set prepared")

    # predict
    surface_prediction_set = []
    for i in range(config.target_channel):
        # load RFT
        with open(os.path.join(config.model_dir, 
                               f'{pre_fix}_surface_rft_level_{level_index}_{i}.pkl'), 'rb') as f:
            rft = pickle.load(f)
        selected_num = config.selected_num_list[level_index - 1]
        selected_input_feature_set = rft.transform(input_feature_set, 
                                                   n_selected=selected_num)

        # xgb_model = xgb.XGBRegressor(tree_method='gpu_hist', n_jobs=3, gpu_id=config.gpu_id,
        #                              objective='reg:squarederror', 
        #                              max_depth=6, n_estimators=1500, subsample=0.8, 
        #                              early_stopping_rounds=15, 
        #                              learning_rate=0.05, gamma=4, min_child_weight=4, 
        #                              colsample_bytree=0.8)
        # xgb_model = xgb.XGBRegressor(tree_method='gpu_hist', n_jobs=3, gpu_id=config.gpu_id,
        #                              objective='reg:squarederror', 
        #                              max_depth=6, n_estimators=1500, subsample=0.8, 
        #                              early_stopping_rounds=20, learning_rate=0.1, 
        #                              colsample_bytree=0.8)
        xgb_model = xgb.XGBRegressor(tree_method='gpu_hist', n_jobs=3, gpu_id=0,
                                     objective='reg:squarederror', 
                                     max_depth=6, n_estimators=2000, subsample=0.8, 
                                     early_stopping_rounds=20, learning_rate=0.15, 
                                     gamma=2, min_child_weight=2, 
                                     colsample_bytree=0.8)
        
        # if xgb json exists, load it
        xgb_model.load_model(os.path.join(config.model_dir, 
                                          f'{pre_fix}_surface_xgb_model_level_{level_index}_{i}.json'))
        surface_prediction = xgb_model.predict(selected_input_feature_set)
        surface_prediction = surface_prediction.reshape(sample_num, height, width, 1)
        surface_prediction_set.append(surface_prediction)
        timer.register(f"xgb model level {level_index} channel {i} predicted")
        del rft, selected_input_feature_set, xgb_model
        gc.collect()
        
    del input_feature_set
    gc.collect()

    surface_prediction_set = np.concatenate(surface_prediction_set, axis=-1)
    surface_prediction_set = np.transpose(surface_prediction_set, (0, 3, 1, 2))
    timer.register(f"surface prediction set level {level_index} prepared")

    for i in range(config.target_channel):
        surface_prediction = surface_prediction_set[test_index, i, :, :]
        surface_prediction = np.squeeze(surface_prediction)
        # save .txt
        np.savetxt(os.path.join(config.temp_dir, 
                                f'{pre_fix}_before_surface_prediction_{level_index}_{i}.txt'), 
                                surface_prediction, fmt='%.2f')
        if level_index == 1 and i == 0:
            surface_prediction = surface_prediction * 255
            # clip to 0-255
            surface_prediction = np.clip(surface_prediction, 0, 255)
        else:
            # min-max normalization
            surface_prediction = (surface_prediction - np.min(surface_prediction)) \
                / (np.max(surface_prediction) - np.min(surface_prediction)) * 255
        surface_prediction = surface_prediction.astype(np.uint8)
        surface = Image.fromarray(surface_prediction, mode='L')
        surface.save(os.path.join(config.temp_dir, 
                                  f'{pre_fix}_before_surface_prediction_{level_index}_{i}.png'))
        timer.register(f"surface prediction level {level_index} channel {i} saved")

    return surface_prediction_set


def predict_final_map(coarse_surface_prediction):
    test_index = 0
    input_feature_set, input_channel, sample_num, height, width = \
        prepare_xgb_data(coarse_surface_prediction)
    timer.register("feature set prepared")
    # load xgb
    xgb_model = xgb.XGBRegressor(tree_method='gpu_hist', n_jobs=3, gpu_id=0,
                                 objective='reg:squarederror', 
                                 max_depth=6, n_estimators=2000, subsample=0.8, 
                                 early_stopping_rounds=20, learning_rate=0.15, 
                                 gamma=2, min_child_weight=2, 
                                 colsample_bytree=0.8)
    xgb_model.load_model(os.path.join(config.model_dir,
                                        f'{pre_fix}_surface_xgb_model_final.json'))
    final_map = xgb_model.predict(input_feature_set)
    # clip to 0-1
    final_map = np.clip(final_map, 0, 1)
    final_map = final_map.reshape(sample_num, height, width, 1)
    final_map = np.transpose(final_map, (0, 3, 1, 2))
    timer.register("final map predicted")
    # save txt
    test_final_map = final_map[test_index, :, :, :]
    test_final_map = np.squeeze(test_final_map)
    np.savetxt(os.path.join(config.temp_dir, f'{pre_fix}_test_final_map.txt'),
                test_final_map, fmt='%.2f')
    test_final_map = test_final_map * 255
    # clip to 0-255
    test_final_map = np.clip(test_final_map, 0, 255)
    test_final_map = test_final_map.astype(np.uint8)
    test_final_map = Image.fromarray(test_final_map, mode='L')
    test_final_map.save(os.path.join(config.temp_dir, f'{pre_fix}_test_final_map.png'))
    timer.register("test final map saved")

    del input_feature_set, xgb_model
    gc.collect()
    return final_map


def evaluate_surface_prediction(surface_prediction, surface_set, threshold=0.4):
    # turn to binary
    surface_prediction = np.where(surface_prediction > threshold, 1, 0)
    # flatten
    surface_prediction = surface_prediction.flatten()
    surface_set = surface_set.flatten()
    # f1
    f1 = f1_score(surface_set, surface_prediction)
    # mcc
    mcc = matthews_corrcoef(surface_set, surface_prediction)
    return f1, mcc



def predict_surface_all(y_set, u_set, v_set, surface_set=None):
    test_index = 0
    # load models
    y_ssl_model = torch.load(
        os.path.join(config.model_dir, f'{pre_fix}_y_ssl_model.pth'))
    y_ssl_model.to(config.device)
    u_ssl_model = torch.load(
        os.path.join(config.model_dir, f'{pre_fix}_u_ssl_model.pth'))
    u_ssl_model.to(config.device)
    v_ssl_model = torch.load(
        os.path.join(config.model_dir, f'{pre_fix}_v_ssl_model.pth'))
    v_ssl_model.to(config.device)
    surface_ssl_model = torch.load(
        os.path.join(config.model_dir, f'{pre_fix}_surface_ssl_model.pth'))
    surface_ssl_model.to(config.device)
    timer.register("models loaded")

    # get features
    y_feature_set = y_ssl_model(y_set, [1, 2, 3, 4])
    u_feature_set = u_ssl_model(u_set, [1, 2, 3, 4])
    v_feature_set = v_ssl_model(v_set, [1, 2, 3, 4])
    yuv_feature_set = []
    # np.concat
    for i in range(4):
        yuv_feature_temp = np.concatenate(
            (y_feature_set[i], u_feature_set[i], v_feature_set[i]), axis=1)
        yuv_feature_set.append(yuv_feature_temp)
    timer.register("yuv feature set extracted")

    # 4, 3, 2, 1
    coarse_surface_prediction = None
    for level_index in range(4, 0, -1):
        # get features
        yuv_feature_level_set = yuv_feature_set[level_index - 1]
        coarse_surface_prediction = \
            predict_surface_decision_level(
                yuv_feature_level_set, level_index, 
                coarse_surface_prediction)
        coarse_surface_prediction = upsample_x2_lanczos(coarse_surface_prediction)
        # print(coarse_surface_prediction)
        # coarse_surface_prediction = \
        #     surface_ssl_model.level_inverse(coarse_surface_prediction, level_index)
        gc.collect()
        timer.register(f"level {level_index} finished")
        for i in range(coarse_surface_prediction.shape[1]):
            surface_prediction = coarse_surface_prediction[test_index, i, :, :]
            surface_prediction = np.squeeze(surface_prediction)
            # save .txt
            np.savetxt(os.path.join(config.temp_dir, 
                                    f'{pre_fix}_surface_prediction_{level_index}_{i}.txt'), 
                                    surface_prediction, fmt='%.2f')
            if level_index == 1 and i == 0:
                surface_prediction = surface_prediction * 255
                # clip to 0-255
                surface_prediction = np.clip(surface_prediction, 0, 255)
            else:
                # min-max normalization
                surface_prediction = (surface_prediction - np.min(surface_prediction)) \
                    / (np.max(surface_prediction) - np.min(surface_prediction)) * 255
            surface_prediction = surface_prediction.astype(np.uint8)
            surface = Image.fromarray(surface_prediction, mode='L')
            surface.save(os.path.join(config.temp_dir, 
                                      f'{pre_fix}_surface_prediction_{level_index}_{i}.png'))
        timer.register("surface prediction set prepared")
        # break
    
    # predict final surface
    final_map = predict_final_map(coarse_surface_prediction)
    if surface_set is not None:
        # evaluate
        f1, mcc = evaluate_surface_prediction(final_map, surface_set)
        print(f"f1: {f1}, mcc: {mcc}")
        timer.register("final map predicted")


if __name__ == '__main__':
    print(config.device)

    # # load data
    # y_set, u_set, v_set, surface_set = load_casia_v2_yuv(4)
    # timer.register("data loaded")
    # # predict surface
    # predict_surface_all(y_set, u_set, v_set)




    # # load data
    # y_set, u_set, v_set, surface_set, \
    #     test_y_set, test_u_set, test_v_set, test_surface_set = \
    #         load_casia_v2_yuv()
    # timer.register("data loaded")
    
    # del y_set, u_set, v_set, surface_set
    # gc.collect()

    # # predict surface
    # print(test_y_set.shape, test_u_set.shape, test_v_set.shape, test_surface_set.shape)
    # predict_surface_all(test_y_set, test_u_set, test_v_set, test_surface_set)
    # timer.register("test data predicted")

    # timer.print()

    
    # 
    y_set, u_set, v_set, surface_set = load_columbia_yuv()
    timer.register("data loaded")
    # predict surface
    predict_surface_all(y_set, u_set, v_set, surface_set)
    timer.register("data predicted")
