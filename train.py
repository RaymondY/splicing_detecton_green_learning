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


def prepare_xgb_data(input_feature_set, target_feature_set, 
                     coarse_surface_prediction_set=[]):
    # coarse_surface_prediction_set is a list of numpy arrays
    if coarse_surface_prediction_set:
        for i in range(len(coarse_surface_prediction_set)):
            print(f"coarse_surface_prediction_set[{i}].shape: "
                    f"{coarse_surface_prediction_set[i].shape}")
            print(f"input_feature_set.shape: {input_feature_set.shape}")
            input_feature_set = np.concatenate(
                (input_feature_set, coarse_surface_prediction_set[i]), axis=1)
    gc.collect()
    
    sample_num, target_channel, height, width = target_feature_set.shape
    print(f"input feature shape: {input_feature_set.shape}")
    input_channel = input_feature_set.shape[1]
    print(f"sample_num: {sample_num}, target_channel: {target_channel}, "
            f"height: {height}, width: {width}, input_channel: {input_channel}")
    input_feature_set = np.transpose(input_feature_set, (0, 2, 3, 1))
    input_feature_set = input_feature_set.reshape(
        (sample_num * height * width, input_channel))
    target_feature_set = np.transpose(target_feature_set, (0, 2, 3, 1))
    target_feature_set = target_feature_set.reshape(
        (sample_num * height * width, target_channel))
    return input_feature_set, target_feature_set, \
        input_channel, target_channel, sample_num, height, width


def train_surface_representation(surface_set):
    surface_ssl_model = SurfaceSSLModel()
    surface_ssl_model.to(config.device)
    timer.register("surface model init")

    surface_ssl_model.fit(surface_set)
    timer.register("surface model fitted")

    # save model
    torch.save(surface_ssl_model, 
               os.path.join(config.model_dir, 
                            f'{pre_fix}_surface_ssl_model.pth'))
    timer.register("surface model saved")


def train_lab_representation(lab_set):
    lab_ssl_model = LABSSLModel()
    lab_ssl_model.to(config.device)
    timer.register("lab model init")

    lab_ssl_model.fit(lab_set)
    timer.register("lab model fitted")

    # save model
    torch.save(lab_ssl_model, 
               os.path.join(config.model_dir, 
                            f'{pre_fix}_lab_ssl_model.pth'))
    timer.register("lab model saved")


def save_all_features(lab_set, surface_set, 
                      test_lab_set,test_surface_set):
    # load models
    lab_ssl_model = torch.load(
        os.path.join(config.model_dir, f'{pre_fix}_lab_ssl_model.pth'))
    surface_ssl_model = torch.load(
        os.path.join(config.model_dir, f'{pre_fix}_surface_ssl_model.pth'))
    timer.register("models loaded")

    # save features
    lab_ssl_model(lab_set, need_save=True, name=f"{pre_fix}")
    timer.register("lab features saved")
    lab_ssl_model(test_lab_set, need_save=True, name=f"{pre_fix}_test")
    timer.register("test lab features saved")
    surface_ssl_model(surface_set, need_save=True, name=f"{pre_fix}_surface")
    timer.register("surface features saved")
    surface_ssl_model(test_surface_set, need_save=True, name=f"{pre_fix}_test_surface")
    timer.register("test surface features saved")


def train_surface_decision_level(lab_feature_level_set, 
                                 surface_feature_level_set, 
                                 test_lab_feature_level_set, 
                                 test_surface_feature_level_set, 
                                 level_index, 
                                 coarse_surface_prediction_set=[],
                                 test_coarse_surface_prediction_set=[]):
    input_feature_set, target_feature_set, \
        input_channel, target_channel, sample_num, height, width = prepare_xgb_data(
            lab_feature_level_set, surface_feature_level_set,
            coarse_surface_prediction_set)
    timer.register("feature set prepared")
    test_input_feature_set, test_target_feature_set, \
        _, _, test_sample_num, _, _ = prepare_xgb_data(
            test_lab_feature_level_set, test_surface_feature_level_set,
            test_coarse_surface_prediction_set)
    timer.register("test feature set prepared")
    # train xgboost
    surface_prediction_set = []
    test_surface_prediction_set = []
    for i in range(target_channel):
        surface_feature = target_feature_set[:, i].reshape(-1, 1)
        test_surface_feature = test_target_feature_set[:, i].reshape(-1, 1)
        # RFT
        if os.path.exists(os.path.join(config.model_dir, 
                                       f'{pre_fix}_surface_rft_level_{level_index}_{i}.pkl')):
            with open(os.path.join(config.model_dir, 
                                   f'{pre_fix}_surface_rft_level_{level_index}_{i}.pkl'), 'rb') as f:
                rft = pickle.load(f)
        else:
            rft = FeatureTest()
            rft.fit(input_feature_set, surface_feature, n_bins=32, outliers=True)
            plt.figure()
            plt.plot(np.arange(input_channel), np.array(list(rft.dim_loss.values())))
            plt.xlabel('Rank')
            plt.ylabel('MSE')
            # save
            plt.savefig(os.path.join(config.model_dir, 
                                     f'{pre_fix}_surface_rft_level_{level_index}_{i}.png'))
            plt.close()
            # save rft
            with open(os.path.join(config.model_dir, 
                                f'{pre_fix}_surface_rft_level_{level_index}_{i}.pkl'), 'wb') as f:
                pickle.dump(rft, f)
            timer.register("rft fitted")
        selected_num = config.selected_num_list[level_index - 1]
        selected_input_feature_set = rft.transform(input_feature_set, n_selected=selected_num)
        selected_test_input_feature_set = rft.transform(test_input_feature_set, 
                                                        n_selected=selected_num)
        # xgboost
        xgb_model = xgb.XGBRegressor(tree_method='gpu_hist', n_jobs=3, gpu_id=config.gpu_id,
                                     objective='reg:squarederror', 
                                     max_depth=6, n_estimators=2000, subsample=0.8, 
                                     early_stopping_rounds=20, learning_rate=0.15, 
                                     gamma=2, min_child_weight=2, 
                                     colsample_bytree=0.8)
        # if xgb json exists, load it
        if os.path.exists(os.path.join(config.model_dir, 
                                       f'{pre_fix}_surface_xgb_model_level_{level_index}_{i}.json')):
            xgb_model.load_model(os.path.join(
                config.model_dir, f'{pre_fix}_surface_xgb_model_level_{level_index}_{i}.json'))
            surface_prediction = xgb_model.predict(selected_input_feature_set)
            surface_prediction = surface_prediction.reshape(sample_num, height, width, 1)
            surface_prediction_set.append(surface_prediction)
            test_surface_prediction = xgb_model.predict(selected_test_input_feature_set)
            test_surface_prediction = test_surface_prediction.reshape(
                test_sample_num, height, width, 1)
            test_surface_prediction_set.append(test_surface_prediction)
            timer.register(f"xgb model level {level_index} channel {i} predicted")
            del xgb_model, selected_input_feature_set, selected_test_input_feature_set
            gc.collect()
            continue
        eval_set = [(selected_test_input_feature_set, test_surface_feature)]
        xgb_model.fit(selected_input_feature_set, surface_feature, eval_set=eval_set)
        results = xgb_model.evals_result()
        # save the learning curves
        plt.plot(results['validation_0']['rmse'], label='testing')
        plt.legend()
        plt.savefig(os.path.join(
            config.temp_dir, 
            f'{pre_fix}_learning_curve_surface_xgb_model_level_{level_index}_{i}.png'
            ))
        plt.close()
        xgb_model.save_model(os.path.join(
            config.model_dir, 
            f'{pre_fix}_surface_xgb_model_level_{level_index}_{i}.json'
            ))
        del results
        gc.collect()
        timer.register(f"xgb model level {level_index} channel {i} fitted")

        # predict
        surface_prediction = xgb_model.predict(selected_input_feature_set)
        surface_prediction = surface_prediction.reshape(sample_num, height, width, 1)
        surface_prediction_set.append(surface_prediction)

        test_surface_prediction = xgb_model.predict(selected_test_input_feature_set)
        test_surface_prediction = test_surface_prediction.reshape(
            test_sample_num, height, width, 1)
        test_surface_prediction_set.append(test_surface_prediction)

        del xgb_model, selected_input_feature_set, selected_test_input_feature_set
        gc.collect()
        timer.register(f"xgb model level {level_index} channel {i} predicted")

    del input_feature_set, surface_feature, \
        test_input_feature_set, test_surface_feature
    gc.collect()

    surface_prediction_set = np.concatenate(surface_prediction_set, axis=-1)
    surface_prediction_set = np.transpose(surface_prediction_set, (0, 3, 1, 2))
    test_surface_prediction_set = np.concatenate(test_surface_prediction_set, axis=-1)
    test_surface_prediction_set = np.transpose(test_surface_prediction_set, 
                                               (0, 3, 1, 2))
    timer.register(f"surface prediction set level {level_index} prepared")
    return surface_prediction_set, test_surface_prediction_set


def train_surface_decision_all():
    coarse_surface_prediction_set = []
    test_coarse_surface_prediction_set = []
    for level_index in range(config.level_num, 0, -1):
        # upsample the previous level coarse surface prediction
        for i in range(config.level_num - level_index):
            coarse_surface_prediction_set[i] = \
                upsample_x2_lanczos(coarse_surface_prediction_set[i])
            test_coarse_surface_prediction_set[i] = \
                upsample_x2_lanczos(test_coarse_surface_prediction_set[i])
        gc.collect()

        lab_feature_level_set = np.load(
            os.path.join(config.model_dir, f"{pre_fix}_lab_level_{level_index}.npy"))
        test_lab_feature_level_set = np.load(
            os.path.join(config.model_dir, f"{pre_fix}_test_lab_level_{level_index}.npy"))
        surface_feature_level_set = np.load(
            os.path.join(config.model_dir, f"{pre_fix}_surface_level_{level_index}.npy"))
        test_surface_feature_level_set = np.load(
            os.path.join(config.model_dir, f"{pre_fix}_test_surface_level_{level_index}.npy"))
        coarse_surface_prediction, test_coarse_surface_prediction = \
            train_surface_decision_level(lab_feature_level_set, 
                                         surface_feature_level_set,
                                         test_lab_feature_level_set, 
                                         test_surface_feature_level_set,
                                         level_index,
                                         coarse_surface_prediction_set, 
                                         test_coarse_surface_prediction_set)
        coarse_surface_prediction_set.append(deepcopy(coarse_surface_prediction))
        test_coarse_surface_prediction_set.append(deepcopy(test_coarse_surface_prediction))
        del lab_feature_level_set, test_lab_feature_level_set, \
            surface_feature_level_set, test_surface_feature_level_set, \
            coarse_surface_prediction, test_coarse_surface_prediction
        gc.collect()
        timer.register(f"surface decision level {level_index} trained")
   

if __name__ == '__main__':
    print(config.device)
    # # !First time run:
    # # load data
    # lab_set, surface_set, \
    #     test_lab_set, test_surface_set = \
    #         load_casia_v2_lab()
    # timer.register("data loaded")

    # # train lab representation
    # train_lab_representation(lab_set)
    # timer.register("lab model trained")

    # # train surface representation
    # train_surface_representation(surface_set)
    # timer.register("surface model trained")

    # # save all features
    # save_all_features(lab_set, surface_set, 
    #                   test_lab_set, test_surface_set)
    # timer.register("all features saved")


    # !Comment out the above code and run the following code 
    # after successfully running the above code once.
    # train surface decision
    train_surface_decision_all()

    # report timing
    timer.print()

    







