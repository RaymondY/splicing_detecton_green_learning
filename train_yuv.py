# Written by Tiankai Yang

import os
import gc
import pickle
import numpy as np
from PIL import Image
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
pre_fix = "gl"


def prepare_xgb_data(input_feature_set, target_feature_set, 
                     coarse_surface_prediction=None):
    if coarse_surface_prediction is not None:
        input_feature_set = np.concatenate(
            (input_feature_set, coarse_surface_prediction), axis=1)
    
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
               os.path.join(config.model_dir, f'{pre_fix}_surface_ssl_model.pth'))
    timer.register("surface model saved")


def train_yuv_representation(y_set, u_set, v_set):
    y_ssl_model = YUVSSLModel()
    y_ssl_model.to(config.device)
    u_ssl_model = YUVSSLModel()
    u_ssl_model.to(config.device)
    v_ssl_model = YUVSSLModel()
    v_ssl_model.to(config.device)
    timer.register("yuv model init")

    y_ssl_model.fit(y_set)
    u_ssl_model.fit(u_set)
    v_ssl_model.fit(v_set)
    timer.register("yuv model fitted")

    torch.save(y_ssl_model,
                os.path.join(config.model_dir, f'{pre_fix}_y_ssl_model.pth'))
    torch.save(u_ssl_model,
                os.path.join(config.model_dir, f'{pre_fix}_u_ssl_model.pth'))
    torch.save(v_ssl_model,
                os.path.join(config.model_dir, f'{pre_fix}_v_ssl_model.pth'))
    timer.register("yuv model saved")


def train_surface_decision_level(yuv_feature_level_set, surface_feature_level_set, 
                                 test_yuv_feature_level_set, 
                                 test_surface_feature_level_set, 
                                 level_index, coarse_surface_prediction=None,
                                 test_coarse_surface_prediction=None):
    input_feature_set, target_feature_set, \
        input_channel, target_channel, sample_num, height, width = prepare_xgb_data(
            yuv_feature_level_set, surface_feature_level_set,
            coarse_surface_prediction)
    timer.register("feature set prepared")
    test_input_feature_set, test_target_feature_set, \
        _, _, test_sample_num, _, _ = prepare_xgb_data(
            test_yuv_feature_level_set, test_surface_feature_level_set,
            test_coarse_surface_prediction)
    timer.register("test feature set prepared")
    # train xgboost
    surface_prediction_set = []
    test_surface_prediction_set = []
    for i in range(target_channel):
        surface_feature = target_feature_set[:, i].reshape(-1, 1)
        test_surface_feature = test_target_feature_set[:, i].reshape(-1, 1)
        # if level_index == 1:
        #     # subsample
        #     subsample_index = np.random.choice(
        #         np.arange(sample_num * height * width),
        #         size=int(sample_num * height * width * 0.5), replace=False)
        #     input_feature_set = input_feature_set[subsample_index, :]
        #     surface_feature = surface_feature[subsample_index, :]
        #     gc.collect()
        # RFT
        if os.path.exists(os.path.join(config.model_dir, f'{pre_fix}_surface_rft_level_{level_index}_{i}.pkl')):
            with open(os.path.join(config.model_dir, f'{pre_fix}_surface_rft_level_{level_index}_{i}.pkl'), 'rb') as f:
                rft = pickle.load(f)
        else:
            rft = FeatureTest()
            rft.fit(input_feature_set, surface_feature, n_bins=32, outliers=True)
            plt.figure()
            plt.plot(np.arange(input_channel), np.array(list(rft.dim_loss.values())))
            plt.xlabel('Rank')
            plt.ylabel('MSE')
            # save
            plt.savefig(os.path.join(config.model_dir, f'{pre_fix}_surface_rft_level_{level_index}_{i}.png'))
            plt.close()
            # save rft
            with open(os.path.join(config.model_dir, 
                                f'{pre_fix}_surface_rft_level_{level_index}_{i}.pkl'), 'wb') as f:
                pickle.dump(rft, f)
            timer.register("rft fitted")
        selected_num = config.selected_num_list[level_index - 1]
        selected_input_feature_set = rft.transform(input_feature_set, n_selected=selected_num)
        selected_test_input_feature_set = rft.transform(test_input_feature_set, n_selected=selected_num)
        
        print(f"selected_input_feature_set.shape: {selected_input_feature_set.shape}")
        print(f"selected_test_input_feature_set.shape: {selected_test_input_feature_set.shape}")
        timer.register("rft transformed")

        # xgb
        # xgb_model = xgb.XGBRegressor(tree_method='gpu_hist', n_jobs=3, gpu_id=config.gpu_id,
        #                              objective='reg:squarederror', 
        #                              max_depth=6, n_estimators=1500, subsample=0.8, 
        #                              early_stopping_rounds=20, learning_rate=0.1, 
        #                              colsample_bytree=0.8)
        xgb_model = xgb.XGBRegressor(tree_method='gpu_hist', n_jobs=3, gpu_id=config.gpu_id,
                                     objective='reg:squarederror', 
                                     max_depth=6, n_estimators=2000, subsample=0.8, 
                                     early_stopping_rounds=20, learning_rate=0.15, 
                                     gamma=2, min_child_weight=2, 
                                     colsample_bytree=0.8)
        # if xgb json exists, load it
        if os.path.exists(os.path.join(config.model_dir, f'{pre_fix}_surface_xgb_model_level_{level_index}_{i}.json')):
            xgb_model.load_model(os.path.join(config.model_dir, f'{pre_fix}_surface_xgb_model_level_{level_index}_{i}.json'))
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

        timer.register(f"xgb model level {level_index} channel {i} predicted")
        del xgb_model, selected_input_feature_set, selected_test_input_feature_set
        gc.collect()

    del input_feature_set, test_input_feature_set, \
        target_feature_set, test_target_feature_set
    gc.collect()

    surface_prediction_set = np.concatenate(surface_prediction_set, axis=-1)
    surface_prediction_set = np.transpose(surface_prediction_set, (0, 3, 1, 2))
    test_surface_prediction_set = np.concatenate(test_surface_prediction_set, axis=-1)
    test_surface_prediction_set = np.transpose(test_surface_prediction_set, (0, 3, 1, 2))
    timer.register(f"surface prediction set level {level_index} prepared")
    return surface_prediction_set, test_surface_prediction_set


def train_final_map(coarse_surface_prediction, surface_set,
                    test_coarse_surface_prediction, test_surface_set):
    input_feature_set, target_feature_set, \
        input_channel, target_channel, sample_num, height, width = prepare_xgb_data(
            coarse_surface_prediction, surface_set)
    timer.register("feature set prepared")
    test_input_feature_set, test_target_feature_set, \
        _, _, test_sample_num, _, _ = prepare_xgb_data(
            test_coarse_surface_prediction, test_surface_set)
    timer.register("test feature set prepared")
    assert target_channel == 1
    # train xgboost
    surface_set = surface_set.reshape(-1, 1)
    test_surface_set = test_surface_set.reshape(-1, 1)
    xgb_model = xgb.XGBRegressor(tree_method='gpu_hist', n_jobs=3, gpu_id=0,
                                 objective='reg:squarederror', 
                                 max_depth=6, n_estimators=2000, subsample=0.8, 
                                 early_stopping_rounds=20, learning_rate=0.15, 
                                 gamma=2, min_child_weight=2, 
                                 colsample_bytree=0.8)
    # if xgb json exists, load it
    if os.path.exists(os.path.join(config.model_dir, f'{pre_fix}_surface_xgb_model_final.json')):
        xgb_model.load_model(os.path.join(config.model_dir, f'{pre_fix}_surface_xgb_model_final.json'))
        surface_prediction = xgb_model.predict(input_feature_set)
        surface_prediction = surface_prediction.reshape(sample_num, height, width, 1)
        test_surface_prediction = xgb_model.predict(test_input_feature_set)
        test_surface_prediction = test_surface_prediction.reshape(
            test_sample_num, height, width, 1)
        timer.register(f"xgb model final predicted")
        return surface_prediction, test_surface_prediction
    eval_set = [(test_input_feature_set, test_surface_set)]
    xgb_model.fit(input_feature_set, surface_set, eval_set=eval_set)
    results = xgb_model.evals_result()
    # save the learning curves
    plt.plot(results['validation_0']['rmse'], label='testing')
    plt.legend()
    plt.savefig(os.path.join(
        config.temp_dir, 
        f'{pre_fix}_learning_curve_surface_xgb_model_final.png'
        ))
    plt.close()
    xgb_model.save_model(os.path.join(
        config.model_dir, 
        f'{pre_fix}_surface_xgb_model_final.json'
        ))
    del results
    gc.collect()
    timer.register(f"xgb model final fitted")

    del input_feature_set, test_input_feature_set, \
        target_feature_set, test_target_feature_set, \
        surface_set, test_surface_set, xgb_model
    
    gc.collect()
    timer.register(f"final map trained")
        

# def train_surface_decision_all(y_set, u_set, v_set, surface_set, 
#                                test_y_set, test_u_set, test_v_set, 
#                                test_surface_set):
def train_surface_decision_all(surface_set, test_surface_set):
    # load models
    # y_ssl_model = torch.load(
    #     os.path.join(config.model_dir, f'{pre_fix}_y_ssl_model.pth'))
    # y_ssl_model.to(config.device)
    # u_ssl_model = torch.load(
    #     os.path.join(config.model_dir, f'{pre_fix}_u_ssl_model.pth'))
    # u_ssl_model.to(config.device)
    # v_ssl_model = torch.load(
    #     os.path.join(config.model_dir, f'{pre_fix}_v_ssl_model.pth'))
    # v_ssl_model.to(config.device)
    surface_ssl_model = torch.load(
        os.path.join(config.model_dir, f'{pre_fix}_surface_ssl_model.pth'))
    surface_ssl_model.to(config.device)
    timer.register("models loaded")

    # get features
    # y_feature_set = y_ssl_model(y_set, [1, 2, 3, 4])
    # u_feature_set = u_ssl_model(u_set, [1, 2, 3, 4])
    # v_feature_set = v_ssl_model(v_set, [1, 2, 3, 4])
    # test_y_feature_set = y_ssl_model(test_y_set, [1, 2, 3, 4])
    # test_u_feature_set = u_ssl_model(test_u_set, [1, 2, 3, 4])
    # test_v_feature_set = v_ssl_model(test_v_set, [1, 2, 3, 4])
    # yuv_feature_set = []
    # test_yuv_feature_set = []
    # # np.concat
    # for i in range(4):
    #     yuv_feature_temp = np.concatenate(
    #         (y_feature_set[i], u_feature_set[i], v_feature_set[i]), axis=1)
    #     yuv_feature_set.append(yuv_feature_temp)
    #     test_yuv_feature_temp = np.concatenate(
    #         (test_y_feature_set[i], test_u_feature_set[i], test_v_feature_set[i]), 
    #         axis=1)
    #     test_yuv_feature_set.append(test_yuv_feature_temp)
    # timer.register("yuv feature set extracted")
    # surface_feature_set = surface_ssl_model(surface_set, [1, 2, 3, 4])
    # test_surface_feature_set = surface_ssl_model(test_surface_set, [1, 2, 3, 4])
    # timer.register("surface feature set extracted")

    # 4, 3, 2, 1
    coarse_surface_prediction = None
    test_coarse_surface_prediction = None
    for level_index in range(4, 0, -1):
        # y_feature_set = y_ssl_model(y_set, [level_index])
        # gc.collect()
        # u_feature_set = u_ssl_model(u_set, [level_index])
        # gc.collect()
        # v_feature_set = v_ssl_model(v_set, [level_index])
        # gc.collect()
        # test_y_feature_set = y_ssl_model(test_y_set, [level_index])
        # gc.collect()
        # test_u_feature_set = u_ssl_model(test_u_set, [level_index])
        # gc.collect()
        # test_v_feature_set = v_ssl_model(test_v_set, [level_index])
        # gc.collect()
        
        # read from file "33_{name}_level_{i+1}.txt"
        # y_feature_set = np.load(
        #     os.path.join(config.model_dir, f"{pre_fix}_y_level_{level_index}.npy"))
        # u_feature_set = np.load(
        #     os.path.join(config.model_dir, f"{pre_fix}_u_level_{level_index}.npy"))
        # v_feature_set = np.load(
        #     os.path.join(config.model_dir, f"{pre_fix}_v_level_{level_index}.npy"))
        # yuv_feature_level_set = np.concatenate(
        #     (y_feature_set, u_feature_set, v_feature_set), axis=1)
        # del y_feature_set, u_feature_set, v_feature_set
        # gc.collect()
        
        # test_y_feature_set = np.load(
        #     os.path.join(config.model_dir, f"{pre_fix}_test_y_level_{level_index}.npy"))
        # test_u_feature_set = np.load(
        #     os.path.join(config.model_dir, f"{pre_fix}_test_u_level_{level_index}.npy"))
        # test_v_feature_set = np.load(
        #     os.path.join(config.model_dir, f"{pre_fix}_test_v_level_{level_index}.npy"))
        # test_yuv_feature_level_set = np.concatenate(
        #     (test_y_feature_set, test_u_feature_set, test_v_feature_set), 
        #     axis=1)
        # del test_y_feature_set, test_u_feature_set, test_v_feature_set
        # gc.collect()
        # timer.register("yuv feature set extracted")

        yuv_feature_level_set = np.load(
            os.path.join(config.model_dir, f"{pre_fix}_yuv_level_{level_index}.npy"))
        test_yuv_feature_level_set = np.load(
            os.path.join(config.model_dir, f"{pre_fix}_test_yuv_level_{level_index}.npy"))

        surface_feature_level_set = surface_ssl_model(surface_set, [level_index])
        gc.collect()
        test_surface_feature_level_set = \
            surface_ssl_model(test_surface_set, [level_index])
        gc.collect()
        if level_index == 1:
            # subsample
            subsample_index = np.random.choice(
                np.arange(yuv_feature_level_set.shape[0]),
                size=2000, replace=False)
            yuv_feature_level_set = yuv_feature_level_set[subsample_index, :]
            surface_feature_level_set = surface_feature_level_set[subsample_index, :]
            coarse_surface_prediction = coarse_surface_prediction[subsample_index, :]
            surface_set = surface_set[subsample_index, :]
            gc.collect()
        timer.register("surface feature set extracted")

        # # get features
        # yuv_feature_level_set = yuv_feature_set[level_index - 1]
        # test_yuv_feature_level_set = test_yuv_feature_set[level_index - 1]
        # surface_feature_level_set = surface_feature_set[level_index - 1]
        # test_surface_feature_level_set = test_surface_feature_set[level_index - 1]
        coarse_surface_prediction, test_coarse_surface_prediction = \
            train_surface_decision_level(
                yuv_feature_level_set, surface_feature_level_set, 
                test_yuv_feature_level_set, test_surface_feature_level_set,
                level_index, coarse_surface_prediction, 
                test_coarse_surface_prediction)
        coarse_surface_prediction = upsample_x2_lanczos(coarse_surface_prediction)
        test_coarse_surface_prediction = \
            upsample_x2_lanczos(test_coarse_surface_prediction)
        # coarse_surface_prediction = \
        #     surface_ssl_model.level_inverse(coarse_surface_prediction, level_index)
        # test_coarse_surface_prediction = \
        #     surface_ssl_model.level_inverse(test_coarse_surface_prediction, level_index)
        # gc.collect()
        timer.register(f"level {level_index} trained")
        del yuv_feature_level_set, test_yuv_feature_level_set, \
            surface_feature_level_set, test_surface_feature_level_set
        gc.collect()
        # break

    # del coarse_surface_prediction
    # gc.collect()
    # from predict import predict_surface_decision_level
    # coarse_surface_prediction = predict_surface_decision_level(
    #     yuv_feature_level_set, surface_feature_level_set,

    del surface_ssl_model
    gc.collect()
    # train final map
    train_final_map(coarse_surface_prediction, surface_set,
                    test_coarse_surface_prediction, test_surface_set)
    timer.register("final map trained")


def save_all_features(y_set, u_set, v_set, 
                      test_y_set, test_u_set, test_v_set):
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

    # save features
    y_ssl_model(y_set, need_save=True, name="y")
    u_ssl_model(u_set, need_save=True, name="u")
    v_ssl_model(v_set, need_save=True, name="v")
    y_ssl_model(test_y_set, need_save=True, name="test_y")
    u_ssl_model(test_u_set, need_save=True, name="test_u")
    v_ssl_model(test_v_set, need_save=True, name="test_v")


def combine_all_features():
    for level_index in range(4, 0, -1):
        y_feature_set = np.load(
            os.path.join(config.model_dir, f"{pre_fix}_y_level_{level_index}.npy"))
        u_feature_set = np.load(
            os.path.join(config.model_dir, f"{pre_fix}_u_level_{level_index}.npy"))
        v_feature_set = np.load(
            os.path.join(config.model_dir, f"{pre_fix}_v_level_{level_index}.npy"))
        yuv_feature_level_set = np.concatenate(
            (y_feature_set, u_feature_set, v_feature_set), axis=1)
        del y_feature_set, u_feature_set, v_feature_set
        gc.collect()
        
        test_y_feature_set = np.load(
            os.path.join(config.model_dir, f"{pre_fix}_test_y_level_{level_index}.npy"))
        test_u_feature_set = np.load(
            os.path.join(config.model_dir, f"{pre_fix}_test_u_level_{level_index}.npy"))
        test_v_feature_set = np.load(
            os.path.join(config.model_dir, f"{pre_fix}_test_v_level_{level_index}.npy"))
        test_yuv_feature_level_set = np.concatenate(
            (test_y_feature_set, test_u_feature_set, test_v_feature_set), 
            axis=1)
        del test_y_feature_set, test_u_feature_set, test_v_feature_set
        gc.collect()
        timer.register("yuv feature set extracted")

        # save features
        np.save(
            os.path.join(config.model_dir, f"{pre_fix}_yuv_level_{level_index}.npy"),
            yuv_feature_level_set)
        np.save(
            os.path.join(config.model_dir, f"{pre_fix}_test_yuv_level_{level_index}.npy"),
            test_yuv_feature_level_set)
        timer.register("yuv feature set saved")
        del yuv_feature_level_set, test_yuv_feature_level_set
        gc.collect()
        timer.register(f"level {level_index} combined")


if __name__ == '__main__':
    # First time run:
    # load data
    y_set, u_set, v_set, surface_set, \
        test_y_set, test_u_set, test_v_set, test_surface_set = \
            load_casia_v2_yuv()
    timer.register("data loaded")

    # train surface ssl model
    train_surface_representation(surface_set)
    # train yuv ssl model
    train_yuv_representation(y_set, u_set, v_set)

    # save features
    save_all_features(y_set, u_set, v_set,
                      test_y_set, test_u_set, test_v_set)
    combine_all_features()

    timer.print()


    # !Comment out the above code and run the following code 
    # after successfully running the above code once.

    # load data
    y_set, u_set, v_set, surface_set, \
        test_y_set, test_u_set, test_v_set, test_surface_set = \
            load_casia_v2_yuv()
    timer.register("data loaded")
    
    del y_set, u_set, v_set, test_y_set, test_u_set, test_v_set
    gc.collect()

    # train surface decision model
    # train_surface_decision_all(y_set, u_set, v_set, surface_set,
    #                            test_y_set, test_u_set, test_v_set, 
    #                            test_surface_set)
    train_surface_decision_all(surface_set, test_surface_set)

    timer.print()


