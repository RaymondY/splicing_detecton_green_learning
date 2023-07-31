import os
import torch


class DefaultConfig:
    # general
    device = torch.device("cpu")
    # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps" if torch.backends.mps.is_available() else device)
    # device_ids=[1, 2, 3, 4]
    gpu_id = 3
    pre_fix = "gl"

    # dir
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, "data")
    temp_dir = os.path.join(root_dir, "temp")
    model_dir = os.path.join(root_dir, "models")

    # CASIA v2.0
    casia_v2_dir = os.path.join(data_dir, "CASIA2")
    casia_v2_image_dir = os.path.join(casia_v2_dir, "img")
    casia_v2_edge_dir = os.path.join(casia_v2_dir, "edge_GT")
    casia_v2_surface_dir = os.path.join(casia_v2_dir, "surface_GT")

    # Columbia
    columbia_dir = os.path.join(data_dir, "Columbia")
    columbia_image_dir = os.path.join(columbia_dir, "img")
    columbia_edge_dir = os.path.join(columbia_dir, "edge_GT")
    columbia_surface_dir = os.path.join(columbia_dir, "surface_GT")

    # Saab
    target_channel = 3
    level_num = 5

    # RFT
    selected_num_list = [50, 150, 500, 2000]

print(torch.cuda.is_available())