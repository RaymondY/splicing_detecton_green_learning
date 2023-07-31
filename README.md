# splicing_detecton_green_learning

## How to set up the environment
1. Install [Anaconda](https://www.anaconda.com/products/individual)
2. Create a new environment with the following command:
```
conda create --name splicing_detection python=3.10
conda activate splicing_detection

conda install numpy scipy scikit-learn matplotlib pandas

conda install -c "nvidia/label/cuda-11.7.0" cuda-toolkit
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
pip install kornia

conda install -c conda-forge py-xgboost-gpu

pip install opencv-python
```

## Tips
1. Use nohup and & to run the program in the background
```
