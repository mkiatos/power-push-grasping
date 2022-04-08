# Power Push-Grasping

Power Push-Grasping (PPG) is a method for planning multi-fingered power grasps in dense clutter. Specifically, robot agents are trained in order to create enough space for the fingers to
wrap around the target object to perform a stable power grasp, using a single primitive action.

This repository provides PyTorch code for training and testing PPG policies in simulation with a Barrett Hand. This is the reference implementation for the paper: 'Learning Push-Grasping in Dense Clutter'.

## Citing
If you find this code useful in your work, please consider citing:
```shell

```

## Installation
```shell
git clone git@github.com:mkiatos/power-push-grasping.git
cd power-push-grasping

virtualenv ./venv --python=python3
source ./venv/bin/activate
pip install -e .
```

In this implementation PytTorch 1.9.0 was used:
```shell
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

## A Quick-Start: Demo in Simulation

1.Download the pretrained models.
```commandline
cd downloads
./download-weights.sh
cd ..
```
2.Then, run the following command.
```commandline
python test.py --fcn_model 'downloads/fcn_model.pt' --reg_model 'downloads/reg_model.pt' --n_scenes 10
```


## Dataset
Collect training data (saved locally).
```commandline
python collect_data.py --n_samples 10000 --seed 1
```

## Training
To train the FCN model for predicting the position and orientation:
```commandline
python train.py --module 'fcn' --epochs 100
```

To train the Aperture-CNN regression module:
```commandline
python train.py --module 'reg' --epochs 100
```

## Evaluation
To evaluate your own models just replace the snapshot:
```commandline
python test.py --fcn_model path-to-fcn-model --reg_model path-to-reg-model --object_set 'seen' --n_scenes 100 --seed 0
```