# 3d-cnn-prometheus

## Setup

1. Activate venv and install requirements

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Update config files in required modules

```
vim 3d_cnn_prometheus/MODULE_NAME/config.json
```

## Preprocessing

1. Update config path in `3d_cnn_prometheus/preprocessing/preprocess_data.py` (tmp)
2. Run preprocessing:

```
python3 3d_cnn_prometheus/preprocessing/preprocess_data.py
```

## Learning

1. Update config path in `3d_cnn_prometheus/learning/model/experiment_setup.py`
   and `3d_cnn_prometheus/learning/model/constants.py`  (tmp)
2. Run training:

```
python3 3d_cnn_prometheus/learning/train_network.py
```