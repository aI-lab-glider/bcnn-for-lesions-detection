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

```
python3 3d_cnn_prometheus/preprocessing/preprocess_data.py
```

## Learning

```
python3 3d_cnn_prometheus/learning/train_network.py
```