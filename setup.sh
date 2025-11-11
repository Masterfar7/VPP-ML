#!/bin/bash
sudo apt update
sudo apt install -y python3 python3-pip g++ python3-venv
python3 -m venv ml_env
source ml_env/bin/activate
pip install --upgrade pip
pip install pandas scikit-learn numpy joblib

