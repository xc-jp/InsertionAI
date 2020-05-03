# InsertionAI
Residual Reinforcement Learning used for insertion

### Dependencies
  
tensorflow==1.14.0

Pillow

stable_baselines

## Installation
It is recommended to use a virtual environment.

Install requirements with
```
pip install -r requirements.txt
```

Install gym wrapper
```
cd gym-insertion
pip install -e .
```

Install the simulation, either for Godot (https://github.com/hoel-bagard/InsertionGodot) or Unity (https://github.com/xc-jp/insertion-simulation)

## Usage

After installing the insertion simulation and starting it, run 
```
python train.py
```
