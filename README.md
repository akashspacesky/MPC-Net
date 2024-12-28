# MPC-Net

**MPC-Net** is an open-source repository for collecting expert data from an MPC (Model Predictive Control) solver and training a Mixture-of-Experts neural network to mimic the MPC policy.

## Features

- Single-step data collection using arc-length coverage
- Modular path generators (line, circle, figure-8, sinusoidal, spiral)
- Unicycle dynamics model
- Customizable CasADi-based MPC solver
- Keras/TensorFlow Mixture-of-Experts architecture
- Visualization scripts

## Installation

1. Clone this repo:

   ```bash
   git clone https://github.com/akashspacesky/MPC-Net.git
   cd mpc-net

2. Create a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate

3. pip3 install -r requirements.txt

## Usage

1. Collect data & train:

    ```bash
    python3 -m scripts.train

This will run the MPC data collection on multiple "" paths, store the expert dataset (X_data, Y_data), and train the MoE model. The final model is saved to moe_model.h5

2. Evaluate

    ```bash
    python3 -m scripts.evaluate

Loads moe_model.h5 and runs the learned policy on a  figure-8 path, producing CTE and yaw error plots and final trajectory.

## Results 



## Folder Structure

    mpc_net/
        dynamics/
        paths/
        mpc/
        models/
        training/
        evaluation/
    scripts/

See individual subfolders for more details.

## Contributing

Feel free to open issues or PRs if you find bugs or have improvements!
Happy tracking!