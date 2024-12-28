"""
mpc_net/training/train_moe.py

Train a Mixture-of-Experts with bigger architecture and more epochs
to get a better policy.
"""

import tensorflow as tf
from mpc_net.models.mixture_of_experts import MixtureOfExperts

def train_mixture_of_experts(X_data, Y_data,
                             num_experts=12,  # up from 8
                             hidden_dim=64,   # bigger hidden layers
                             lr=1e-3,
                             batch_size=32,
                             epochs=300):     # more epochs => better fit
    input_dim = X_data.shape[1]
    model = MixtureOfExperts(num_experts=num_experts,
                             input_dim=input_dim,
                             hidden_dim=hidden_dim)
    model.build((None, input_dim))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')

    print(f"DEBUG: Starting MoE training => experts={num_experts}, hidden={hidden_dim}, epochs={epochs}")
    history = model.fit(
        X_data, Y_data,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return model, history
