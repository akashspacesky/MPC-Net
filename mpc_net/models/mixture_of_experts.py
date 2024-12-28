# mpc_net/models/mixture_of_experts.py

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class MixtureOfExperts(Model):
    def __init__(self, num_experts=8, input_dim=6, hidden_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts

        self.experts = []
        for _ in range(num_experts):
            exp = tf.keras.Sequential([
                layers.Dense(hidden_dim, activation='relu'),
                layers.Dense(hidden_dim, activation='relu'),
                layers.Dense(2)
            ])
            self.experts.append(exp)

        self.gating_network = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(num_experts, activation='softmax')
        ])

    def call(self, inputs, training=None):
        gating_probs = self.gating_network(inputs, training=training)  # (B,E)
        expert_outputs = [exp(inputs, training=training) for exp in self.experts]
        expert_outputs = tf.stack(expert_outputs, axis=1)  # (B,E,2)
        gating_probs   = tf.expand_dims(gating_probs, axis=-1)  # (B,E,1)
        out = tf.reduce_sum(gating_probs * expert_outputs, axis=1)  # (B,2)
        return out
