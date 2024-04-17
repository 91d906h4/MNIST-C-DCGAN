# Set Hyper-Parameters.

# Dataset.
dataset_path    = "../dataset"  # The path of dataset.
batch_size      = 128           # Batch size.

# Optimizer.
beta1           = 0.5           # Beta values for Adam.
beta2           = 0.999
learning_rate   = 1e-5          # Learning rate.

# Model.
z_shape         = (100, 1, 1)   # Noise shape.

# Trainer.
epochs          = 10            # Epochs.