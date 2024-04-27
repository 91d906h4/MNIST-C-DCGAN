# Set Hyper-Parameters.

# Dataset.
DATASET_PATH        = "../dataset"  # The path of dataset.
BATCH_SIZE          = 16            # Batch size.
IMAGE_SIZE          = 28            # The size of image.

# Optimizer.
BETA1               = 0.5           # Beta values for Adam.
BETA2               = 0.999
LEARNING_RATE       = 1e-5          # Learning rate.

# Model.
Z_SHAPE             = (100, 1, 1)   # Noise shape.
                                    # The first number is the noise
                                    # dimension, and the other is 
                                    # the image shape.

# Trainer.
EPOCHS              = 10            # Epochs.
GD_RATIO            = 2             # The number of times to train
                                    # the generator while training
                                    # the discriminator.
TEST_WHILE_TRAIN    = True          # Whether to test while training.
TEST_NUM            = 30            # The number of test images to generate
                                    # while training.