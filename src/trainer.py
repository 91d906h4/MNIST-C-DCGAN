import time
import torch

import matplotlib.pyplot as plt

from torch import nn, optim
from torch.utils.data import DataLoader
from network import Discriminator, Generator


class Trainer():
    def __init__(self, data_loader: DataLoader, batch_size: int, d_model: Discriminator, g_model: Generator, d_optimizer: optim.Adam, g_optimizer: optim.Adam, d_loss_fn: nn.BCELoss, g_loss_fn: nn.BCELoss, device: torch.device, z_shape: tuple) -> None:
        """Trainer class

        The trainer class of the discriminator and generator models.
        
        Args:
            data_loader (DataLoader): Data loader.
            batch_size (int): Batch size.
            d_model (Discriminator): Discriminator model.
            g_model (Generator): Generator model.
            d_optimizer (optim.Adam): Discriminator optimizer.
            g_optimizer (optim.Adam): Generator optimizer.
            d_loss_fn (nn.BCELoss): Discriminator loss function.
            g_loss_fn (nn.BCELoss): Generator loss function.
            device (torch.device): Device to train models.
            z_shape (tuple): Noise shape. (e.g., (100, 1, 1))

        """

        self.data_loader    = data_loader
        self.batch_size     = batch_size
        self.d_model        = d_model
        self.g_model        = g_model
        self.d_optimizer    = d_optimizer
        self.g_optimizer    = g_optimizer
        self.d_loss_fn      = d_loss_fn
        self.g_loss_fn      = g_loss_fn
        self.device         = device
        self.z_shape        = z_shape

    def _train_discriminator(self, x: torch.Tensor, label: torch.Tensor) -> float:
        """_train_discriminator private method
        
        The private method to train the discriminator model.

        Args:
            x (torch.Tensor): Input data.
            label (torch.Tensor): Text prompt data.

        Returns:
            float: Discriminator loss.

        """

        # 1. Train the discriminator with real data.

        # Generate real data and label.
        # We want the discriminator to classify real data as 1, so we
        # set all y_real to ones.
        x_real = x.to(device=self.device)
        y_real = torch.ones(self.batch_size, 1).to(device=self.device)

        # The real label of 0 ~ 9.
        label = label.to(device=self.device)

        # Get output of real data from discriminator.
        output_real = self.d_model(x_real, label)

        # Calculate loss of real data.
        loss_real = self.d_loss_fn(output_real, y_real)

        # 2. Train the discriminator with fake data.

        # Generate noise input and fake label.
        # We want the discriminator to classify fake data as 0, so we
        # set all y_fake to zeros.
        z = torch.randn((self.batch_size, *self.z_shape)).to(device=self.device)
        y_fake = torch.zeros(self.batch_size, 1).to(device=self.device)

        # Generate fake data.
        x_fake = self.g_model(z, label)

        # Detach data from g_model to prevent backpropagation.
        x_fake = x_fake.detach()

        # Get output of fake data from discriminator.
        output_fake = self.d_model(x_fake, label)

        # Calculate loss of fake data.
        loss_fake = self.d_loss_fn(output_fake, y_fake)

        # Calculate total loss.
        loss = loss_real + loss_fake

        # Update model.
        self.d_model.zero_grad()
        loss.backward()
        self.d_optimizer.step()

        return loss.item()

    def _train_generator(self, x: torch.Tensor, label: torch.Tensor) -> float:
        """_train_generator private method
        
        The private method to train the generator model.

        Args:
            x (torch.Tensor): Input data.
            prompt (torch.Tensor): Text prompt data.

        Returns:
            float: Generator loss.

        """

        # Generate noise input and fake label.
        # In the training step of generator, we want the discriminator
        # to classify fake data as 1, so we set all y_fake to ones.
        z = torch.randn((self.batch_size, *self.z_shape), dtype=torch.float32).to(device=self.device)
        y = torch.ones(self.batch_size, 1).to(device=self.device)

        # Generate random label of 0 ~ 9.
        label = torch.randint(low=0, high=10, size=(self.batch_size, 1)).to(device=self.device)

        # Generate fake data.
        x_fake = self.g_model(z, label)

        # Get output of fake data from discriminator.
        y_fake = self.d_model(x_fake, label)

        # Calculate loss of fake data.
        loss = self.g_loss_fn(y_fake, y)

        # Update model.
        self.g_optimizer.zero_grad()
        loss.backward()
        self.g_optimizer.step()

        return loss.item()

    def train(self, epochs: int, dg_ratio: int, test: bool=False, test_num: int=30) -> None:
        """train public method
        
        The public method to train the discriminator and generator models.
        
        Args:
            epochs (int): The number of epochs.
            dg_ratio (int): Number of times to train the discriminator while training the generator.
            test (bool, optional): Whether to test the model. Defaults to False.
            test_num (int, optional): Number of images to test. (default: 30)

        """

        # Set model to training mode.
        self.d_model.train()
        self.g_model.train()

        for epoch in range(epochs):
            # Set defualt values.
            d_loss = 0
            g_loss = 0

            # Set clock.
            start_time = time.time()

            # Set counter.
            total = len(self.data_loader)
            counter = 0

            # Iterate over all data in data loader.
            for data, label in self.data_loader:
                # Skip batch if not enough data.
                if data.shape[0] != self.batch_size: continue

                # Train discriminator.
                for _ in range(dg_ratio):
                    d_loss += self._train_discriminator(data, label) / dg_ratio

                # Train generator.
                g_loss += self._train_generator(data, label)

                counter += 1

                # Print training progress.
                print(
                    f"Epoch: {epoch} | "
                    f"Time: {time.time() - start_time:.3f} | "
                    f"Progress: {counter / total * 100:.3f}% | "
                    f"D Loss: {d_loss / counter:.3f} | "
                    f"G Loss: {g_loss / counter:.3f}",
                    end="\r"
                )

            # Print training progress.
            print(
                f"Epoch: {epoch} | "
                f"Time: {time.time() - start_time:.3f} | "
                f"D Loss: {d_loss / total:.3f} | "
                f"G Loss: {g_loss / total:.3f}"
                f"                              " # Append empty string to clear the output.
            )

            # Test while training.
            if test: self.test(num=test_num)

    @torch.no_grad()
    def test(self, num: int=30) -> None:
        """test public method
        
        The public method to test the model.

        Args:
            num (int, optional): Number of images to generate. (default: 30)

        """

        # Set model to evaluation mode.
        self.d_model.eval()
        self.g_model.eval()

        # Generate noise input.
        z = torch.randn((num, *self.z_shape), dtype=torch.float32).to(device=self.device)

        # Generate labels.
        # The label is a repeating sequence of 0 ~ 9.
        y = torch.tensor([x % 10 for x in range(num)])
        y = y.view(-1, 1).to(device=self.device)

        # Generate data.
        outputs = self.g_model(z, y)

        # Set figure.
        figure = plt.figure(figsize=(10, 10))

        for i, image in enumerate(outputs):
            # Set subplot.
            axes = figure.add_subplot(10, 10, i + 1)
            axes.set_axis_off()

            # Detach data from GPU.
            image = image.cpu().detach().numpy()
            image = image[0, :, :]

            # Add image.
            plt.imshow(image, cmap='gray')

        plt.show()