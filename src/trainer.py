import time
import torch

import matplotlib.pyplot as plt

from torch import nn, optim
from torch.utils.data import DataLoader
from network import Discriminator, Generator


class Trainer():
    def __init__(self, epochs: int, data_loader: DataLoader, batch_size: int, z_dim: int, d_model: Discriminator, g_model: Generator, d_optimizer: optim.Adam, g_optimizer: optim.Adam, d_loss_fn: nn.BCELoss, g_loss_fn: nn.BCELoss, device: torch.device) -> None:
        self.epochs         = epochs
        self.data_loader    = data_loader
        self.batch_size     = batch_size
        self.z_dim          = z_dim
        self.d_model        = d_model
        self.g_model        = g_model
        self.d_optimizer    = d_optimizer
        self.g_optimizer    = g_optimizer
        self.d_loss_fn      = d_loss_fn
        self.g_loss_fn      = g_loss_fn
        self.device         = device

        self.prompt         = torch.tensor(0).to(device=device)

    def _train_discriminator(self, x: torch.Tensor) -> float:
        x_real = x.to(device=self.device)
        y_real = torch.ones(self.batch_size, 1).to(device=self.device)

        output_real = self.d_model(x_real)
        loss_real = self.d_loss_fn(output_real, y_real)

        z = torch.randn(self.batch_size, 1, 7, 7).to(device=self.device)
        y_fake = torch.zeros(self.batch_size, 1).to(device=self.device)

        with torch.no_grad():
            x_fake = self.g_model(z)

        output_fake = self.d_model(x_fake)
        loss_fake = self.d_loss_fn(output_fake, y_fake)

        loss = loss_real + loss_fake

        # Update model.
        self.d_model.zero_grad()
        loss.backward()
        self.d_optimizer.step()

        return loss.item()

    def _train_generator(self, x: torch.Tensor, prompt: torch.Tensor) -> float:
        z = torch.randn(self.batch_size, 1, 7, 7, dtype=torch.float32).to(device=self.device)
        y = torch.ones(self.batch_size, 1).to(device=self.device)

        x_fake = self.g_model(z)

        y_fake = self.d_model(x_fake)

        loss = self.g_loss_fn(y_fake, y)

        # Update model.
        self.g_optimizer.zero_grad()
        loss.backward()
        self.g_optimizer.step()

        return loss.item()

    def train(self) -> None:
        self.d_model.train()
        self.g_model.train()

        for epoch in range(self.epochs):
            # Set defualt values.
            d_loss = 0
            g_loss = 0

            # Set clock.
            start_time = time.time()

            # Set counter.
            total = len(self.data_loader)
            counter = 0

            for x, _ in self.data_loader:
                # Skip batch if not enough data.
                if x.shape[0] != self.batch_size: continue

                d_loss += self._train_discriminator(x)

                for _ in range(1):
                    g_loss += self._train_generator(x, self.prompt)

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
                f"                              ",
            )

            self.test()

    @torch.no_grad()
    def test(self) -> None:
        self.d_model.eval()
        self.g_model.eval()

        z = torch.randn((16, 1, 7, 7), dtype=torch.float32).to(device=self.device)
        outputs = self.g_model(z)

        figure = plt.figure(figsize=(8, 8))

        for i, image in enumerate(outputs):
            figure.add_subplot(8, 8, i + 1)
            plt.imshow(image.cpu().detach().numpy()[0, :, :], cmap='gray')

        plt.show()