
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from visualization import plot_loss_curve, save_generated_images




# ===================================
# Trainer Class for WGAN-GP
# ===================================
class Trainer(nn.Module):
  def __init__(self, generator, discriminator, dis_optimizer, gen_optimizer, dis_iteration, gp_lambda, print_every, plot_every, noise_dim, fix_noise):
     super().__init__()

     # Models
     self.G = generator      # Generator
     self.D = discriminator  # Discriminator / Critic

     # Optimizer
     self.D_opt = dis_optimizer
     self.G_opt = gen_optimizer

     # Training settings
     self.D_iteration = dis_iteration   # How many times to update D per G update
     self.GP_lambda = gp_lambda         # Weight for Gradient Penalty
     self.print_every = print_every     # Print loss every N epochs
     self.plot_every = plot_every       # Plot curves every N epochs
     self.noise_dim = noise_dim         # Latent space dimension (z)
     self.fix_noise = fix_noise         # Fixed noise for visualization


     # Loss history storage
     self.G_loss_list = []
     self.D_loss_list = []


  # ===================================
  # Compute Gradient Penalty
  # ===================================
  def __compute_gp(self, netD, real_data, fake_data):
    batch_size = real_data.size(0)

    # Sample Epsilon from uniform distribution
    eps = torch.rand(batch_size, 1, 1, 1).to(real_data.device)
    eps = eps.expand_as(real_data)

    # Interpolation between real data and fake data.
    interpolation = (eps * real_data + (1 - eps) * fake_data).requires_grad_(True)


    # Forward through discriminator
    interp_logits = netD(interpolation)
    grad_outputs = torch.ones_like(interp_logits)

    # Compute Gradients
    gradients = torch.autograd.grad(
        outputs=interp_logits,
        inputs=interpolation,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]

    # Flatten gradients per sample
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, 1)     # L2 norm

    # Penalty = (‖grad‖2 − 1)^2
    return torch.mean((grad_norm - 1) ** 2)


  # ===================================
  # Discriminator Loss
  # ===================================
  def __D_loss(self, D_outputs, images, netD):
    real_output, fake_output = D_outputs
    real_images, fake_images = images

    # WGAN loss: maximize D(real) - D(fake)
    real_loss = torch.mean(real_output)
    fake_loss = torch.mean(fake_output)

    # Add gradient penalty
    GP = self.__compute_gp(netD, real_images, fake_images)

    # Final loss (to minimize, so we flip sign)
    return (fake_loss - real_loss) + self.GP_lambda * GP


  # ===================================
  # Generator Loss
  # ===================================
  def __G_loss(self, fake_output):

    # WGAN loss: maximize D(fake), so minimize -D(fake)
    return -torch.mean(fake_output)


  # ===================================
  # Train Discriminator Step
  # ===================================
  def __train_D(self, batch_size, image_batch, epoch_loss_D):

    self.D_opt.zero_grad()

    # Generate fake images
    noise = torch.randn(batch_size, self.noise_dim).to('cuda')
    fake_image_batch = self.G(noise)

    # Get discriminator outputs
    real_output_batch = self.D(image_batch)
    fake_output_batch = self.D(fake_image_batch)

    # Compute D loss (with GP)
    D_loss_value = self.__D_loss(D_outputs=(real_output_batch,fake_output_batch), images=(image_batch, fake_image_batch) , netD=self.D)
    epoch_loss_D += D_loss_value.item()

    # Backpropagation & update
    D_loss_value.backward()
    self.D_opt.step()

    return epoch_loss_D


  # ===================================
  # Train Generator Step
  # ===================================
  def __train_G(self, batch_size, epoch_loss_G):

    self.G_opt.zero_grad()

    # Generate fake images
    noise = torch.randn(batch_size, self.noise_dim).to('cuda')
    fake_image_batch = self.G(noise)

    # Evaluate with discriminator
    fake_output_batch = self.D(fake_image_batch)

    # Compute G loss
    G_loss_value = self.__G_loss(fake_output_batch)
    epoch_loss_G += G_loss_value.item()

    # Backpropagation & update
    G_loss_value.backward()
    self.G_opt.step()

    return epoch_loss_G


  # ===================================
  # Full Training Loop
  # ===================================
  def train_GAN(self, epoch, train_dataset, batch_size):

    # Wrap dataset in DataLoader
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True, num_workers=2)

    epochs_list = []

    for epoch_num in range(epoch):

      self.D.train()
      self.G.train()

      epochs_list.append(epoch_num + 1)

      # Track average losses
      epoch_loss_G = 0.0
      epoch_loss_D = 0.0
      D_batch_iter = 0
      G_batch_iter = 0


      for i, data in enumerate(trainloader):

        image_batch = data[0].to('cuda')
        batch_size = image_batch.shape[0]

        # Train Discriminator
        D_batch_iter += 1
        epoch_loss_D = self.__train_D(batch_size, image_batch, epoch_loss_D)

        # Train Generator every D_iteration steps
        if (i+1) % self.D_iteration == 0:
          G_batch_iter += 1
          epoch_loss_G = self.__train_G(batch_size, epoch_loss_G)

      # Average losses per epoch
      epoch_loss_D /= D_batch_iter
      epoch_loss_G /= G_batch_iter

      # Save losses
      self.D_loss_list.append(epoch_loss_D)
      self.G_loss_list.append(epoch_loss_G)


      # Print progress
      if epoch_num % self.print_every == 0:
        print('Epoch: ', epoch_num)
        print('generator loss: ', epoch_loss_G)
        print('discriminator loss: ', epoch_loss_D)

      # Plot & save generated samples
      if epoch_num % self.plot_every == 0:
        figsize=(10,5)
        plot_loss_curve(epoch_num, epochs_list, epoch_loss_G, epoch_loss_D, figsize, self.G_loss_list, self.D_loss_list)
        save_generated_images(self.G, epoch_num, self.fix_noise, 128, "samples")



    return self.G

