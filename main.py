
from data_preprocessing import prepare_dataset
import model
from evaluation import FID
from train import Trainer
import torch
from visualization import make_gif, make_loss_gif

def main():

  # ------------------------------
  # Hyperparameters
  # ------------------------------

  NOISE_DIM = 128           # Size of latent noise vector (input to Generator)
  BATCH_SIZE = 64           # Number of samples per training batch
  D_ITERATION = 5           # Number of Discriminator updates per Generator update
  GP_LAMBDA = 10            # Gradient penalty coefficient for WGAN-GP
  PRINT_EVERY = 10          # How often to print training progress
  PLOT_EVERY = 2            # How often to save loss curves and generated images
  LEARNING_RATE = 0.0001    # Learning rate for optimizers
  EPOCH = 200               # Total number of training epochs

  # Fixed noise for monitoring generator’s progress (always same input → track improvements)
  FIXED_NOISE = torch.randn(16, NOISE_DIM).to('cuda')

  # ------------------------------
  # Initialize Generator & Discriminator
  # ------------------------------
  G = model.Generator().to('cuda')
  D = model.Discriminator().to('cuda')

  # Optimizers (Adam with betas recommended for WGAN-GP)
  G_opt = torch.optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
  D_opt = torch.optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

  # ------------------------------
  # Load datasets
  # ------------------------------
  train_dataset, test_dataset = prepare_dataset()


  # ------------------------------
  # Trainer setup
  # ------------------------------
  trainer = Trainer(G, D, D_opt, G_opt, D_ITERATION, GP_LAMBDA, PRINT_EVERY, PLOT_EVERY, NOISE_DIM, FIXED_NOISE)


  # ------------------------------
  # Train GAN
  # ------------------------------
  G = trainer.train_GAN(epoch=EPOCH, train_dataset=train_dataset, batch_size=BATCH_SIZE)

  # ------------------------------
  # Save training visualization and loss curve as GIF
  # ------------------------------
  make_gif()
  make_loss_gif()

  # ------------------------------
  # Evaluate Generator with FID metric
  # ------------------------------

  # Pass generator and test dataset into FID evaluator
  fid_metric = FID(G, test_dataset.data)

  # Compute FID score
  fid_result = fid_metric.compute_fid(BATCH_SIZE, NOISE_DIM)
  print('Fréchet Inception Distance is: ', fid_result)

if __name__=='__main__':
  main()
