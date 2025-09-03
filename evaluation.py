
import torch.nn as nn
import numpy as np
import torch
from torchvision.transforms import v2

class FID(nn.Module):
  def __init__(self, generator, test_dataset):
     super().__init__()


      # Save generator and test dataset for use
     self.G = generator
     self.test_dataset = test_dataset

     # Preprocessing pipeline to match InceptionV3 input requirements:
     #  - Resize to 299x299
     #  - Normalize with ImageNet mean and std
     self.preprocess = v2.Compose([
      v2.Resize(299),
      v2.CenterCrop(299),
      v2.ToDtype(torch.float32, scale=True),
      v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])



  # -------------------------------
  # Step 1: Load random real images
  # ----------------------------
  def __load_real_images(self, batch_size):
    # Sample random indices from test dataset
    random_ids = np.random.randint(len(self.test_dataset), size=batch_size)
    real_images = self.test_dataset[random_ids]
    return real_images

  # -------------------------------
  # Step 2: Generate fake images
  # -------------------------------
  def __load_fake_images(self, noise_dim, batch_size):
    # Generate noise input for generator
    noise = torch.randn(batch_size, noise_dim).to('cuda')
    self.G.eval()
    return self.G(noise)

  # -------------------------------
  # Step 3a: Preprocess real images
  # -------------------------------
  def __preprocess_real_images(self, real_images):
    # Convert grayscale (1 channel) → RGB (3 channels)
    expanded_real_images = real_images[:, None,: ,:]
    size_value = expanded_real_images.size()
    list_size_value = list(size_value)
    list_size_value[1] = 3
    final_real_images = expanded_real_images.expand((list_size_value))

    # Apply resize + normalization
    pre_real_images = self.preprocess(final_real_images)
    if torch.cuda.is_available():
      pre_real_images = pre_real_images.to('cuda')

    return pre_real_images

  # -------------------------------
  # Step 3b: Preprocess fake images
  # ------------------------------
  def __preprocess_fake_images(self, fake_images):
    # Convert grayscale (1 channel) → RGB (3 channels)
    size_value = fake_images.size()
    list_size_value = list(size_value)
    list_size_value[1] = 3
    list_size_value
    final_fake_images = fake_images.expand((list_size_value))

    # Apply resize + normalization
    pre_fake_images = self.preprocess(final_fake_images)
    if torch.cuda.is_available():
      pre_fake_images = pre_fake_images.to('cuda')

    return pre_fake_images


  # -------------------------------
  # Step 4: Load pretrained InceptionV3
  # -------------------------------
  def __load_inception_v3(self):

    # Load pretrained InceptionV3 from torchvision
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    model.eval()

    # Move to GPU if available
    if torch.cuda.is_available():
        model.to('cuda')

    # Remove final classification layer (use features only)
    model.fc = torch.nn.Identity()

    return model


  # -------------------------------
  # Step 5: Extract features
  # -------------------------------
  def __extract_features(self, model, pre_real_images, pre_fake_images):
    # Get InceptionV3 features for both sets of images
    with torch.no_grad():
      real_output = model(pre_real_images)
      fake_output = model(pre_fake_images)

    return (real_output, fake_output)

  # -------------------------------
  # Step 6a: Mean difference (part of FID)
  # -------------------------------
  def __fid_mean_component(self, real_output, fake_output):
    mean_real = torch.mean(real_output, dim=1)
    mean_fake = torch.mean(fake_output, dim=1)
    diff_mean = mean_real - mean_fake
    return torch.matmul(diff_mean, diff_mean)

  # -------------------------------
  # Step 6b: Covariance difference (part of FID)
  # -------------------------------
  def __fid_covariance_component(self, real_output, fake_output):

    # Compute covariance matrices
    cov_real = torch.cov(real_output.T)
    cov_fake = torch.cov(fake_output.T)

    # Approximate covariance difference term
    sum_cov = cov_real + cov_fake
    cov_prod_sqrt = 2 * torch.sqrt(cov_real* cov_fake)
    diff = sum_cov - cov_prod_sqrt

    trace_cov_prod_sqrt = torch.trace(diff)
    return trace_cov_prod_sqrt



  # -------------------------------
  # Step 7: Compute FID score
  # -------------------------------
  def compute_fid(self, batch_size, noise_dim):

    # Load real and fake images
    real_images = self.__load_real_images(batch_size)
    fake_images = self.__load_fake_images(noise_dim, batch_size)

    # Preprocess for InceptionV3
    pre_real_images = self.__preprocess_real_images(real_images)
    pre_fake_images = self.__preprocess_fake_images(fake_images)

    # Load Inception model
    model = self.__load_inception_v3()

    # Extract features
    real_output, fake_output = self.__extract_features(model, pre_real_images, pre_fake_images)


    # Compute FID = mean component + covariance component
    return self.__fid_mean_component(real_output, fake_output) + self.__fid_covariance_component(real_output, fake_output)
