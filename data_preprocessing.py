from torchvision import datasets
from torchvision.transforms import v2
import torch



def prepare_dataset(mean=(0.5,), std=(0.5,), dtype=torch.float32, scale=True):

  """
  Prepares the FashionMNIST dataset with normalization and returns train/test datasets.

  Args:

      mean(tuple): Mean for normalization (default: (0.5,))
      std (tuple): Standard derivation for normalization (default: (0.5,))
      dtype (torch.dtype): Data type for tensors (default: torch.float32)
      scale (bool): Whether to scale images to [0,1] (default: True)

  Returns:
      tuple: (train_dataset, test_dataset) after applying transformations


  """

  # Define the transformation pipeline for preprocessing
  transform = v2.Compose([
    v2.ToTensor(),                    # Convert images to Pytorch tensors (shape: C*H*W)
    v2.ToDtype(dtype, scale=scale),   # Convert dtype (float32) and scale pixel values to [0, 1] if scale=True
    v2.Normalize(mean, std)           # Normalize tensor: (x - mean) / std

])

  # Load the training dataset with preprocessing
  train_dataset = datasets.FashionMNIST(
      root='./data',        # Directory to store the dataset
      train=True,           # Specify training set
      download=True,        # Download if not already present
      transform=transform   # Apply defined transformations
  )

  # Load the test dataset with preprocessing
  test_dataset = datasets.FashionMNIST(
    root='./data',
    train=False,          # Specify test set
    download=True,
    transform=transform
)

  return (train_dataset, test_dataset)


def print_loader_stats(train_dataset, test_dataset):

  """
  Prints dataset statistics (shape, type, dtype).
  Useful for verifying data preprocessing.
  """

  # Shape (number of samples, height, width)
  print('train_loader shape:', train_dataset.data.shape)
  print('test_loader shape:', test_dataset.data.shape)
  print()

  # Type (should be torch.Tensor)
  print('train_loader type:', type(train_dataset.data))
  print('test_loader type:', type(test_dataset.data))
  print()

  # Data type (should be uint8 before transforms)
  print('train_loader dtype:', train_dataset.data.dtype)
  print('test_loader dtype:', test_dataset.data.dtype)


if __name__ == "__main__":

  # Run the preprocessing and print dataset details when executed directly
  train_dataset, test_dataset = prepare_dataset()
  print_loader_stats(train_dataset, test_dataset)
