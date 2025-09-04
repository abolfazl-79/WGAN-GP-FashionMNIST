# 🧵 WGAN-GP on FashionMNIST

An implementation of **Wasserstein GAN with Gradient Penalty (WGAN-GP)** trained on the **FashionMNIST** dataset.  
This project demonstrates stable GAN training, image generation, loss curve visualization, and evaluation using **Fréchet Inception Distance (FID)**.

---

## 📂 Project Structure

├── data_preprocessing.py # Dataset loading & normalization
├── model.py # Generator & Discriminator
├── train.py # Training loop (WGAN-GP)
├── evaluation.py # FID computation
├── visualization.py # Save samples, plot loss curves, make GIFs
├── main.py # Run training & evaluation
├── results/ # Generated samples, loss curves, GIFs


---

## ✨ Features

- ✅ **WGAN-GP loss** → stable GAN training  
- ✅ **FashionMNIST dataset** (grayscale clothing items)  
- ✅ **Generator & Discriminator** with InstanceNorm  
- ✅ **Visualization**:
  - Save generated samples per epoch  
  - Track loss curves  
  - GIF animations of training progress  
- ✅ **Evaluation**:
  - Compute **FID score** to measure quality/diversity  

---


📌 Requirements

Python 3.8+

PyTorch

torchvision

matplotlib

imageio

numpy

## 🚀 Usage

Train the WGAN-GP on **FashionMNIST**:

```bash
python main.py


Outputs:


### Generated Samples During Training
![Training Progress](results/training_progress.gif)
### Loss Curve Progress
![Loss Curve Progress](results/loss_curve_progress.gif)


Evaluation

We use Fréchet Inception Distance (FID) as a metric:

Lower FID = Better quality & diversity