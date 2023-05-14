import numpy as np
import torch
import torchvision
from scipy import linalg

def calculate_fid_score_improved_diffusion(generator, real_images, batch_size, device):
    """
    Calculates FID score for Improved Diffusion.

    Args:
        generator: The Improved Diffusion generator.
        real_images: The real images for comparison.
        batch_size: Batch size to use for generating images.
        device: Device to use for computation (e.g. 'cuda' or 'cpu').

    Returns:
        The FID score.
    """
    # Set generator to evaluation mode
    generator.eval()

    # Calculate the features for the real images
    real_features = get_features(real_images, batch_size, device)

    # Generate fake images and calculate their features
    num_batches = len(real_images) // batch_size
    fake_features = []
    for i in range(num_batches):
        with torch.no_grad():
            fake_images = generator.sample(batch_size, device=device).clamp(-1, 1)
        fake_features.append(get_features(fake_images, batch_size, device))
    fake_features = torch.cat(fake_features, dim=0)

    # Calculate the mean and covariance of the real and fake features
    mu1, sigma1 = torch.mean(real_features, dim=0), torch.cov(real_features, rowvar=False)
    mu2, sigma2 = torch.mean(fake_features, dim=0), torch.cov(fake_features, rowvar=False)

    # Calculate the FID score
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.mm(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        cov_eps = 1e-6 * np.eye(sigma1.shape[0], dtype=sigma1.dtype)
        covmean = linalg.sqrtm((sigma1 + cov_eps).mm(sigma2 + cov_eps))
    fid_score = diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * torch.trace(covmean)

    return fid_score.item()

def get_features(images, batch_size, device):
    """
    Calculates the features of the given images.

    Args:
        images: The images to calculate the features for.
        batch_size: Batch size to use for feature extraction.
        device: Device to use for computation (e.g. 'cuda' or 'cpu').

    Returns:
        The features of the given images.
    """
    # Load pre-trained Inception V3 model
    model = torchvision.models.inception_v3(pretrained=True, aux_logits=False)
    model.eval()
    model.to(device)

    # Remove last two layers of the model
    model.fc = torch.nn.Identity()
    model.avgpool = torch.nn.Identity()

    # Calculate the features in batches
    num_batches = len(images) // batch_size
    features = []
    for i in range(num_batches):
        batch = images[i * batch_size:(i + 1) * batch_size]
        batch = (batch + 1) / 2  # Scale images from [-1, 1] to [0, 1]
        batch = torch.nn.functional.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=True)
        batch = batch.to(device)
        with torch.no_grad():
            feature = model(batch)
        features.append(feature)
    features = torch.cat(features, dim=0)

    return features




------





import torch
from fid_score_improved_diffusion import calculate_fid_score_improved_diffusion

# Load the Improved Diffusion generator
generator = torch.load('generator.pt')

# Load the real images for comparison
real_images = torch.load('real_images.pt')

# Set batch size and device
batch_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Calculate the FID score
fid_score = calculate_fid_score_improved_diffusion(generator, real_images, batch_size, device)
print('FID score:', fid_score)
