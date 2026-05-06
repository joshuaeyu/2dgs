import cv2
import numpy as np
import torch
from skimage.measure import shannon_entropy as sk_shannon_entropy

def ssim(x, y):
    """Structural similarity."""
    x_var, x_mean = torch.var_mean(x)
    y_var, y_mean = torch.var_mean(y)
    cov = torch.mean((x - x_mean) * (y - y_mean))
    ssim = ((2.*x_mean*y_mean)*(2.*cov)) / ((x_mean**2 + y_mean**2)*(x_var + y_var))
    return ssim

def psnr(output, target):
    """Peak signal-to-noise ratio."""
    with torch.no_grad():
        mse = torch.nn.functional.mse_loss(output, target)
        psnr = -10. * torch.log(mse)
        return psnr
    
def shannon_entropy(image):
    """"Shannon entropy."""
    image = image.detach().cpu().numpy()
    se = sk_shannon_entropy(image).item()
    return se

def edgeness(image):
    """Average gradient magnitude."""
    image = image.detach().cpu().numpy()
    gradx = cv2.Sobel(image, cv2.CV_32F, dx=1, dy=0, ksize=cv2.FILTER_SCHARR)
    grady = cv2.Sobel(image, cv2.CV_32F, dx=0, dy=1, ksize=cv2.FILTER_SCHARR)
    grad_mag = np.mean(np.sqrt(np.square(gradx) + np.square(grady))).item()
    return grad_mag