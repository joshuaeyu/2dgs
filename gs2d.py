
import cv2
from metrics import *
from PIL import Image

## Image and domain helper functions

def load_image(path, width=None, device=torch.device('cpu')):
    """Load image as tensor. Optionally rescale."""
    img = Image.open(path)
    out_scale = min(1., width/img.width) if width else 1.
    out_size = (int(img.width * out_scale), int(img.height * out_scale))
    target = torch.from_numpy(np.array(img.resize(out_size), dtype=np.float32) / 255).to(device)
    return target

def build_domain_grid(res_i, res_j, device=torch.device('cpu')):
    """Build 2D array spanning [0,0] to [1,1] with resolution (res_i,res_j)."""
    domain_i, domain_j = torch.meshgrid(
        torch.linspace(0,1,res_i, device=device), 
        torch.linspace(0,1,res_j, device=device), 
        indexing='ij')
    domain_grid = torch.stack([domain_i, domain_j], dim=-1) # (H,W,2)
    return domain_grid

## Gaussian helper functions

def _prob_init(target, lambda_init=0.3):
    """Compute Gaussian initialization probability at each pixel based on image gradient."""
    target = target.detach().cpu().numpy()
    gradx = cv2.Sobel(target, cv2.CV_32F, dx=1, dy=0, ksize=cv2.FILTER_SCHARR)
    grady = cv2.Sobel(target, cv2.CV_32F, dx=0, dy=1, ksize=cv2.FILTER_SCHARR)
    grad_mag2 = np.sum(np.square(gradx) + np.square(grady), axis=-1)
    prob = (1.-lambda_init)*grad_mag2 /np.sum(grad_mag2) + lambda_init/(target.shape[0]*target.shape[1])
    prob = torch.from_numpy(prob)
    return prob

def init_gaussians(count, target=None, domain_grid=None, lambda_init=0.3, device=torch.device('cpu')):
    """Initialize 2D Gaussians via content-aware initialization or randomly."""
    if target is not None and domain_grid is not None:
        # Content-aware initialization of positions and colors 
        with torch.no_grad():
            # Sample pixels based on P_init
            prob = _prob_init(target, lambda_init=lambda_init)
            idxs_flat = torch.multinomial(torch.ravel(prob), count)
            idxs = torch.unravel_index(idxs_flat, target.shape[:2])
        params = dict(
            positions = domain_grid[idxs].clone().detach().requires_grad_().to(device),
            inv_scales = torch.full((count, 2), 1.0/0.005, requires_grad=True, device=device), # sensitive to initial scales!
            rotations = torch.zeros(count, requires_grad=True, device=device),
            colors = target[idxs].clone().detach().requires_grad_().to(device)
        )
    else:
        # Random initialization
        params = dict(
            positions = torch.rand((count, 2), requires_grad=True, device=device),
            inv_scales = torch.full((count, 2), 1.0/0.005, requires_grad=True, device=device), # sensitive to initial scales!
            rotations = torch.zeros(count, requires_grad=True, device=device),
            colors = torch.rand((count, 3), requires_grad=True, device=device),
        )
    return params

def inv_covariance(inv_scale, rotation):
    """Compute inverse covariance matrix for a batch of 2D Gaussians."""
    cos = torch.cos(rotation) # (N,)
    sin = torch.sin(rotation) # (N,)
    R = torch.stack([cos,-sin,sin,cos], dim=1).reshape(-1,2,2) # (N,2,2)
    S2_inv = torch.diag_embed(torch.square(inv_scale)) # (N,2,2)
    inv_cov = torch.einsum('...ij,...jk,...lk->...il', R, S2_inv, R) # (N,2,2)
    return inv_cov

def render_gaussians(gaussians, domain_grid):
    """Rasterize a batch of 2D Gaussians."""
    # Displacements and inverse covariance
    disp_grid = domain_grid[None,...] - gaussians['positions'][:,None,None,:] # (N,H,W,2)
    inv_cov = inv_covariance(gaussians['inv_scales'], gaussians['rotations']) # (N,2,2)
    # Contribution at each pixel, for each gaussian
    exponent = torch.einsum('n...i,nij,n...j->n...', disp_grid, inv_cov, disp_grid) # (N,H,W)
    pdf = torch.exp(-0.5 * exponent) # (N,H,W)
    # Normalized weighted sum of contributions
    contribs = pdf[...,None] * gaussians['colors'][:,None,None,:] # (N,H,W,3)
    output = torch.sum(contribs, dim=0) / (torch.sum(pdf, dim=0)[...,None] + 1e-9) # (H,W,3)
    return output

def loss_fn(prediction, target, recon_type='l2', ssim_weight=0.1):
    """L1 or L2 loss with optional SSIM loss."""
    if recon_type == 'l1':
        loss = torch.nn.functional.l1_loss(prediction, target)
    elif recon_type == 'l2':
        loss = torch.nn.functional.mse_loss(prediction, target)
    else:
        ValueError("recon_type must be 'l1' or 'l2'.")
    if ssim_weight != 0:
        loss += ssim_weight * (1. - ssim(prediction, target))
    return loss