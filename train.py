from datetime import datetime
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import argparse
import torch
from tqdm import tqdm

from gs2d import *
from metrics import *

def train(target_path, n_gaussians=400, out_width=200, epochs=200, out_dir="output/train"):
    print("-"*40)

    # Set device
    device = torch.device("mps" if torch.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else
                          "cpu")

    ## ===== Initialize target image and model =====

    # Target image
    target = load_image(target_path, width=out_width, device=device)
    print(f"Target image: {target_path}")

    # Initial model
    domain_grid = build_domain_grid(target.shape[0], target.shape[1], device=device)
    gaussians = init_gaussians(n_gaussians, target, domain_grid, lambda_init=0.3, device=device)
    output = render_gaussians(gaussians, domain_grid)

    # Initial output image
    run_dir = f"{out_dir}/train_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{os.path.splitext(os.path.basename(target_path))[0]}"
    os.makedirs(f"{run_dir}/epochs", exist_ok=True)
    anim_fig, anim_ax = plt.subplots()
    anim_ax.axis('off')
    anim_frames = [[anim_ax.imshow(output.detach().cpu().numpy())]]
    anim_fig.savefig(f"{run_dir}/epochs/epoch{0:05d}.png")

    ## ===== Optimize model =====

    # Training settings
    optim_params = [
        {'name': 'positions', 'params': gaussians['positions'], 'lr': 1e-2},
        {'name': 'inv_scales', 'params': gaussians['inv_scales'], 'lr': 2e0},
        {'name': 'rotations', 'params': gaussians['rotations'], 'lr': 1e-1},
        {'name': 'colors', 'params': gaussians['colors'], 'lr': 1e-1},
    ]
    lr_step_size = 100
    lr_gamma = 0.5
    optimizer = torch.optim.AdamW(optim_params)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    # GIF animation settings
    epochs_per_frame = 10
    save_indv_frames_every = 50

    # Training loop
    train_stats = {
        'target_path': target_path,
        'target_shape': target.shape[0:2],
        'target_edgeness': edgeness(target),
        'target_shannon_entropy': shannon_entropy(target),
        'output_dir': run_dir,
        'output_shape': domain_grid.shape[0:2],
        'n_gaussians': len(gaussians['positions']),
        'epochs': epochs,
        'lr': {op['name']: op['lr'] for op in optim_params},
        'lr_step_size': lr_step_size,
        'lr_gamma': lr_gamma,
        'loss_final': None,
        'psnr_final': None,
        'ssim_final': None,
        'loss': [],
        'psnr': [],
        'ssim': [],
    }
    print(f"Fitting target image at resolution {tuple(target.shape[0:2])} with {len(gaussians['positions'])} gaussians over {epochs} epochs.")
    for epoch in (pbar := tqdm(range(epochs))):
        # Forward
        output = render_gaussians(gaussians, domain_grid)
        loss = loss_fn(output, target, recon_type='l1', ssim_weight=0.1) # Loss from Image-GS paper
        # Backward
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        # Clamp values to meaningful ranges
        with torch.no_grad():
            gaussians['inv_scales'].clamp_min_(0)
            gaussians['rotations'].clamp_(0,np.pi)
            gaussians['colors'].clamp_(0,1)
        # Update progress
        snr = psnr(output, target)
        ssi = ssim(output, target)
        pbar.set_postfix(dict(psnr=snr.item()), loss=f"{loss.item():.6f}", lr=scheduler.get_last_lr())
        train_stats['loss'].append(loss.item())
        train_stats['psnr'].append(snr.item())
        train_stats['ssim'].append(ssi.item())
        train_stats['loss_final'] = train_stats['loss'][-1]
        train_stats['psnr_final'] = train_stats['psnr'][-1]
        train_stats['ssim_final'] = train_stats['ssim'][-1]
        # Save current stats
        with open(f"{run_dir}/stats.json", 'w') as f:
            json.dump(train_stats, f, indent=4)
        # Save current output
        if (epoch+1) % epochs_per_frame == 0 or epoch == epochs-1:
            output_plot = output.detach().cpu().numpy().clip(0,1)
            anim_frames.append([anim_ax.imshow(output_plot)])
        if (save_indv_frames_every > 0 and (epoch+1) % save_indv_frames_every == 0) or epoch == epochs-1:
            anim_fig.savefig(f"{run_dir}/epochs/epoch{epoch+1:05d}.png")

    # Save GIF
    anim = animation.ArtistAnimation(anim_fig, anim_frames)
    anim.save(f"{run_dir}/movie.gif", writer=animation.PillowWriter())
    plt.close(anim_fig)

    # Save loss, PSNR, and SSIM curves
    curves_fig, curves_ax0 = plt.subplots()
    curves_ax0.plot(train_stats['loss'], color='C0')
    curves_ax0.set_xlabel("Epoch")
    curves_ax0.set_ylabel("Loss")
    curves_ax1 = curves_ax0.twinx()
    curves_ax1.plot(train_stats['psnr'], color='C1')
    curves_ax1.set_ylabel("PSNR")
    curves_fig.savefig(f"{run_dir}/loss_psnr.png")
    curves_ax1.clear()
    curves_ax1.plot(train_stats['ssim'], color='C1')
    curves_ax1.set_ylabel("SSIM")
    curves_fig.savefig(f"{run_dir}/loss_ssim.png")
    plt.close(curves_fig)

    print(f"Training outputs saved to: {run_dir}")
    return train_stats

if __name__=="__main__":
    # Run train() directly from command line
    parser = argparse.ArgumentParser(description="Fit a set of 2D Gaussians to a target image.")
    parser.add_argument('target_path', type=str)
    parser.add_argument('--n_gaussians', required=False, default=400, type=int)
    parser.add_argument('--out_width', required=False, default=200, type=int)
    parser.add_argument('--epochs', required=False, default=200, type=int)
    parser.add_argument('--out_dir', required=False, default="data/train", type=str)
    args = parser.parse_args()

    train(args.target_path, args.n_gaussians, args.out_width, args.epochs, args.out_dir)