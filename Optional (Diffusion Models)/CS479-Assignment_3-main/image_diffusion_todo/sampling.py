import argparse

import numpy as np
import torch
from dataset import tensor_to_pil_image
from ddpm import DiffusionModule
from pathlib import Path


def main(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    device = f"cuda:{args.gpu}"

    ddpm = DiffusionModule(None, None)
    ddpm.load(args.ckpt_path)
    ddpm.eval()
    ddpm = ddpm.to(device)

    total_num_samples = 500
    num_batches = int(np.ceil(total_num_samples / args.batch_size))

    for i in range(num_batches):
        sidx = i * args.batch_size
        eidx = min(sidx + args.batch_size, total_num_samples)
        B = eidx - sidx
        shape = (B, 3, 64, 64)
        samples = ddpm.p_sample_loop(shape)
        # Or, you can use DDIM sampling with any inference timesteps.
        # samples = ddpm.ddim_p_sample_loop(shape, num_inference_timesteps=50)
        pil_images = tensor_to_pil_image(samples)

        for j, img in zip(range(sidx, eidx), pil_images):
            img.save(save_dir / f"{j}.png")
            print(f"Saved the {j}-th image.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--save_dir", type=str)

    args = parser.parse_args()
    main(args)
