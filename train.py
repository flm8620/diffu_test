import os
import random
import math
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils
from tqdm import tqdm

import hydra
from omegaconf import DictConfig


# -------------------- Data Generation --------------------
class NextDigitDataset(Dataset):
    """
    Generates pairs (x, y) where x is an image with a digit n,
    and y is the image with digit n+1.
    """

    def __init__(self, length=10000, img_size=64, max_digit=8, transform=None):
        self.length = length
        self.img_size = img_size
        self.max_digit = max_digit
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        n = random.randint(0, self.max_digit)
        x = self._make_img(n)
        y = self._make_img(n + 1)
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y

    def _make_img(self, digit: int) -> Image.Image:
        img = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = random.uniform(1.0, 2.0)
        thickness = random.randint(2, 4)
        text = str(digit)
        size = cv2.getTextSize(text, font, scale, thickness)[0]
        max_x = self.img_size - size[0] - 1
        max_y = self.img_size - size[1] - 1
        x = random.randint(0, max_x)
        y = random.randint(size[1], self.img_size - 1)
        cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness,
                    cv2.LINE_AA)
        return Image.fromarray(img)


# -------------------- Diffusion Utilities --------------------
class GaussianDiffusion(nn.Module):

    def __init__(self,
                 model,
                 timesteps=1000,
                 beta_start=1e-4,
                 beta_end=2e-2,
                 device='cuda'):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.device = device
        betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1 - alphas_cumprod))

    def forward(self, x0, cond):
        b = x0.size(0)
        t = torch.randint(0, self.timesteps, (b, ), device=self.device)
        noise = torch.randn_like(x0)
        x_t = self._q_sample(x0, t, noise)
        pred_noise = self.model(x_t, t, cond)
        return F.mse_loss(pred_noise, noise)

    def _q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        am1 = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return a * x0 + am1 * noise

    @torch.no_grad()
    def sample(self, cond):
        # sample with full or subsampled timesteps
        steps = steps or self.timesteps
        b, c, h, w = cond.size()
        x = torch.randn_like(cond)
        ts = list(range(self.timesteps))
        # reverse loop
        for i in reversed(ts):
            t = torch.full((b, ), i, device=self.device, dtype=torch.long)
            pred_noise = self.model(x, t, cond)
            beta = self.betas[t].view(-1, 1, 1, 1)
            a_cum = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            x = (1 / torch.sqrt(1 - beta)) * (
                x - beta / torch.sqrt(1 - a_cum) * pred_noise)
            if i > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)
        return x


# -------------------- Model Definition --------------------
class ResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(8, in_ch), num_channels=in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=min(8, out_ch),
                                  num_channels=out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_embed = nn.Linear(time_emb_dim, out_ch)
        self.res_conv = nn.Conv2d(in_ch, out_ch,
                                  1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.norm1(x)
        h = F.silu(self.conv1(h))
        h = h + self.time_embed(F.silu(t)).view(-1, h.size(1), 1, 1)
        h = self.norm2(h)
        h = F.silu(self.conv2(h))
        return h + self.res_conv(x)


class TransformerBlock(nn.Module):

    def __init__(self, dim, heads=4, mlp_dim=512):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, mlp_dim),
                                 nn.GELU(), nn.Linear(mlp_dim, dim))

    def forward(self, x):
        b, c, h, w = x.size()
        x_flat = x.view(b, c, -1).permute(0, 2, 1)
        x_norm = self.norm(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out
        x_flat = x_flat + self.mlp(x_flat)
        return x_flat.permute(0, 2, 1).view(b, c, h, w)


class UNet(nn.Module):

    def __init__(self, channels=3, base_ch=128, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, time_emb_dim), nn.GELU(),
                                      nn.Linear(time_emb_dim, time_emb_dim))
        # down
        self.rb1 = ResidualBlock(channels * 2, base_ch, time_emb_dim)
        self.rb2 = ResidualBlock(base_ch, base_ch * 2, time_emb_dim)
        self.rb3 = ResidualBlock(base_ch * 2, base_ch * 4, time_emb_dim)
        # bottleneck
        self.trans = TransformerBlock(base_ch * 4)
        # up
        self.rb4 = ResidualBlock(base_ch * 4 + base_ch * 2, base_ch * 2,
                                 time_emb_dim)
        self.rb5 = ResidualBlock(base_ch * 2 + base_ch, base_ch, time_emb_dim)
        self.final = nn.Conv2d(base_ch, channels, 1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, t, cond):
        t = t.float().unsqueeze(-1) / self.time_mlp[2].in_features
        t_emb = self.time_mlp(t)
        d1 = self.rb1(torch.cat([x, cond], dim=1), t_emb)
        d2 = self.rb2(self.pool(d1), t_emb)
        d3 = self.rb3(self.pool(d2), t_emb)
        b = self.trans(d3)
        u2 = self.up(b)
        u2 = self.rb4(torch.cat([u2, d2], dim=1), t_emb)
        u1 = self.up(u2)
        u1 = self.rb5(torch.cat([u1, d1], dim=1), t_emb)
        return self.final(u1)


# -------------------- Training & Utils --------------------
def train(model,
          diffusion,
          loader,
          optim,
          device,
          cfg: DictConfig,
          start_epoch=1):
    img_dir = 'images'
    os.makedirs(img_dir, exist_ok=True)
    writer = SummaryWriter(log_dir='.')
    model.train()
    for epoch in range(start_epoch, cfg.train.epochs + 1):
        total = 0
        print('Training...')
        for x, y in tqdm(loader, desc=f'Epoch {epoch}'):
            x, y = x.to(device), y.to(device)
            loss = diffusion(y, x)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item()
        avg = total / len(loader)
        print(f'Epoch {epoch} loss {avg:.4f}')
        writer.add_scalar('Loss', avg, epoch)
        if epoch % cfg.train.viz_per_epoch == 0:
            x_s, y_s = next(iter(loader))
            x_s, y_s = x_s.to(device), y_s.to(device)
            print('Sampling...')
            p_s = diffusion.sample(x_s)
            vis_x = (x_s + 1) / 2
            vis_y = (y_s + 1) / 2
            vis_p = (p_s.clamp(-1, 1) + 1) / 2
            print('Logging...')
            writer.add_image('Input', utils.make_grid(vis_x, nrow=4), epoch)
            writer.add_image('GT', utils.make_grid(vis_y, nrow=4), epoch)
            writer.add_image('Pred', utils.make_grid(vis_p, nrow=4), epoch)
            # save images
            print('Saving images...')
            to_pil = transforms.ToPILImage()
            to_pil(utils.make_grid(vis_x.cpu(), nrow=4)).save(
                os.path.join(img_dir, f'epoch{epoch}_in.png'))
            to_pil(utils.make_grid(vis_y.cpu(), nrow=4)).save(
                os.path.join(img_dir, f'epoch{epoch}_gt.png'))
            to_pil(utils.make_grid(vis_p.cpu(), nrow=4)).save(
                os.path.join(img_dir, f'epoch{epoch}_pred.png'))
        # save checkpoint
        print('Saving checkpoint...')
        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optim.state_dict()
        }
        torch.save(ckpt, 'checkpoint.pth')
    writer.close()


# -------------------- Main --------------------
@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.general.seed)
    device = torch.device(cfg.general.device)

    assert torch.cuda.is_available(), 'CUDA required'
    device = torch.device('cuda')

    # Data
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5] * 3, [0.5] * 3)])
    dataset = NextDigitDataset(length=cfg.train.dataset_length,
                               img_size=cfg.train.img_size,
                               transform=transform)
    loader = DataLoader(dataset,
                        batch_size=cfg.train.batch_size,
                        shuffle=True,
                        num_workers=cfg.train.num_workers,
                        pin_memory=True)

    # Model + Diffusion
    model = UNet(channels=3,
                 base_ch=cfg.train.base_ch,
                 time_emb_dim=cfg.train.time_emb_dim).to(device)
    diffusion = GaussianDiffusion(model,
                                  timesteps=cfg.train.timesteps,
                                  device='cuda').to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    start_epoch = 1
    if cfg.general.resume:
        ckpt_path = os.path.join(os.getcwd(), "checkpoint.pth")
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt['model'])
            optim.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt['epoch'] + 1
            print(f"Resuming from epoch {ckpt['epoch']}")

    # Train
    train(model, diffusion, loader, optim, device, cfg, start_epoch)


if __name__ == '__main__':
    main()
