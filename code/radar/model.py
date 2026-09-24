"""Radar pillar network: learned PointNet pillar encoder -> small BEV U-Net -> heatmap + box-regression heads.

    points (B, N, 8) + mask (B, N)
      -> PillarEncoder : per-point MLP, then max over the points of each pillar, scattered onto the grid -> (B, 64, 128, 128)
      -> BEVBackbone   : small U-Net (128 -> 64 -> 32 -> back up to 128)                                 -> (B, 64, 128, 128)
      -> heads         : heatmap logits (B, 2, 128, 128), regression (B, 6, 128, 128)   (same layout as bev_targets.py)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from pillar_dataset import GRID

# rough magnitude of each raw column [x, y, z, vx, vy, v_r, rcs, dt], so the MLP sees numbers of order 1
SCALE = (50.0, 50.0, 5.0, 10.0, 10.0, 10.0, 20.0, 0.3)


class PillarEncoder(nn.Module):
    """PointNet on the points of every pillar. Only pillars that contain points are computed (the map is ~97% empty)."""

    def __init__(self, out_channels=64, grid=GRID):
        super().__init__()
        self.grid, self.out_channels = grid, out_channels
        self.register_buffer("scale", torch.tensor(SCALE))
        d_in = 8 + 2 + 3        # 8 raw columns + offset to the pillar centre (x, y) + offset to the pillar's mean point (x, y, z)
        self.mlp = nn.Sequential(nn.Linear(d_in, out_channels), nn.BatchNorm1d(out_channels), nn.ReLU(),
                                 nn.Linear(out_channels, out_channels), nn.BatchNorm1d(out_channels), nn.ReLU())

    def forward(self, points, mask):
        g, C = self.grid, self.out_channels
        B, N, _ = points.shape
        ix = torch.floor((points[..., 0] - g["x0"]) / g["cell"]).long()
        iy = torch.floor((points[..., 1] - g["y0"]) / g["cell"]).long()
        valid = mask & (ix >= 0) & (ix < g["nx"]) & (iy >= 0) & (iy < g["ny"])         # real points that fall inside the grid
        batch = torch.arange(B, device=points.device)[:, None].expand(B, N)
        pts, ix, iy, batch = points[valid], ix[valid], iy[valid], batch[valid]

        flat = batch * g["nx"] * g["ny"] + ix * g["ny"] + iy                             # which (frame, pillar) each point belongs to
        pillars, inv = torch.unique(flat, return_inverse=True)
        count = torch.zeros(len(pillars), device=pts.device).index_add_(0, inv, torch.ones_like(inv, dtype=torch.float))
        mean = torch.zeros(len(pillars), 3, device=pts.device).index_add_(0, inv, pts[:, :3]) / count[:, None]

        cx = g["x0"] + (ix.float() + 0.5) * g["cell"]
        cy = g["y0"] + (iy.float() + 0.5) * g["cell"]
        feats = torch.cat([pts / self.scale,
                           ((pts[:, 0] - cx) / g["cell"])[:, None], ((pts[:, 1] - cy) / g["cell"])[:, None],
                           (pts[:, :3] - mean[inv]) / pts.new_tensor([g["cell"], g["cell"], 1.0])], dim=1)

        h = self.mlp(feats)                                                              # (M, C) one vector per point
        per_pillar = h.new_zeros(len(pillars), C).scatter_reduce(0, inv[:, None].expand(-1, C), h, "amax", include_self=False)
        canvas = h.new_zeros(B * g["nx"] * g["ny"], C)
        canvas[pillars] = per_pillar                                                     # empty pillars stay 0
        return canvas.view(B, g["nx"], g["ny"], C).permute(0, 3, 1, 2).contiguous()


def block(c_in, c_out, stride=1):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class BEVBackbone(nn.Module):
    """Small U-Net. The coarse levels give each pixel a wide view (radar dots are sparse), the skip connections keep the detail."""

    def __init__(self, c_in=64):
        super().__init__()
        self.s1 = nn.Sequential(block(c_in, 64), block(64, 64))                                  # 128 x 128
        self.s2 = nn.Sequential(block(64, 128, 2), block(128, 128))                              # 64 x 64
        self.s3 = nn.Sequential(block(128, 192, 2), block(192, 192), block(192, 192))            # 32 x 32
        self.u3 = block(192, 128)
        self.m2 = block(256, 128)
        self.u2 = block(128, 64)
        self.m1 = block(128, 64)

    def forward(self, x):
        a = self.s1(x)
        b = self.s2(a)
        c = self.s3(b)
        b = self.m2(torch.cat([b, self.u3(F.interpolate(c, scale_factor=2, mode="bilinear", align_corners=False))], 1))
        a = self.m1(torch.cat([a, self.u2(F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False))], 1))
        return a


class RadarPillarNet(nn.Module):
    def __init__(self, n_classes=2, n_reg=6, channels=64):
        super().__init__()
        self.encoder = PillarEncoder(channels)
        self.backbone = BEVBackbone(channels)
        self.hm_head = nn.Sequential(block(64, 64), nn.Conv2d(64, n_classes, 1))
        self.reg_head = nn.Sequential(block(64, 64), nn.Conv2d(64, n_reg, 1))
        nn.init.constant_(self.hm_head[-1].bias, -2.19)      # start with "almost nothing here" (p ~ 0.1), standard for focal loss

    def forward(self, points, mask):
        feat = self.backbone(self.encoder(points, mask))
        return dict(hm=self.hm_head(feat), reg=self.reg_head(feat))
