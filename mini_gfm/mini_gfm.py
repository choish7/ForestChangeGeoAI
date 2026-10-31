#!/usr/bin/env python3
# coding: utf-8
"""CarbonAce Mini-GFM: 3-year raster masked self-supervised pretraining and inference.

Supports 18-band (6 x 3), 14-band (4 x 3 + terrain), 11-band
(3 x 3 + terrain), and 15-band (5 x 3; NDBI inverted to NDMI) GeoTIFFs.
"""
from __future__ import annotations
import argparse
import glob
import json
import math
import random
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Common input logical order:
# 0..8: [NDVI, NDWI, NDMI] x 3 years; 9: Slope degrees, 10: Elevation m
LOGICAL_BANDS = [f'{b}_{year}' for year in ['Y1', 'Y2', 'Y3']
                 for b in ['NDVI', 'NDWI', 'NDMI']] + ['Slope', 'Elevation']
NODATA = -9999.0


def source_mapping(count):
    if count == 18:  # [NDVI NDWI NDBI NDMI Slope Elev] x 3 years
        return [1, 2, 4, 7, 8, 10, 13, 14, 16, 5, 6], []
    if count == 14:  # [NDVI NDWI NDBI NDMI] x 3, Slope, Elev
        return [1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14], []
    if count == 11:  # [NDVI NDWI NDMI] x 3, Slope, Elev
        return list(range(1, 12)), []
    if count == 15:  # [NDVI NDWI NDBI Slope Elev] x 3
        return [1, 2, 3, 6, 7, 8, 11, 12, 13, 4, 5], [2, 5, 8]
    raise ValueError(f'Expected a 18/14/11/15-band GeoTIFF, found {count} bands')


def check_raster(path):
    with rasterio.open(path) as ds:
        source_mapping(ds.count)
        if ds.crs is None:
            raise ValueError(f'Raster has no CRS: {path}')
        if ds.width < 16 or ds.height < 16:
            raise ValueError(f'Raster too small: {path}')
        if ds.count == 18:
            expected = ['NDVI_', 'NDWI_', 'NDBI_', 'NDMI_', 'Slope_', 'Elevation_'] * 3
            desc = ds.descriptions
            # Earth Engine often writes band names. If present, catch wrong order.
            if all(desc):
                problems = [(i + 1, x, e) for i, (x, e) in enumerate(zip(desc, expected))
                            if not str(x).startswith(e)]
                if problems:
                    raise ValueError(f'18-band order differs from GEE export: {problems[:5]}')


def read_logical(ds, window):
    indexes, neg = source_mapping(ds.count)
    # rasterio mask respects per-band nodata and out-of-bound pixels.
    ma = ds.read(indexes=indexes, window=window, masked=True,
                 boundless=True, fill_value=NODATA).astype('float32')
    arr = np.asarray(ma.filled(np.nan), dtype=np.float32)
    for i in neg:
        arr[i] *= -1.0
    valid = np.isfinite(arr).all(axis=0)
    valid &= (np.abs(arr[:9]) <= 1.05).all(axis=0)
    valid &= (arr[9] >= 0) & (arr[9] <= 90)
    valid &= (arr[10] >= -500) & (arr[10] <= 10000)
    return arr, valid


def normalize(arr, valid, stats):
    x = np.zeros_like(arr, dtype=np.float32)
    x[:9] = np.clip(arr[:9], -1.0, 1.0) * 2.0
    x[9] = np.clip((arr[9] - stats['slope_mean']) / stats['slope_std'], -5, 5)
    x[10] = np.clip((arr[10] - stats['elev_mean']) / stats['elev_std'], -5, 5)
    x[:, ~valid] = 0.0
    x[~np.isfinite(x)] = 0.0
    return x


def available_tiles(paths, patch):
    tiles = []
    for path in paths:
        check_raster(path)
        with rasterio.open(path) as ds:
            if ds.width < patch or ds.height < patch:
                raise ValueError(f'{path}: dimensions must be >= --patch ({patch})')
            for y in range(0, ds.height - patch + 1, patch):
                for x in range(0, ds.width - patch + 1, patch):
                    tiles.append((str(path), x, y))
    if len(tiles) < 2:
        raise ValueError('Need >=2 non-overlapping patches for train/validation; '
                         'reduce --patch or use larger/more GeoTIFFs')
    return tiles


def filter_tiles(tiles, patch, min_valid=0.5):
    kept = []
    for path, x, y in tiles:
        with rasterio.open(path) as ds:
            # Downsample window in preview to quickly reject cloudy/nodata chips
            ma = ds.read(1, window=Window(x, y, patch, patch),
                         out_shape=(16, 16), masked=True)
            frac = (~np.ma.getmaskarray(ma) & np.isfinite(ma.filled(np.nan))).mean()
            if frac >= min_valid:
                kept.append((path, x, y))
    if len(kept) < 2:
        raise ValueError('Fewer than 2 apparently valid patches in rasters')
    return kept


def make_split(tiles, seed, val_fraction=0.2):
    # Spatially non-overlapping tiles, but neighboring context may still be correlated.
    tiles = tiles.copy()
    random.Random(seed).shuffle(tiles)
    n_val = max(1, int(round(len(tiles) * val_fraction)))
    return tiles[n_val:], tiles[:n_val]


def fit_terrain_stats(tiles, patch, seed):
    rng = random.Random(seed)
    sample = rng.sample(tiles, min(48, len(tiles)))
    vals = [[], []]
    for path, x, y in sample:
        with rasterio.open(path) as ds:
            arr, valid = read_logical(ds, Window(x, y, patch, patch))
            # Maximum ~100k pixels per variable to limit memory.
            iy, ix = np.where(valid)
            if len(iy) == 0:
                continue
            take = np.linspace(0, len(iy) - 1, min(2048, len(iy)), dtype=int)
            vals[0].append(arr[9, iy[take], ix[take]])
            vals[1].append(arr[10, iy[take], ix[take]])
    if not vals[0]:
        raise ValueError('No valid training pixels; inspect nodata and clouds')
    slope = np.concatenate(vals[0])
    elev = np.concatenate(vals[1])
    return {'slope_mean': float(np.mean(slope)),
            'slope_std': float(max(1.0, np.std(slope))),
            'elev_mean': float(np.mean(elev)),
            'elev_std': float(max(1.0, np.std(elev)))}


class GeoDataset(Dataset):
    def __init__(self, tiles, patch, stats, samples, augment=False):
        self.tiles = tiles
        self.patch = patch
        self.stats = stats
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        # Random tile per training step; validation deterministically cycles tiles.
        if self.augment:
            path, x, y = random.choice(self.tiles)
        else:
            path, x, y = self.tiles[idx % len(self.tiles)]
        with rasterio.open(path) as ds:
            arr, valid = read_logical(ds, Window(x, y, self.patch, self.patch))
        xdata = normalize(arr, valid, self.stats)
        if self.augment:
            k = random.randrange(4)
            xdata = np.rot90(xdata, k, axes=(1, 2))
            valid = np.rot90(valid, k)
            if random.random() < 0.5:
                xdata = xdata[:, :, ::-1]
                valid = valid[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(xdata)), torch.from_numpy(
            np.ascontiguousarray(valid[None].astype('float32')))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(4, channels), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(4, channels))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class TemporalMaskedAE(nn.Module):
    def __init__(self, dim=24):
        super().__init__()
        if dim % 4:
            raise ValueError('--dim must be a multiple of 4')
        # Three index channels, 2 terrain channels, one spatial-mask indicator
        self.encoder = nn.Sequential(
            nn.Conv2d(6, dim, 3, padding=1), nn.GELU(),
            ResidualBlock(dim), ResidualBlock(dim))
        self.decoder = nn.Sequential(
            nn.Conv2d(3 * dim, dim * 2, 3, padding=1), nn.GELU(),
            ResidualBlock(dim * 2), nn.Conv2d(dim * 2, 9, 1))

    def encode_years(self, x, spatial_mask=None):
        # x: [B,11,H,W]; spatial_mask: [B,3,H,W]
        topo = x[:, 9:11]
        emb = []
        for y in range(3):
            idx = x[:, y * 3:(y + 1) * 3]
            if spatial_mask is None:
                m = torch.zeros_like(idx[:, :1])
            else:
                m = spatial_mask[:, y:y + 1]
                idx = idx * (1.0 - m)
            emb.append(self.encoder(torch.cat((idx, topo, m), dim=1)))
        return emb

    def forward(self, x, spatial_mask):
        emb = self.encode_years(x, spatial_mask)
        return self.decoder(torch.cat(emb, dim=1))


def block_mask(batch, hw, ratio, block, device):
    h, w = hw
    mh = math.ceil(h / block)
    mw = math.ceil(w / block)
    small = torch.rand(batch, 3, mh, mw, device=device) < ratio
    return torch.nn.functional.interpolate(small.float(), size=(h, w), mode='nearest')


def masked_loss(pred, x, valid, mask):
    # Only masked spectral index pixels provide reconstruction supervision.
    use = mask.repeat_interleave(3, dim=1) * valid
    return (torch.abs(pred - x[:, :9]) * use).sum() / use.sum().clamp_min(1.0)


def find_inputs(patterns):
    paths = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])
    paths = [str(Path(p).resolve()) for p in dict.fromkeys(paths)]
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f'Input files not found: {missing}')
    return paths


def choose_device(value):
    if value == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if value == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; select --device cpu')
    return torch.device(value)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    paths = find_inputs(args.inputs)
    tiles = filter_tiles(available_tiles(paths, args.patch), args.patch)
    train_tiles, val_tiles = make_split(tiles, args.seed)
    stats = fit_terrain_stats(train_tiles, args.patch, args.seed)
    print('Device:', device)
    print('Files:', len(paths), '| train tiles:', len(train_tiles), '| val tiles:', len(val_tiles))
    print('Terrain normalization:', json.dumps(stats, ensure_ascii=False))
    train_ds = GeoDataset(train_tiles, args.patch, stats, args.steps * args.batch, augment=True)
    val_ds = GeoDataset(val_tiles, args.patch, stats, min(16, max(1, len(val_tiles))), augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    model = TemporalMaskedAE(args.dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr = []
        for x, valid in train_dl:
            x, valid = x.to(device), valid.to(device)
            mask = block_mask(x.shape[0], x.shape[-2:], args.mask_ratio, args.mask_block, device)
            opt.zero_grad(set_to_none=True)
            pred = model(x, mask)
            loss = masked_loss(pred, x, valid, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr.append(float(loss.detach().cpu()))
        model.eval()
        vl = []
        with torch.no_grad():
            for x, valid in val_dl:
                x, valid = x.to(device), valid.to(device)
                mask = block_mask(x.shape[0], x.shape[-2:], args.mask_ratio,
                                  args.mask_block, device)
                pred = model(x, mask)
                vl.append(float(masked_loss(pred, x, valid, mask).cpu()))
        tr_loss = float(np.mean(tr))
        vl_loss = float(np.mean(vl))
        print(f'Epoch {epoch:03d}/{args.epochs}: masked MAE train={tr_loss:.5f} val={vl_loss:.5f}', flush=True)
        history.append({'epoch': epoch, 'train_mae': tr_loss, 'val_mae': vl_loss})
        if vl_loss < best:
            best = vl_loss
            torch.save({
                'model_state': model.state_dict(), 'stats': stats,
                'dim': args.dim, 'years': list(args.years),
                'input_layout': LOGICAL_BANDS,
                'architecture': 'temporal_shared_encoder_masked_autoencoder_v1',
                'best_val_masked_mae': best}, outdir / 'best_model.pt')
            print('  saved:', outdir / 'best_model.pt')
    (outdir / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    (outdir / 'config.json').write_text(json.dumps({k: v for k, v in vars(args).items() if k != 'func'}, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Training finished. Best masked MAE:', round(best, 6))


def tiles_for_inference(width, height, size):
    for y in range(0, height, size):
        for x in range(0, width, size):
            yield Window(x, y, min(size, width - x), min(size, height - y))


def change_scores(x, emb):
    # Per-pixel embedding distance = anomaly score, not a calibrated probability.
    ds = []
    for a, b in [(0, 1), (1, 2), (0, 2)]:
        ds.append((emb[a] - emb[b]).square().mean(dim=1).clamp_min(0).sqrt())
    # Raw changes in reflectance indices, not learned anomaly scores.
    dndvi = (x[:, 6] - x[:, 0]) / 2.0  # x is NDVI / 0.5
    dndmi = (x[:, 8] - x[:, 2]) / 2.0
    return torch.stack([*ds, dndvi, dndmi], dim=1)


def output_prefix(input_path, prefix=None):
    prefix = Path(input_path).stem if prefix is None else prefix
    if not prefix or any(c in prefix for c in '/\\:<>|?*'):
        raise ValueError('output prefix must be a non-empty filename')
    return prefix.rstrip('_') + '_'


def infer(args):
    prefix = output_prefix(args.input, getattr(args, 'output_prefix', None))
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = TemporalMaskedAE(int(checkpoint['dim']))
    model.load_state_dict(checkpoint['model_state'])
    model.to(device).eval()
    stats = checkpoint['stats']
    path = str(Path(args.input).resolve())
    check_raster(path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        profile.update(driver='GTiff', dtype='float32', count=5,
                       nodata=NODATA, compress='deflate', predictor=3, tiled=True,
                       blockxsize=256, blockysize=256, interleave='band')
        output_file = outdir / f'{prefix}change_scores.tif'
        emb_file = outdir / f'{prefix}embedding_3year_mean.tif'
        emb_profile = profile.copy()
        emb_profile['count'] = int(checkpoint['dim'])
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.descriptions = (
                'EmbeddingDistance_Y1_Y2', 'EmbeddingDistance_Y2_Y3',
                'EmbeddingDistance_Y1_Y3', 'DeltaNDVI_Y3_minus_Y1',
                'DeltaNDMI_Y3_minus_Y1')
            dst.update_tags(years=','.join(str(y) for y in checkpoint['years']),
                            notes='embedding distance is NOT change probability')
            # optional per-pixel embedding: average three year features
            if args.save_embedding:
                embdst = rasterio.open(emb_file, 'w', **emb_profile)
                embdst.descriptions = tuple(f'Embedding_{i:02d}' for i in range(int(checkpoint['dim'])))
            else:
                embdst = None
            try:
                for n, core in enumerate(tiles_for_inference(src.width, src.height, args.tile), 1):
                    pad = args.halo
                    expanded = Window(core.col_off - pad, core.row_off - pad,
                                      core.width + 2 * pad, core.height + 2 * pad)
                    raw, valid = read_logical(src, expanded)
                    x = normalize(raw, valid, stats)
                    x_t = torch.from_numpy(np.ascontiguousarray(x[None])).to(device)
                    with torch.inference_mode():
                        embs = model.encode_years(x_t)
                        scores = change_scores(x_t, embs)[0].cpu().numpy()
                        if embdst is not None:
                            combined = torch.stack(embs, dim=0).mean(dim=0)[0].cpu().numpy()
                    hh, ww = int(core.height), int(core.width)
                    crop = (slice(pad, pad + hh), slice(pad, pad + ww))
                    sc = scores[:, crop[0], crop[1]].copy()
                    core_valid = valid[crop]
                    sc[:, ~core_valid] = NODATA
                    dst.write(sc.astype('float32'), window=core)
                    if embdst is not None:
                        out = combined[:, crop[0], crop[1]].copy()
                        out[:, ~core_valid] = NODATA
                        embdst.write(out.astype('float32'), window=core)
                    if n % 20 == 0:
                        print('  processed tiles:', n, flush=True)
            finally:
                if embdst is not None:
                    embdst.close()
    print('Saved:', output_file)
    if args.save_embedding:
        print('Saved:', emb_file)


def main():
    p = argparse.ArgumentParser(description='CarbonAce Mini-GFM masked pretraining and inference')
    sub = p.add_subparsers(dest='mode', required=True)
    tr = sub.add_parser('train', help='Self-supervised masked reconstruction training')
    tr.add_argument('--inputs', nargs='+', required=True, help='GeoTIFF paths or glob patterns')
    tr.add_argument('--outdir', default='runs/mini_gfm')
    tr.add_argument('--years', nargs=3, type=int, default=[2023, 2024, 2025])
    tr.add_argument('--epochs', type=int, default=30)
    tr.add_argument('--steps', type=int, default=100, help='batches per epoch')
    tr.add_argument('--batch', type=int, default=4)
    tr.add_argument('--patch', type=int, default=128)
    tr.add_argument('--dim', type=int, default=24)
    tr.add_argument('--lr', type=float, default=0.001)
    tr.add_argument('--mask-ratio', type=float, default=0.35)
    tr.add_argument('--mask-block', type=int, default=16)
    tr.add_argument('--workers', type=int, default=0)
    tr.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    tr.add_argument('--seed', type=int, default=42)
    tr.set_defaults(func=train)
    inf = sub.add_parser('infer', help='Inference to georeferenced change-score GeoTIFF')
    inf.add_argument('--input', required=True)
    inf.add_argument('--checkpoint', required=True)
    inf.add_argument('--outdir', default='outputs')
    inf.add_argument('--tile', type=int, default=256)
    inf.add_argument('--halo', type=int, default=16)
    inf.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    inf.add_argument('--save-embedding', action='store_true')
    inf.add_argument('--output-prefix', default=None, help='Filename prefix (default: input stem)')
    inf.set_defaults(func=infer)
    args = p.parse_args()
    if args.mode == 'train' and not 0.05 <= args.mask_ratio <= 0.9:
        p.error('--mask-ratio should be between 0.05 and 0.9')
    if args.mode == 'infer' and (args.tile < 32 or args.halo < 8):
        p.error('--tile >= 32, --halo >= 8 are required')
    args.func(args)


if __name__ == '__main__':
    main()
