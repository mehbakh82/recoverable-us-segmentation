import gc
import math
import os
import time

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import label as components
from torch import nn

from .data import pixels, reference
from .shared import hash_file, patients, save, signature

LIMIT = 20_000_000_000
FEATURES = ("area", "components", "largest", "crop", "confidence", "maximum", "entropy", "empty")


def block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs), nn.ReLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs), nn.ReLU())


class UNet(nn.Module):
    def __init__(self, base):
        super().__init__()
        widths = [base * 2**index for index in range(5)]
        self.encoder = nn.ModuleList(block(inputs, outputs) for inputs, outputs in zip([1] + widths, widths))
        self.up = nn.ModuleList(nn.ConvTranspose2d(width, width // 2, 2, stride=2) for width in widths[:0:-1])
        self.decoder = nn.ModuleList(block(width, width // 2) for width in widths[:0:-1])
        self.output = nn.Conv2d(base, 1, 1)

    def forward(self, image):
        skips = []
        for encoder in self.encoder[:-1]:
            image = encoder(image)
            skips.append(image)
            image = nn.functional.max_pool2d(image, 2)
        image = self.encoder[-1](image)
        for up, decoder, skip in zip(self.up, self.decoder, reversed(skips)):
            image = decoder(torch.cat((up(image), skip), dim=1))
        return self.output(image)


def crop_box(mask, rng=None):
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0, width, height
    left, right, top, bottom = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad = [0.25] * 4 if rng is None else rng.uniform(0.05, 0.45, 4)
    return (max(0, int(left - pad[0] * (right-left))), max(0, int(top - pad[1] * (bottom-top))),
            min(width, math.ceil(right + pad[2] * (right-left))), min(height, math.ceil(bottom + pad[3] * (bottom-top))))


def cut(image, box):
    left, top, right, bottom = box
    return image[top:bottom, left:right]


def letterbox(image, size, mask=False):
    height, width = image.shape
    scale = size / max(width, height)
    resized = max(1, round(width*scale)), max(1, round(height*scale))
    left, top = (size-resized[0])//2, (size-resized[1])//2
    inner = (left, top, left+resized[0], top+resized[1])
    resample = Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
    canvas = np.zeros((size, size), dtype=np.float32)
    cut(canvas, inner)[:] = np.asarray(Image.fromarray(image.astype(np.uint8)).resize(resized, resample))
    return torch.from_numpy(canvas[None]) / (1 if mask else 255), inner


def example(item, kind, size, rng):
    image = pixels(item["image"], "L")
    target = reference(item, image.shape)
    if kind == "refine" and rng.random() >= 0.2:
        box = crop_box(target, rng)
        image, target = cut(image, box), cut(target, box)
    image, inner = letterbox(image, size)
    target, _ = letterbox(target, size, mask=True)
    valid = torch.zeros_like(target)
    left, top, right, bottom = inner
    valid[:, top:bottom, left:right] = 1
    return image, target, valid


def loss_fn(logits, target, valid):
    logits = logits.float()
    prob = logits.sigmoid() * valid
    dims = (1, 2, 3)
    dice = (2*(prob*target).sum(dims)+1) / (prob.sum(dims)+target.sum(dims)+1)
    bce = (nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")*valid).sum(dims)/valid.sum(dims)
    return (1-dice+bce).mean()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def select_device(demo):
    if demo:
        torch.set_num_threads(1)
        return torch.device("cpu")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Clinical training requires a BF16-capable allocated CUDA device. Use demo for CPU.")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, LIMIT/total), 0)
    return torch.device("cuda:0")


def amp(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def smoke(config, device, output):
    if device.type == "cpu":
        save(output, dict(device="CPU synthetic only", micro=4))
        return 4
    free, total = torch.cuda.mem_get_info()
    ledger = dict(device=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  free_before=free, visible_total=total, cap_bytes=LIMIT, attempts=[],
                  scheduler_limit=os.getenv("RUS_SCHEDULER_LIMIT", "unknown"),
                  shared=os.getenv("RUS_GPU_SHARED", "unknown"))
    def try_step(size, micro):
        model = UNet(config["base"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
        tensor = torch.zeros(micro, 1, size, size, device=device)
        with amp(device):
            loss = loss_fn(model(tensor), tensor, torch.ones_like(tensor))
        loss.backward()
        optimizer.step()
        sync(device)
    for micro in (4, 2, 1):
        passed = True
        for kind in ("first", "refine"):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            try:
                try_step(config[kind], micro)
            except torch.cuda.OutOfMemoryError:
                passed = False
            peak = torch.cuda.max_memory_reserved()
            passed = passed and peak <= LIMIT
            ledger["attempts"].append(dict(kind=kind, micro=micro, reserved=peak,
                allocated=torch.cuda.max_memory_allocated(), seconds=time.perf_counter()-started, passed=passed))
            save(output, ledger)
            if not passed:
                break
        if passed:
            ledger["micro"] = micro
            save(output, ledger)
            return micro
    raise RuntimeError("GPU allocation insufficient even for microbatch 1; study paused.")


def train_one(rows, kind, config, device, micro, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(config=config, kind=kind, patients=patients(rows),
                    inputs=signature([(r["id"], r["image_sha"], r["mask_sha"]) for r in rows]))
    torch.manual_seed(config["seed"])
    model = UNet(config["base"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    start, history = 0, []
    if path.exists():
        previous = torch.load(path, map_location=device, weights_only=True)
        if previous["meta"] != metadata:
            raise ValueError("Checkpoint and training inputs differ")
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        start, history = previous["epoch"], previous["history"]
        del previous
    for epoch in range(start, config["epochs"]):
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        rng = np.random.default_rng(config["seed"] + epoch)
        order = rng.permutation(len(rows))
        total_loss = 0.0
        for offset in range(0, len(order), config["effective"]):
            indices = order[offset:offset+config["effective"]]
            optimizer.zero_grad(set_to_none=True)
            for interval in range(0, len(indices), micro):
                batch = indices[interval:interval+micro]
                entries = [example(rows[index], kind, config[kind], rng) for index in batch]
                images, masks, valid = (torch.stack(values).to(device) for values in zip(*entries))
                with amp(device):
                    loss = loss_fn(model(images), masks, valid)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                (loss*(len(batch)/len(indices))).backward()
                total_loss += loss.item()*len(batch)
            optimizer.step()
        sync(device)
        history.append(dict(epoch=epoch+1, loss=total_loss/len(rows), seconds=time.perf_counter()-started,
                            micro=micro, allocated=torch.cuda.max_memory_allocated() if device.type=="cuda" else 0,
                            reserved=torch.cuda.max_memory_reserved() if device.type=="cuda" else 0))
        temp = path.with_suffix(".tmp")
        torch.save(dict(meta=metadata, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        epoch=epoch+1, history=history), temp)
        os.replace(temp, path)
        save(path.with_suffix(".json"), history)
    return hash_file(path)


def load_model(path, config, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["meta"]["config"] != config or checkpoint["epoch"] != config["epochs"]:
        raise ValueError("Incomplete or incompatible checkpoint")
    model = UNet(config["base"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint["meta"]


@torch.inference_mode()
def probability(model, image, size, device):
    tensor, inner = letterbox(image, size)
    with amp(device):
        result = model(tensor[None].to(device)).float().sigmoid()[0, 0].cpu().numpy()
    return np.asarray(Image.fromarray(cut(result, inner)).resize(image.shape[::-1], Image.Resampling.BILINEAR))


def features(prob, box):
    mask = prob >= 0.5
    labels, count = components(mask)
    sizes = np.bincount(labels.ravel())[1:]
    clipped = np.clip(prob.astype(np.float64), 1e-8, 1-1e-8)
    entropy = -clipped*np.log(clipped)-(1-clipped)*np.log1p(-clipped)
    active = prob >= 0.1
    return dict(area=float(mask.mean()), components=int(count), largest=float(sizes.max()/mask.size) if count else 0.,
                crop=(box[2]-box[0])*(box[3]-box[1])/mask.size,
                confidence=float(prob[mask].mean()) if count else 0., maximum=float(prob.max()),
                entropy=float(entropy[active].mean() if active.any() else entropy.max()), empty=int(not count))


def first_pass(model, image, config, device):
    prob = probability(model, image, config["first"], device)
    mask = prob >= config["threshold"]
    box = crop_box(mask)
    return mask, box, features(prob, box)


def second_pass(model, image, box, config, device):
    result = np.zeros_like(image, dtype=bool)
    cut(result, box)[:] = probability(model, cut(image, box), config["refine"], device) >= config["threshold"]
    return result
