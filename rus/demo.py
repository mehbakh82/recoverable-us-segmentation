import numpy as np
from PIL import Image

from .data import row


def synthetic(root):
    rng = np.random.default_rng(2411)
    rows = []
    for cohort, count in (("busbra", 30), ("breast", 10), ("uclm", 10)):
        folder = root / cohort
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            height, width = 40+index%5, 52+index%4
            yy, xx = np.ogrid[:height, :width]
            mask = ((xx-rng.integers(12, 37))/8)**2+((yy-rng.integers(10, 28))/6)**2 < 1
            normal = cohort != "busbra" and index%4 == 0
            if normal:
                mask[:] = False
            image = np.clip(rng.normal(120, 33, (height,width))-mask*75,0,255).astype(np.uint8)
            filename, target = folder/f"scan{index}.png", folder/f"mask{index}.png"
            Image.fromarray(image).save(filename)
            binary = (mask*255).astype(np.uint8)
            if cohort == "uclm":
                binary = np.stack([binary,np.zeros_like(binary),np.zeros_like(binary)], axis=-1)
            Image.fromarray(binary).save(target)
            rows.append(row(cohort, index//2, filename, [target], "normal" if normal else "benign",
                            1+(index//2)%5 if cohort=="busbra" else 0))
    return rows
