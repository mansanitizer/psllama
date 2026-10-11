"""Image preparation as in the OpenVLA LIBERO evaluation: JPEG round trip,
Lanczos resize to 224, then the centre crop covering 90% of the area, resized back."""
import io, math
import numpy as np
from PIL import Image

def prepare(frame, center_crop=True):
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=95)
    img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB").resize((224, 224), Image.LANCZOS)
    if center_crop:
        side = 224 * math.sqrt(0.9)
        off = (224 - side) / 2
        img = img.resize((224, 224), Image.BILINEAR, box=(off, off, off + side, off + side))
    return np.asarray(img, dtype=np.uint8)
