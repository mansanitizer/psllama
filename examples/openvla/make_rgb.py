import sys, numpy as np
from PIL import Image
from vla_image import prepare
a = prepare(np.asarray(Image.open(sys.argv[1]).convert("RGB")))
a.tofile(sys.argv[2]); print("wrote", sys.argv[2], a.shape)
