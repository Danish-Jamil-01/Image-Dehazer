import PIL.Image as Image
import numpy as np
import time
from gf import guided_filter
import skimage.io as io
import cv2

class HazeRemoval(object):
    def __init__(self, omega=0.95, t0=0.1, radius=7, r=20, eps=0.001):
        self.omega = omega
        self.t0 = t0
        self.radius = radius
        self.r = r
        self.eps = eps

    def open_image(self, img_path):
        img = Image.open(img_path)
        self.src = np.array(img).astype(np.double) / 255.
        self.rows, self.cols, _ = self.src.shape
        self.dark = np.zeros((self.rows, self.cols), dtype=np.double)
        self.Alight = np.zeros((3), dtype=np.double)
        self.tran = np.zeros((self.rows, self.cols), dtype=np.double)
        self.dst = np.zeros_like(self.src, dtype=np.double)

    def get_dark_channel(self):
        print("Computing dark channel prior...")
        start = time.time()
        self.dark = np.min(self.src, axis=2)
        self.dark = guided_filter(self.dark, self.dark, self.radius, self.eps)
        print("Time:", time.time() - start)

    def get_air_light(self):
        print("Computing air light prior...")
        start = time.time()
        flat_dark = self.dark.flatten()
        flat_dark.sort()
        num = int(self.rows * self.cols * 0.001)
        threshold = flat_dark[-num]
        dark_mask = self.dark >= threshold
        self.Alight = np.mean(self.src[dark_mask], axis=0)
        print("Time:", time.time() - start)

    def get_transmission(self):
        print("Computing transmission...")
        start = time.time()
        self.tran = 1. - self.omega * (self.dark / np.max(self.dark))
        self.tran = guided_filter(self.src, self.tran, self.r, self.eps)
        print("Time:", time.time() - start)

    def recover(self):
        print("Recovering...")
        start = time.time()
        self.tran[self.tran < self.t0] = self.t0
        t = np.expand_dims(self.tran, axis=2)
        self.dst = ((self.src - self.Alight) / t) + self.Alight
        self.dst = np.clip(self.dst, 0, 1) * 255
        self.dst = self.dst.astype(np.uint8)
        print("Time:", time.time() - start)

    def show(self, output_dir="img"):
        cv2.imwrite(f"{output_dir}/src.jpg", (self.src * 255).astype(np.uint8)[:, :, ::-1])
        cv2.imwrite(f"{output_dir}/dark.jpg", (self.dark * 255).astype(np.uint8))
        cv2.imwrite(f"{output_dir}/tran.jpg", (self.tran * 255).astype(np.uint8))
        cv2.imwrite(f"{output_dir}/dst.jpg", self.dst[:, :, ::-1])
        io.imsave("dehazed.jpg", self.dst)

if __name__ == '__main__':
    import sys
    hr = HazeRemoval()
    hr.open_image(sys.argv[1])
    hr.get_dark_channel()
    hr.get_air_light()
    hr.get_transmission()
    hr.recover()
    hr.show()
