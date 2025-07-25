
import cv2
import numpy as np
import time
from .gf import guided_filter

class HazeRemover:
    def __init__(self, omega=0.95, t0=0.1, patch_size=15, r=60, eps=1e-3):
        self.omega = omega
        self.t0 = t0
        self.patch_size = patch_size
        self.r = r
        self.eps = eps
        self.intermediates = {}

    def process(self, img_bgr, enhance=True):
        img_rgb = img_bgr.astype(np.float64) / 255.0
        
        print("1. Computing dark channel prior...")
        dark_channel = self._get_dark_channel(img_rgb)
        
        print("2. Estimating atmospheric light...")
        atmospheric_light = self._get_atmospheric_light(img_rgb, dark_channel)
        
        print("3. Estimating and refining transmission map...")
        transmission = self._get_transmission(img_rgb, atmospheric_light)
        refined_transmission = self._refine_transmission(img_rgb, transmission)

        self.intermediates['dark_channel'] = (dark_channel * 255).astype(np.uint8)
        self.intermediates['transmission'] = (refined_transmission * 255).astype(np.uint8)

        print("4. Recovering haze-free image...")
        dehazed_img_rgb = self._recover_image(img_rgb, atmospheric_light, refined_transmission)

        dehazed_img_bgr = (np.clip(dehazed_img_rgb, 0, 1) * 255).astype(np.uint8)
        
        
        if enhance:
            print("5. Applying post-processing enhancement...")
            return self._enhance_image(dehazed_img_bgr)
        else:
            return dehazed_img_bgr

    def _get_dark_channel(self, img):
        min_channel = np.min(img, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.patch_size, self.patch_size))
        dark_channel = cv2.erode(min_channel, kernel)
        return dark_channel

    def _get_atmospheric_light(self, img, dark_channel):
        h, w, _ = img.shape
        num_pixels = int(max(1, h * w * 0.001))
        dark_flat = dark_channel.flatten()
        img_flat = img.reshape(h * w, 3)
        indices = np.argpartition(dark_flat, -num_pixels)[-num_pixels:]
        brightest_pixels = img_flat[indices]
        atmospheric_light = np.mean(brightest_pixels, axis=0)
        return atmospheric_light

    def _get_transmission(self, img, atmospheric_light):
        norm_hazy = img / atmospheric_light
        transmission = 1 - self.omega * self._get_dark_channel(norm_hazy)
        return transmission
        
    def _refine_transmission(self, img_rgb, transmission):
        gray_img = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray_img = gray_img.astype(np.float64) / 255.0
        refined_t = guided_filter(gray_img, transmission, self.r, self.eps)
        return refined_t

    def _recover_image(self, img, atmospheric_light, transmission):
        transmission_clipped = np.maximum(transmission, self.t0)
        t_reshaped = transmission_clipped[:, :, np.newaxis]
        A_reshaped = atmospheric_light[np.newaxis, np.newaxis, :]
        recovered_img = (img - A_reshaped) / t_reshaped + A_reshaped
        return recovered_img

    def _enhance_image(self, img_bgr):
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(img_lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        merged = cv2.merge((cl, a_channel, b_channel))
        enhanced_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return enhanced_img


if __name__ == '__main__':
    import sys
    import os
    
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    if len(sys.argv) != 2:
        print("Usage: python dehazer/haze_remover.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    hazy_image = cv2.imread(image_path)

    if hazy_image is None:
        print(f"Error: Could not open or find the image at '{image_path}'")
        sys.exit(1)

    remover = HazeRemover()
 
    dehazed_image = remover.process(hazy_image, enhance=True)

    cv2.imwrite(os.path.join(output_dir, "dehazed_output_cmd.png"), dehazed_image)
    cv2.imwrite(os.path.join(output_dir, "intermediate_dark_channel.png"), remover.intermediates['dark_channel'])
    cv2.imwrite(os.path.join(output_dir, "intermediate_transmission.png"), remover.intermediates['transmission'])

    print(f"\nProcessing complete. All images saved in '{output_dir}' folder.")