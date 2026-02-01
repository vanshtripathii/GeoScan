import rasterio
import numpy as np
import cv2
import os
from PIL import Image

class TiffProcessor:
    def __init__(self):
        pass
    
    def read_tiff(self, file_path):
        """Read TIFF file and return as numpy array"""
        try:
            with rasterio.open(file_path) as src:
                # Read all bands
                image = src.read()
                # Convert to (H, W, C) format
                image = np.transpose(image, (1, 2, 0))
                return image, src.profile
        except Exception as e:
            print(f"Error reading TIFF: {e}")
            return None, None
    
    def extract_bands(self, image, bands=[3, 2, 1]):  # Common RGB bands for satellite
        """Extract specific bands to create RGB image"""
        if image.shape[2] < max(bands):
            # If not enough bands, use first 3 bands
            bands = [0, 1, 2]
        
        rgb_image = image[:, :, bands]
        return rgb_image
    
    def normalize_tiff(self, image):
        """Normalize TIFF image for visualization and processing"""
        # Handle different bit depths
        if image.dtype == np.uint16:
            image = (image / 256).astype(np.uint8)
        
        # Normalize to 0-255 range for each band
        normalized = np.zeros_like(image, dtype=np.float32)
        for i in range(image.shape[2]):
            band = image[:, :, i]
            if band.max() > band.min():
                normalized[:, :, i] = (band - band.min()) / (band.max() - band.min()) * 255
            else:
                normalized[:, :, i] = band
        
        return normalized.astype(np.uint8)
    
    def create_change_mask(self, image1, image2, method='difference'):
        """Create change mask from two temporal images (if you don't have ground truth)"""
        if method == 'difference':
            # Simple difference method
            diff = np.abs(image1.astype(float) - image2.astype(float))
            change_mask = np.mean(diff, axis=2) > 30  # Threshold
            return change_mask.astype(np.uint8) * 255
        else:
            # More sophisticated methods can be added
            return None

    def save_as_png(self, tiff_path, output_path):
        """Convert TIFF to PNG for web display"""
        image, _ = self.read_tiff(tiff_path)
        if image is not None:
            rgb_image = self.extract_bands(image)
            normalized = self.normalize_tiff(rgb_image)
            cv2.imwrite(output_path, cv2.cvtColor(normalized, cv2.COLOR_RGB2BGR))
            return True
        return False