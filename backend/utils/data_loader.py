import tensorflow as tf
import numpy as np
import os
from tiff_processor import TiffProcessor

class TiffDataLoader:
    def __init__(self, data_path='datasets/', batch_size=8, image_size=256):
        self.data_path = data_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.tiff_processor = TiffProcessor()
    
    def load_tiff_pair(self, image1_path, image2_path, mask_path=None):
        """Load two temporal TIFF images and optional mask"""
        # Load TIFF images
        image1, _ = self.tiff_processor.read_tiff(image1_path)
        image2, _ = self.tiff_processor.read_tiff(image2_path)
        
        if image1 is None or image2 is None:
            return None, None
        
        # Extract RGB bands
        image1_rgb = self.tiff_processor.extract_bands(image1)
        image2_rgb = self.tiff_processor.extract_bands(image2)
        
        # Normalize
        image1_normalized = self.tiff_processor.normalize_tiff(image1_rgb)
        image2_normalized = self.tiff_processor.normalize_tiff(image2_rgb)
        
        # Resize
        image1_resized = cv2.resize(image1_normalized, (self.image_size, self.image_size))
        image2_resized = cv2.resize(image2_normalized, (self.image_size, self.image_size))
        
        # Load or create mask
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.image_size, self.image_size))
            mask = (mask > 127).astype(np.float32)
        else:
            # Create synthetic mask for training (you should have real masks)
            mask = self.tiff_processor.create_change_mask(image1_resized, image2_resized)
            if mask is not None:
                mask = mask.astype(np.float32) / 255.0
        
        # Combine images (using difference for better change detection)
        combined = np.abs(image1_resized.astype(float) - image2_resized.astype(float))
        combined = combined.astype(np.float32) / 255.0
        
        return combined, mask
    
    def discover_datasets(self):
        """Auto-discover TIFF datasets in the data directory"""
        datasets = []
        
        for root, dirs, files in os.walk(self.data_path):
            tiff_files = [f for f in files if f.lower().endswith(('.tif', '.tiff'))]
            
            # Group by timestamp or location
            locations = {}
            for file in tiff_files:
                # Extract location and timestamp from filename
                # Adjust this pattern based on your naming convention
                base_name = os.path.splitext(file)[0]
                parts = base_name.split('_')
                
                # Example: location_timestamp.tif
                if len(parts) >= 2:
                    location = parts[0]
                    timestamp = parts[1]
                    
                    if location not in locations:
                        locations[location] = []
                    locations[location].append({
                        'path': os.path.join(root, file),
                        'timestamp': timestamp
                    })
            
            # Create pairs for each location
            for location, images in locations.items():
                images.sort(key=lambda x: x['timestamp'])
                
                # Create consecutive pairs
                for i in range(len(images) - 1):
                    image1 = images[i]['path']
                    image2 = images[i + 1]['path']
                    
                    # Look for mask file
                    mask_path = os.path.join(root, f"mask_{location}_{images[i]['timestamp']}_{images[i+1]['timestamp']}.png")
                    if not os.path.exists(mask_path):
                        mask_path = None
                    
                    datasets.append((image1, image2, mask_path))
        
        return datasets
    
    def create_data_generator(self):
        """Create data generator for training"""
        datasets = self.discover_datasets()
        
        def generator():
            while True:
                # Shuffle datasets
                np.random.shuffle(datasets)
                
                for i in range(0, len(datasets), self.batch_size):
                    batch_datasets = datasets[i:i + self.batch_size]
                    batch_x = []
                    batch_y = []
                    
                    for dataset in batch_datasets:
                        image1_path, image2_path, mask_path = dataset
                        x, y = self.load_tiff_pair(image1_path, image2_path, mask_path)
                        
                        if x is not None and y is not None:
                            batch_x.append(x)
                            batch_y.append(y)
                    
                    if batch_x:
                        yield np.array(batch_x), np.array(batch_y)
        
        return generator()