from requirements import *

image_dir = r"E:\image"
mask_dir = r"E:\mask"

TARGET_SHAPE = (128, 128, 128)  

def load_nifti(file_path):
    nifti_img = nib.load(file_path) 
    img_array = np.array(nifti_img.get_fdata(), dtype=np.float32) 
    return img_array

def resize_volume(volume, target_shape=TARGET_SHAPE):
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    resized_volume = zoom(volume, zoom_factors, order=1) 
    return resized_volume

class NiftiDataset(Dataset):
    def __init__(self, image_dir, mask_dir, target_shape=TARGET_SHAPE, augment=False):
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_shape = target_shape
        self.augment = augment

        self.augmentations = [
            RandFlip(prob=0.5, spatial_axis=0),
            RandRotate(range_x=0.1, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
            RandGaussianNoise(prob=0.2, mean=0.0, std=0.1)
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = load_nifti(img_path)
        mask = load_nifti(mask_path)

        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

        image = resize_volume(image, self.target_shape)
        mask = resize_volume(mask, self.target_shape)

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Shape: (1, H, W, D)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        if self.augment:
            for transform in self.augmentations:
                image = transform(image)  
                mask = transform(mask) 

        return image, mask

train_dataset = NiftiDataset(image_dir, mask_dir, augment=True)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

print(f"Training set size: {len(train_dataset)}")

for images, masks in train_dataloader:
    print(f"Image Shape: {images.shape}, Mask Shape: {masks.shape}")
    break  # Check only one batch

# ---------------- For Validation ----------------
# %%
image_dir = r"E:\valid\image"
mask_dir = r"E:\valid\mask"

TARGET_SHAPE = (128, 128, 128)  

def load_nifti(file_path):
    nifti_img = nib.load(file_path)  
    img_array = np.array(nifti_img.get_fdata(), dtype=np.float32) 
    return img_array

def resize_volume(volume, target_shape=TARGET_SHAPE):
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    resized_volume = zoom(volume, zoom_factors, order=1)  
    return resized_volume

class NiftiDataset(Dataset):
    def __init__(self, image_dir, mask_dir, target_shape=TARGET_SHAPE, augment=False):
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_shape = target_shape
        self.augment = augment

        self.augmentations = [
            RandFlip(prob=0.5, spatial_axis=0),
            RandRotate(range_x=0.1, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
            RandGaussianNoise(prob=0.2, mean=0.0, std=0.1)
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = load_nifti(img_path)
        mask = load_nifti(mask_path)

        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

        image = resize_volume(image, self.target_shape)
        mask = resize_volume(mask, self.target_shape)

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Shape: (1, H, W, D)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        if self.augment:
            for transform in self.augmentations:
                image = transform(image)
                mask = transform(mask) 

        return image, mask

val_dataset = NiftiDataset(image_dir, mask_dir, augment=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

print(f"Validation set size: {len(val_dataset)}")

for images, masks in val_dataloader:
    print(f"Image Shape: {images.shape}, Mask Shape: {masks.shape}")
    break  # Check only one batch

# ---------------- For Testing ----------------
image_dir = r"E:\test\image"
mask_dir = r"E:\test\mask"

TARGET_SHAPE = (128, 128, 128)  

def load_nifti(file_path):
    nifti_img = nib.load(file_path) 
    img_array = np.array(nifti_img.get_fdata(), dtype=np.float32) 
    return img_array

def resize_volume(volume, target_shape=TARGET_SHAPE):
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    resized_volume = zoom(volume, zoom_factors, order=1) 
    return resized_volume

class NiftiDataset(Dataset):
    def __init__(self, image_dir, mask_dir, target_shape=TARGET_SHAPE, augment=False):
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_shape = target_shape
        self.augment = augment

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = load_nifti(img_path)
        mask = load_nifti(mask_path)

        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

        image = resize_volume(image, self.target_shape)
        mask = resize_volume(mask, self.target_shape)

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Shape: (1, H, W, D)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        if self.augment:
            for transform in self.augmentations:
                image = transform(image)  
                mask = transform(mask)  

        return image, mask

test_dataset = NiftiDataset(image_dir, mask_dir, augment=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

print(f"Training set size: {len(test_dataset)}")

for images, masks in test_dataloader:
    print(f"Image Shape: {images.shape}, Mask Shape: {masks.shape}")
    break  # Check only one batch