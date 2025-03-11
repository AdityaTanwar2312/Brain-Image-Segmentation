from requirements import *
from preprocess import *

# Load trained model
model_path = "checkpoint.pth"
device = torch.device("cpu")

model = AttentionUnet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3, 
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    dropout=0.3
)

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# %%
def apply_postprocessing(mask):
    mask = mask.squeeze().cpu().numpy()
    mask = np.argmax(mask, axis=0)
    
    for class_id in range(1, mask.max() + 1):
        binary_mask = (mask == class_id).astype(np.uint8)
        binary_mask = binary_fill_holes(binary_mask)
        binary_mask = binary_closing(binary_mask, structure=np.ones((3, 3, 3)))
        labeled_mask, num_features = label(binary_mask)
        
        if num_features > 1:
            largest_component = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1
            binary_mask = (labeled_mask == largest_component).astype(np.uint8)
        
        mask = np.where(mask == class_id, binary_mask * class_id, mask)

    mask = gaussian_filter(mask.astype(float), sigma=1)
    return mask

save_dir = "E:/test_results"
os.makedirs(save_dir, exist_ok=True)

def run_inference(test_dataloader, save_dir):
    for i, (image, _) in enumerate(test_dataloader):  # Ignore masks in test dataset
        image = image.to(device, dtype=torch.float32)
        
        with torch.no_grad():
            output = model(image)
            output = F.softmax(output, dim=1)
        
        processed_mask = apply_postprocessing(output)
        
        output_nifti = nib.Nifti1Image(processed_mask.astype(np.float32), affine=np.eye(4))
        nib.save(output_nifti, os.path.join(save_dir, f"segmented_{i}.nii.gz"))
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image.squeeze().cpu().numpy()[image.shape[2]//2], cmap='gray')
        plt.title("Original Image")

        plt.subplot(1, 2, 2)
        plt.imshow(processed_mask[processed_mask.shape[2]//2], cmap='jet', alpha=0.5)
        plt.title("Segmented Mask")
        plt.show()

    print("Segmentation completed. Processed masks saved.")

run_inference(test_dataloader, save_dir)

# %%
mask_dir = r"E:\temp_extracted_masks"
output_csv = "segmentation_features.csv"

def extract_features(mask_path):
    try:
        mask_nifti = nib.load(mask_path)
        mask = mask_nifti.get_fdata()

        if mask is None or mask.size == 0:
            print(f" Empty or invalid mask: {mask_path}")
            return None

        # Get nonzero voxel mask
        nonzero_mask = mask > 0

        # Compute volume (number of voxels)
        volume = np.sum(nonzero_mask)

        # Compute intensity statistics (mean, std of nonzero voxels)
        mean_intensity = np.mean(mask[nonzero_mask]) if volume > 0 else 0
        std_intensity = np.std(mask[nonzero_mask]) if volume > 0 else 0

        return {"Mask Name": os.path.basename(mask_path), "Volume": volume, "Mean Intensity": mean_intensity, "Std Intensity": std_intensity}

    except Exception as e:
        print(f" Error processing {mask_path}: {e}")
        return None

mask_files = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]

if not mask_files:
    print(" No NIfTI files found! Check the folder path.")
else:
    print(f" Found {len(mask_files)} NIfTI files for processing.")

features_list = []
print("Extracting segmentation features...")
for mask_path in tqdm(mask_files, desc="Processing", unit="mask"):
    features = extract_features(mask_path)
    if features:
        features_list.append(features)

if features_list:
    df = pd.DataFrame(features_list)
    
    scaler = StandardScaler()
    df[["Volume", "Mean Intensity", "Std Intensity"]] = scaler.fit_transform(df[["Volume", "Mean Intensity", "Std Intensity"]])

    df.to_csv(output_csv, index=False)
    print(f"Feature extraction complete. Data saved to {output_csv}")
else:
    print("No valid features extracted.")