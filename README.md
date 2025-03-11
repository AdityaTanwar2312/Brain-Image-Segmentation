# **MONAI U-Net Model for Medical Image Segmentation**

## **Overview**
This project focuses on **medical image segmentation** using **MONAI’s 3D Attention U-Net**. The model processes **NIfTI (.nii) format** medical images, segments anatomical structures, and extracts meaningful features for clustering and statistical analysis. 

The pipeline includes:
- **Data Preprocessing**
- **3D U-Net Model for Segmentation**
- **Feature Extraction & Clustering**
- **Statistical Analysis (ANOVA, Tukey's HSD, Kruskal-Wallis)**

## **1. Data Loading and Preprocessing**
The dataset consists of **NIfTI (.nii)** medical images, processed using the `nibabel` library. Preprocessing steps include:
- **Normalization**: Min-max normalization scales pixel values to [0,1].
- **Resizing**: Images resized to **(128,128,128)** using `scipy.ndimage.zoom`.
- **Data Augmentation**:
  - Random flips, rotations, zooms, and Gaussian noise to improve model generalization.

## **2. Model Architecture: MONAI 3D Attention U-Net**
The segmentation model is based on **MONAI’s 3D Attention U-Net**, which enhances U-Net with attention gates for better focus on anatomical structures.

### **Model Layers**
| Layer | Channels | Operation |
|--------|----------|-------------------------|
| Encoder Block 1 | 16 | Conv3D → ReLU → BN |
| Encoder Block 2 | 32 | Conv3D → ReLU → BN |
| Encoder Block 3 | 64 | Conv3D → ReLU → BN |
| Encoder Block 4 | 128 | Conv3D → ReLU → BN |
| Bottleneck | 256 | Conv3D → ReLU → BN |
| Decoder Block 1 | 128 | UpConv3D → Attention Gate |
| Decoder Block 2 | 64 | UpConv3D → Attention Gate |
| Decoder Block 3 | 32 | UpConv3D → Attention Gate |
| Decoder Block 4 | 16 | UpConv3D → Attention Gate |
| Output Layer | 3 | Final segmentation mask |

## **3. Loss Function and Optimizer**
- **Loss Function**: Cross-Entropy Loss for multi-class segmentation.
- **Optimizer**: AdamW with weight decay for regularization.

### **Training Configuration**
- **Learning Rate**: `1e-4`
- **Weight Decay**: `5e-4`
- **Scheduler**: `ReduceLROnPlateau` (adaptive learning rate reduction)

## **4. Model Inference and Post-Processing**
Once trained, the model segments test images using:
1. **Forward Pass** → Generates raw logits.
2. **Softmax Activation** → Converts logits into class probabilities.
3. **Post-Processing**:
   - **Binary Hole Filling**, **Morphological Closing**, and **Gaussian Smoothing** to refine segmentation.

## **5. Feature Extraction and Clustering Analysis**
After segmentation, key features are extracted:

### **Feature Extraction**
- **Volume Calculation**
- **Mean Intensity**
- **Standard Deviation of Intensity**

### **Clustering Using K-Means**
- **Elbow Method** → Determines the optimal number of clusters.
- **Silhouette Score** → Measures clustering quality.
- **PCA Scatter Plot** → Reduces feature space to 2D.

### **Clustered Data Sample**
| Mask Name | Volume | Mean Intensity | Std Intensity | Cluster |
|-------------|----------|-----------------|----------------|---------|
| segmented_0.nii.gz | 10532 | 0.72 | 0.21 | 0 |
| segmented_1.nii.gz | 9531 | 0.65 | 0.18 | 1 |
| segmented_2.nii.gz | 13421 | 0.81 | 0.27 | 2 |

## **6. Statistical Analysis**
To validate segmentation results, statistical tests were conducted:

### **ANOVA**
- Tests whether the means of volume, intensity, and PCA components differ significantly across clusters.

### **Kruskal-Wallis Test**
- A non-parametric alternative to ANOVA for non-normally distributed data.

### **Tukey’s HSD Test**
- Post-hoc test that identifies which clusters differ significantly.

## **Key Findings**
1. **Cluster 5** has the highest volume; **Cluster 2** the lowest.
2. **Cluster 4** has the highest mean intensity.
3. **Cluster 1 & Cluster 2** show similar intensity distributions.
4. **Cluster 4** is unique, with significantly different PCA scores.

## **7. Future Improvements**
- **Train with larger datasets** for better segmentation.
- **Label segmented cohorts** to enhance medical interpretation.
- **Use advanced feature extraction techniques** for improved clustering.

---

## **How to Use**
1. **Run Data Preprocessing**: `preprocess.py`
2. **Train the MONAI U-Net**: `train.py`
3. **Perform Inference & Save Masks**: `inference.py`
4. **Run Clustering Analysis**: `clustering.py`
5. **Perform Statistical Tests**: `stats.py`

---

## **Contributors**
- **Aditya Tanwar**
- **Project Goal:** Medical Image Segmentation & Analysis

---
