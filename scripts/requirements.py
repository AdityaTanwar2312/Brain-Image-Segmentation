import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torchio as tio
from scipy.ndimage import zoom
from monai.transforms import RandFlip, RandRotate, RandZoom, RandGaussianNoise
from monai.networks.nets import AttentionUnet
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, binary_closing, label, gaussian_filter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score