import os
import sys
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import re
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow
import pytesseract
from src.utils import download_images
from src.constants import ALLOWED_UNITS
from src.sanity import run_sanity_check