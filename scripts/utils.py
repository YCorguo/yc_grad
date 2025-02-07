import os
import random
import gc
import torch
import PIL
import sys
from PIL import Image
import hashlib
import pandas as pd
import numpy as np
import json
from mmpretrain import get_model
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import tqdm
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
import scipy.stats as stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
from datetime import datetime, timedelta
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind
from collections import Counter
import pickle
import dill
import types
import warnings
from scipy.stats import ks_2samp
import concurrent.futures

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def leave_str_only_last_part_behind_comma(strr):
    if not strr:
        return None
    if ',' in strr:
        strr = strr.split(',')[-1]
    return strr.strip()

def process_date_time(time_str):
        # sample: "2008-02-11 17:18:31"
        dt_object = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        timestamp_seconds = int(dt_object.timestamp())
        return timestamp_seconds

# dataset
class ConstructDataset(Dataset):
    def __init__(self, img_data):
        self.dataframe = img_data
        img_classification_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform = img_classification_transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.loc[idx, 'image_path']
        image = Image.open(img_path).convert('RGB')
        label = self.dataframe.loc[idx, 'label']

        if self.transform:
            image = self.transform(image)

        return image, label
    
# classical statistic features
class ImageClassification():
    def __init__(self, img_data):
        self.imagenet_1k_labels_file = os.path.join(data_path, "imagenet-simple-labels.json")
        self.imagenet_1k_labels = json.load(open(self.imagenet_1k_labels_file))
        self.img_dataset = img_data
        data_loader = DataLoader(self.img_dataset, batch_size=32, shuffle=False)
        img_classification_model = get_model('efficientnet-b0_3rdparty_8xb32_in1k', pretrained=True).to(device)
        img_classification_model.eval()
        outputs = []
        with torch.no_grad():
            for image, labels in tqdm.tqdm(data_loader):
                image = image.to(device)
                results = img_classification_model(image)
                _, preds = torch.max(results, 1)
                
                outputs.append(preds.cpu())
        self.result = torch.cat(outputs, dim=0)
    
    def show_img1k_result(self):
        result = self.result.tolist()
        result = [self.imagenet_1k_labels[class_idx] for class_idx in result]
        print(result)

    def show_model(self):
        img_classification_model = get_model('efficientnet-b0_3rdparty_8xb32_in1k', pretrained=True).to(device)
        print(img_classification_model)

class RGBValue():
    def __init__(self, img_data):
        self.img_dataset = img_data
        data_loader = DataLoader(self.img_dataset, batch_size=32, shuffle=False)
        outputs = []
        for image, labels in tqdm.tqdm(data_loader):
            mean_per_channel = torch.mean(image, dim=(2, 3))
            std_per_channel = torch.std(image, dim=(2, 3))
            rgb_values = torch.cat((mean_per_channel, std_per_channel), dim=1)
            outputs.append(rgb_values)
        self.result = torch.cat(outputs, dim=0)

class HSVValue():
    def __init__(self, img_data):
        self.img_dataset = img_data
        data_loader = DataLoader(self.img_dataset, batch_size=32, shuffle=False)
        outputs = []
        for image, labels in tqdm.tqdm(data_loader):
            image = image.numpy()
            image = image * 255
            image = image.astype(np.uint8)
            image = np.transpose(image, (0, 2, 3, 1))
            for i in range(len(image)):
                image[i] = cv2.cvtColor(image[i], cv2.COLOR_RGB2HSV)
            image = np.transpose(image, (0, 3, 1, 2))
            image = torch.tensor(image, dtype=torch.float32)
            mean_per_channel = torch.mean(image, dim=(2, 3))
            std_per_channel = torch.std(image, dim=(2, 3))
            hsv_values = torch.cat((mean_per_channel, std_per_channel), dim=1)
            outputs.append(hsv_values)
        self.result = torch.cat(outputs, dim=0)

class Texture():
    def __init__(self, img_data):
        self.img_dataset = img_data
        data_loader = DataLoader(self.img_dataset, batch_size=32, shuffle=False)
        outputs = []
        for image, labels in tqdm.tqdm(data_loader):
            image = image.numpy()
            image = image * 255
            image = image.astype(np.uint8)
            image = np.transpose(image, (0, 2, 3, 1))
            for i in range(len(image)):
                temp_image = cv2.cvtColor(image[i], cv2.COLOR_RGB2GRAY)
                contrast, homogeneity, energy, entropy, correlation, ASM, dissimilarity = self.calculate_glcm_features(temp_image)
                textile_values = torch.tensor([contrast, homogeneity, energy, entropy, correlation, ASM, dissimilarity])
                outputs.append(textile_values)
        self.result = torch.stack(outputs)

    def calculate_glcm_features(self, image, distances=[1], angles=[0]):
        gray_image = img_as_ubyte(image)
        glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        ASM = graycoprops(glcm, 'ASM')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        entropy = -np.sum(glcm * np.log2(glcm + (glcm == 0)))
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        return contrast, homogeneity, energy, entropy, correlation, ASM, dissimilarity
    
# deep learning features
class ClassificationEmbedding():
    def __init__(self, img_data):
        # https://link.springer.com/chapter/10.1007/978-1-4842-6168-2_10
        self.imagenet_1k_labels_file = os.path.join(data_path, "imagenet-simple-labels.json")
        self.imagenet_1k_labels = json.load(open(self.imagenet_1k_labels_file))
        self.img_dataset = img_data
        data_loader = DataLoader(self.img_dataset, batch_size=32, shuffle=False)
        img_classification_model = get_model('efficientnet-b0_3rdparty_8xb32_in1k', pretrained=True).to(device)
        img_classification_model.eval()
        outputs = []
        with torch.no_grad():
            for image, labels in tqdm.tqdm(data_loader):
                image = image.to(device)
                results = img_classification_model(image)
                outputs.append(results.cpu())
        self.result = torch.cat(outputs, dim=0)
    
    def show_img1k_result(self):
        result = self.result.tolist()
        result = [self.imagenet_1k_labels[class_idx] for class_idx in result]
        print(result)

    def show_model(self):
        img_classification_model = get_model('efficientnet-b0_3rdparty_8xb32_in1k', pretrained=True).to(device)
        print(img_classification_model)

class SegmentationEmbedding():
    def __init__(self, img_data):
        # https://github.com/CASIA-IVA-Lab/FastSAM
        sys.path[-3] = os.path.abspath(os.path.join(scripts_path, "models", "FastSAM"))
        from fastsam import FastSAM
        import torch
        from PIL import Image
        sys.path[-3] = ''
        self.img_dataset = img_data
        data_loader = DataLoader(self.img_dataset, batch_size=32, shuffle=False)
        model_path = os.path.join(scripts_path, 'models', 'FastSAM', 'weights', 'FastSAM-s.pt')
        self.segmentation_model = FastSAM(model_path)
        retina_masks = True
        imgsz = 224
        conf = 0.4
        iou = 0.9
        outputs = []
        with torch.no_grad():
            for image, labels in tqdm.tqdm(data_loader):
                image = image.numpy()
                image = np.transpose(image, (0, 2, 3, 1))
                image = torch.tensor(image)
                image = image.to(device)
                results = self.segmentation_model(
                    image,
                    device=device,
                    retina_masks=retina_masks,
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou
                )
                # results = self.postprocess(results)
                print(results[0].boxes)
                print(results[0].masks)
                print(results[0].masks.shape)
                break
                outputs.append(results)
        self.result = torch.cat(outputs, dim=0)

    def show_model(self):
        print(self.segmentation_model)

    def postprocess(self, results):
        results = [result.masks for result in results]
        for i in range(len(results)):
            print(results[i].shape)
            results[i] = torch.tensor(results[i])
            while len(results[i]) < 5:
                results[i] = torch.cat([results[i], torch.zeros_like(results[i])], dim=0)
            if len(results[i]) > 5:
                results[i] = results[i][:5]
        return results
    
# add features
# def add_low_level_feature(data):
#     dataset = ConstructDataset(data)
#     rgb_value = RGBValue(dataset)
#     hsv_value = HSVValue(dataset)
#     texture = Texture(dataset)
#     data[['red_mean', 'green_mean', 'blue_mean', 'red_std', 'green_std', 'blue_std']] = \
#         pd.DataFrame(rgb_value.result.numpy(), 
#             columns=['red_mean', 'green_mean', 'blue_mean', 'red_std', 'green_std', 'blue_std'])
#     data[['hue_mean', 'saturation_mean', 'value_mean', 'hue_std', 'saturation_std', 'value_std']] = \
#         pd.DataFrame(hsv_value.result.numpy(), 
#             columns=['hue_mean', 'saturation_mean', 'value_mean', 'hue_std', 'saturation_std', 'value_std'])
#     data[['contrast', 'homogeneity', 'energy', 'entropy', 'correlation', 'ASM', 'dissimilarity']] = \
#         pd.DataFrame(texture.result.numpy(), 
#             columns=['contrast', 'homogeneity', 'energy', 'entropy', 'correlation', 'ASM', 'dissimilarity'])
#     return data
def add_low_level_feature(data):
    dataset = ConstructDataset(data)
    rgb_value = RGBValue(dataset)
    rgb_df = pd.DataFrame(rgb_value.result.numpy(), 
                          columns=['red_mean', 'green_mean', 'blue_mean', 
                                   'red_std', 'green_std', 'blue_std'])
    hsv_value = HSVValue(dataset)
    hsv_df = pd.DataFrame(hsv_value.result.numpy(), 
                          columns=['hue_mean', 'saturation_mean', 'value_mean', 
                                   'hue_std', 'saturation_std', 'value_std'])
    texture = Texture(dataset)
    texture_df = pd.DataFrame(texture.result.numpy(), 
                              columns=['contrast', 'homogeneity', 'energy', 
                                       'entropy', 'correlation', 'ASM', 'dissimilarity'])
    new_data = pd.concat([data, rgb_df, hsv_df, texture_df], axis=1)
    return new_data

def add_deep_learning_feature(data):
    dataset = ConstructDataset(data)
    classification_embedding = ClassificationEmbedding(dataset)
    data['classification_embedding'] = classification_embedding.result.numpy()
    classification = ImageClassification(dataset)
    data['classification'] = classification.result.numpy()
    segmentation_embedding = SegmentationEmbedding(dataset)
    data['segmentation_embedding'] = segmentation_embedding.result.numpy()
    return data

# save results
def analysis_feature(data, data_path, analysis_name):
    imgs_path = os.path.join(data_path, analysis_name)
    if analysis_name[:4] in ["smp_"]:
        analysis_name = analysis_name[4:]
    data_analysis = data.copy()
    data_analysis = data_analysis.dropna()
    data_analysis = data_analysis.reset_index(drop=True)

    significant_results = []
    
    # spearman correlation
    spearman_corr_matrix = data_analysis.corr(method='spearman')
    t_matrix = pd.DataFrame(index=spearman_corr_matrix.index, columns=spearman_corr_matrix.columns)
    p_matrix = pd.DataFrame(index=spearman_corr_matrix.index, columns=spearman_corr_matrix.columns)
    num_samples = len(data_analysis)
    for col1 in spearman_corr_matrix.columns:
        for col2 in spearman_corr_matrix.columns:
            if col1 != col2:
                r = spearman_corr_matrix.loc[col1, col2]
                if abs(r) == 1:
                    t = np.inf
                else:
                    t = r * np.sqrt((num_samples-2) / (1-r**2))
                p = 2 * (1 - stats.t.cdf(abs(t), num_samples-2))
                t_matrix.loc[col1, col2] = t
                p_matrix.loc[col1, col2] = p
                if p <= 0.05 and col1 == 'label':
                    significant_results.append({
                        'Analysis': 'Spearman Correlation',
                        'Variable1': col1,
                        'Variable2': col2,
                        'Correlation Coefficient': r,
                        'P-value': p
                    })
    if not os.path.exists(imgs_path):
        os.makedirs(imgs_path)
    p_matrix = p_matrix.apply(pd.to_numeric, errors='coerce')
    t_matrix = t_matrix.apply(pd.to_numeric, errors='coerce')
    # save matrices in one png
    plt.figure(figsize=(90, 30))
    plt.subplot(1, 3, 1)
    sns.heatmap(spearman_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('corr matrix', fontsize=16)
    plt.subplot(1, 3, 2)
    sns.heatmap(t_matrix, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('t value matrix', fontsize=16)
    plt.subplot(1, 3, 3)
    sns.heatmap(p_matrix, annot=True, cmap='YlGnBu', fmt='.4f')
    plt.title('p value matrix', fontsize=16)
    plt.savefig(os.path.join(imgs_path, f"spearman_correlation_analysis_{analysis_name}.png"))
    plt.close()
    spearman_corr_matrix_path = os.path.join(imgs_path, f"spearman_corr_matrix_{analysis_name}.csv")
    spearman_corr_matrix.to_csv(spearman_corr_matrix_path, index=True)
    p_matrix_path = os.path.join(imgs_path, f"spearman_p_matrix_{analysis_name}.csv")
    p_matrix.to_csv(p_matrix_path, index=True)
    
    # pearson correlation
    pearson_corr_matrix = data_analysis.corr(method='pearson')
    t_matrix = pd.DataFrame(index=pearson_corr_matrix.index, columns=pearson_corr_matrix.columns)
    p_matrix = pd.DataFrame(index=pearson_corr_matrix.index, columns=pearson_corr_matrix.columns)
    num_samples = len(data_analysis)
    for col1 in pearson_corr_matrix.columns:
        for col2 in pearson_corr_matrix.columns:
            if col1 != col2:
                r = pearson_corr_matrix.loc[col1, col2]
                if abs(r) == 1:
                    t = np.inf  # 或者根据需要处理这种特殊情况
                else:
                    t = r * np.sqrt((num_samples-2) / (1-r**2))
                p = 2 * (1 - stats.t.cdf(abs(t), num_samples-2))
                t_matrix.loc[col1, col2] = t
                p_matrix.loc[col1, col2] = p
                if p <= 0.05 and col1 == 'label':
                    significant_results.append({
                        'Analysis': 'Pearson Correlation',
                        'Variable1': col1,
                        'Variable2': col2,
                        'Correlation Coefficient': r,
                        'P-value': p
                    })
    p_matrix = p_matrix.apply(pd.to_numeric, errors='coerce')
    t_matrix = t_matrix.apply(pd.to_numeric, errors='coerce')
    plt.figure(figsize=(90, 30))
    plt.subplot(1, 3, 1)
    sns.heatmap(pearson_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('corr matrix', fontsize=16)
    plt.subplot(1, 3, 2)
    sns.heatmap(t_matrix, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('t value matrix', fontsize=16)
    plt.subplot(1, 3, 3)
    sns.heatmap(p_matrix, annot=True, cmap='YlGnBu', fmt='.4f')
    plt.title('p value matrix', fontsize=16)
    plt.savefig(os.path.join(imgs_path, f"pearson_correlation_analysis_{analysis_name}.png"))
    plt.close()
    pearson_corr_matrix_path = os.path.join(imgs_path, f"pearson_corr_matrix_{analysis_name}.csv")
    pearson_corr_matrix.to_csv(pearson_corr_matrix_path, index=True)
    p_matrix_path = os.path.join(imgs_path, f"pearson_p_matrix_{analysis_name}.csv")
    p_matrix.to_csv(p_matrix_path, index=True)
    
    # kendall correlation
    kendall_corr_matrix = data_analysis.corr(method='kendall')
    t_matrix = pd.DataFrame(index=kendall_corr_matrix.index, columns=kendall_corr_matrix.columns)
    p_matrix = pd.DataFrame(index=kendall_corr_matrix.index, columns=kendall_corr_matrix.columns)
    num_samples = len(data_analysis)
    for col1 in kendall_corr_matrix.columns:
        for col2 in kendall_corr_matrix.columns:
            if col1 != col2:
                r = kendall_corr_matrix.loc[col1, col2]
                if abs(r) == 1:
                    t = np.inf
                else:
                    t = r * np.sqrt((num_samples-2) / (1-r**2))
                p = 2 * (1 - stats.t.cdf(abs(t), num_samples-2))
                t_matrix.loc[col1, col2] = t
                p_matrix.loc[col1, col2] = p
                if p <= 0.05 and col1 == 'label':
                    significant_results.append({
                        'Analysis': 'Kendall Correlation',
                        'Variable1': col1,
                        'Variable2': col2,
                        'Correlation Coefficient': r,
                        'P-value': p
                    })
    p_matrix = p_matrix.apply(pd.to_numeric, errors='coerce')
    t_matrix = t_matrix.apply(pd.to_numeric, errors='coerce')
    plt.figure(figsize=(90, 30))
    plt.subplot(1, 3, 1)
    sns.heatmap(kendall_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('corr matrix', fontsize=16)
    plt.subplot(1, 3, 2)
    sns.heatmap(t_matrix, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('t value matrix', fontsize=16)
    plt.subplot(1, 3, 3)
    sns.heatmap(p_matrix, annot=True, cmap='YlGnBu', fmt='.4f')
    plt.title('p value matrix', fontsize=16)
    plt.savefig(os.path.join(imgs_path, f"kendall_correlation_analysis_{analysis_name}.png"))
    plt.close()
    kendall_corr_matrix_path = os.path.join(imgs_path, f"kendall_corr_matrix_{analysis_name}.csv")
    kendall_corr_matrix.to_csv(kendall_corr_matrix_path, index=True)
    p_matrix_path = os.path.join(imgs_path, f"kendall_p_matrix_{analysis_name}.csv")
    p_matrix.to_csv(p_matrix_path, index=True)

    # ---------------------------------------
    # Linear Regression
    # ---------------------------------------

    # Create copies of the features and target variable to avoid modifying the original data
    X_linear = data_analysis.drop('label', axis=1).copy()
    y_linear = np.exp(data_analysis['label'])  # Apply exponential transformation to 'label'

    # Add a constant term to the predictors
    X_linear = sm.add_constant(X_linear)

    # Split the data into training and testing sets
    X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(
        X_linear, y_linear, test_size=0.2, random_state=42)

    # Build the linear regression model
    linear_model = sm.OLS(y_train_linear, X_train_linear)
    linear_results = linear_model.fit()

    # Save linear regression results to a text file
    with open(os.path.join(imgs_path, f"linear_regression_analysis_{analysis_name}.txt"), "w") as f:
        f.write(linear_results.summary().as_text())
        coefficients = linear_results.params
        p_values = linear_results.pvalues
        t_values = linear_results.tvalues
        results_df = pd.DataFrame({
            'Coefficient': coefficients,
            'P-value': p_values,
            'T-value': t_values
        })
        f.write("\nCoefficient, P-value, and T-value:\n")
        f.write(results_df.to_string())
        y_pred_linear = linear_results.predict(X_test_linear)
        mse_linear = mean_squared_error(y_test_linear, y_pred_linear)
        r2_linear = r2_score(y_test_linear, y_pred_linear)
        f.write(f"\nMean Squared Error (MSE): {mse_linear}")
        f.write(f"\nR-squared: {r2_linear}")
        # Construct the regression equation
        equation = "y = "
        for i, column in enumerate(X_linear.columns):
            coef = coefficients.iloc[i]
            sign = " + " if coef >= 0 else " - "
            term = f"{abs(coef):.4f} * {column}" if column != 'const' else f"{coef:.4f}"
            equation += sign + term if i > 0 else term
        f.write(f"\nLinear Regression Equation:\n")
        f.write(equation)

    for index, row in results_df.iterrows():
        if index != 'const' and row['P-value'] <= 0.05:
            significant_results.append({
                'Analysis': 'Linear Regression',
                'Variable': index,
                'Coefficient': row['Coefficient'],
                'P-value': row['P-value'],
                'T-value': row['T-value']
            })

    # Plot heatmaps for coefficients and p-values
    plt.figure(figsize=(30, 15))

    # Heatmap for Coefficients
    plt.subplot(1, 2, 1)
    sns.heatmap(results_df[['Coefficient']].sort_values(by='Coefficient', ascending=False),
                annot=True, cmap='coolwarm', center=0)
    plt.title('Coefficient')

    # Heatmap for P-values
    plt.subplot(1, 2, 2)
    sns.heatmap(results_df[['P-value']].sort_values(by='P-value'),
                annot=True, cmap='YlOrRd')
    plt.title('P-value')

    plt.tight_layout()
    plt.savefig(os.path.join(imgs_path, f"linear_regression_analysis_plots_{analysis_name}.png"))
    plt.close()

    # ---------------------------------------
    # Poisson Regression
    # ---------------------------------------

    # Create copies of the features and target variable to avoid modifying the original data
    X_poisson = data_analysis.drop('label', axis=1).copy()
    y_poisson = np.exp(data_analysis['label'])  # Apply exponential transformation to 'label'

    # Convert target variable to non-negative integers (required for Poisson regression)
    y_poisson = y_poisson.round().astype(int)

    # Check for negative values in the target variable
    if (y_poisson < 0).any():
        raise ValueError("The target variable contains negative values, which are invalid for Poisson regression.")

    # Add a constant term to the predictors
    X_poisson = sm.add_constant(X_poisson)

    # Split the data into training and testing sets
    X_train_poisson, X_test_poisson, y_train_poisson, y_test_poisson = train_test_split(
        X_poisson, y_poisson, test_size=0.2, random_state=42)

    # Build the Poisson regression model using Generalized Linear Model (GLM)
    poisson_model = sm.GLM(y_train_poisson, X_train_poisson, family=sm.families.Poisson())
    poisson_results = poisson_model.fit()

    # Save Poisson regression results to a text file
    with open(os.path.join(imgs_path, f"poisson_regression_analysis_{analysis_name}.txt"), "w") as f:
        f.write(poisson_results.summary().as_text())
        coefficients = poisson_results.params
        p_values = poisson_results.pvalues
        t_values = poisson_results.tvalues
        results_df = pd.DataFrame({
            'Coefficient': coefficients,
            'P-value': p_values,
            'T-value': t_values
        })
        f.write("\nCoefficient, P-value, and T-value:\n")
        f.write(results_df.to_string())
        y_pred_poisson = poisson_results.predict(X_test_poisson)
        mse_poisson = mean_squared_error(y_test_poisson, y_pred_poisson)
        r2_poisson = r2_score(y_test_poisson, y_pred_poisson)
        f.write(f"\nMean Squared Error (MSE): {mse_poisson}")
        f.write(f"\nR-squared: {r2_poisson}")
        # Construct the regression equation
        equation = "y = "
        for i, column in enumerate(X_poisson.columns):
            coef = coefficients.iloc[i]
            sign = " + " if coef >= 0 else " - "
            term = f"{abs(coef):.4f} * {column}" if column != 'const' else f"{coef:.4f}"
            equation += sign + term if i > 0 else term
        f.write(f"\nPoisson Regression Equation:\n")
        f.write(equation)

    for index, row in results_df.iterrows():
        if index != 'const' and row['P-value'] <= 0.05:
            significant_results.append({
                'Analysis': 'Poisson Regression',
                'Variable': index,
                'Coefficient': row['Coefficient'],
                'P-value': row['P-value'],
                'T-value': row['T-value']
            })

    # Plot heatmaps for coefficients and p-values
    plt.figure(figsize=(30, 15))

    # Heatmap for Coefficients
    plt.subplot(1, 2, 1)
    sns.heatmap(results_df[['Coefficient']].sort_values(by='Coefficient', ascending=False),
                annot=True, cmap='coolwarm', center=0)
    plt.title('Coefficient')

    # Heatmap for P-values
    plt.subplot(1, 2, 2)
    sns.heatmap(results_df[['P-value']].sort_values(by='P-value'),
                annot=True, cmap='YlOrRd')
    plt.title('P-value')

    plt.tight_layout()
    plt.savefig(os.path.join(imgs_path, f"poisson_regression_analysis_plots_{analysis_name}.png"))
    plt.close()

    # Chi-square Test and t-test
    chi2_results = {}
    t_results = {}
    X = data_analysis.drop('label', axis=1)
    y = data_analysis['label']
    for column in X.columns:
        contingency_table = pd.crosstab(y, X[column])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        if p <= 0.05:
            significant_results.append({
                'Analysis': 'Chi-square Test',
                'Variable': column,
                'Chi2': chi2,
                'P-value': p,
                'Degrees of Freedom': dof
            })
        chi2_results[column] = {'chi2': chi2, 'p': p, 'dof': dof}
        t_stat, p_value = ttest_ind(y, X[column])
        t_results[column] = {'t_stat': t_stat, 'p_value': p_value}
        if p_value <= 0.05:
            significant_results.append({
                'Analysis': 't-test',
                'Variable': column,
                't_stat': t_stat,
                'P-value': p_value
            })
    # save results
    with open(os.path.join(imgs_path, f"chi2_test_results_{analysis_name}.txt"), "w") as f:
        f.write(str(chi2_results))
    with open(os.path.join(imgs_path, f"t_test_results_{analysis_name}.txt"), "w") as f:
        f.write(str(t_results))

    return significant_results

def get_smp_data(scripts_path, data_path, configs):
    # Get data
    smp_data_dir = os.path.abspath(os.path.join(scripts_path, "..", "data", "smp_2019"))

    # Load image paths
    with open(os.path.join(smp_data_dir, "train_img_filepath.txt")) as f:
        smp_image_path_files = f.readlines()
        smp_image_path_files = [os.path.join(smp_data_dir, path.replace("train/", "").strip()) 
                                for path in smp_image_path_files]

    # Load labels
    with open(os.path.join(smp_data_dir, "train_label.txt")) as f:
        smp_labels = [float(label.strip()) for label in f]

    # Load categories
    with open(os.path.join(data_path, "smp_2019", "train_category.json")) as f:
        smp_category_dict = json.load(f)
        smp_category = [item['Category'] for item in smp_category_dict]
        smp_subcategory = [item['Subcategory'] for item in smp_category_dict]
        smp_concept = [item['Concept'] for item in smp_category_dict]

    # Combine categories and subcategories
    smp_category_n_subcategory = [f"{cat}:{subcat}" for cat, subcat in zip(smp_category, smp_subcategory)]

    # Count occurrences
    smp_subcategory_2_counter = Counter(smp_category_n_subcategory)
    # for i, (category, count) in enumerate(smp_subcategory_2_counter.most_common(30), 1):
    #     print(f"{i}. {category}: {count}")

    # Load additional information
    with open(os.path.join(smp_data_dir, "train_additional_information.json")) as f:
        smp_additional_info = json.load(f)
        smp_Mediastatus = [info['Mediastatus'] for info in smp_additional_info]
        smp_Pathalias = [info['Pathalias'] for info in smp_additional_info]
        smp_Ispublic = [info['Ispublic'] for info in smp_additional_info]
        smp_Pid = [info['Pid'] for info in smp_additional_info]
        smp_Uid = [info['Uid'] for info in smp_additional_info]

    # Load temporal and spatial information
    with open(os.path.join(smp_data_dir, "train_temporalspatial_information.json")) as f:
        smp_temporalspatial_information = json.load(f)
        smp_Postdate = [info['Postdate'] for info in smp_temporalspatial_information]
        smp_Longitude = [info['Longitude'] for info in smp_temporalspatial_information]
        smp_Geoaccuracy = [info['Geoaccuracy'] for info in smp_temporalspatial_information]
        smp_Latitude = [info['Latitude'] for info in smp_temporalspatial_information]

    # Load user data
    with open(os.path.join(smp_data_dir, "train_user_data.json")) as f:
        smp_user_data = json.load(f)
        n = len(smp_user_data['Pid'])
        # 'Uid', 'Pid', 'photo_firstdate', 'photo_count', 'ispro', 'timezone_offset', 'photo_firstdatetaken', 'timezone_id', 'user_description', 'location_description'
        smp_photo_firstdate = [smp_user_data['photo_firstdate'][str(i)] for i in range(n)]
        smp_photo_count = [smp_user_data['photo_count'][str(i)] for i in range(n)]
        smp_ispro = [smp_user_data['ispro'][str(i)] for i in range(n)]
        smp_timezone_offset = [smp_user_data['timezone_offset'][str(i)] for i in range(n)]
        smp_photo_firstdatetaken = [smp_user_data['photo_firstdatetaken'][str(i)] for i in range(n)]
        smp_timezone_id = [smp_user_data['timezone_id'][str(i)] for i in range(n)]
        smp_user_description = [smp_user_data['user_description'][str(i)] for i in range(n)]
        smp_location_description = [smp_user_data['location_description'][str(i)] for i in range(n)]

    # "Alltags": "rock punk transgender tranny electronicmusic electro glam electronica luisdrayton fusionrecords thefusionnetwork lmwcphotography", "Pid": "775", "Uid": "59@N75", "Mediatype": "photo", "Title": "Luis Drayton - Edinburgh shoot #6"
    with open(os.path.join(smp_data_dir, "train_text.json")) as f:
        smp_text_data = json.load(f)
        smp_all_tags = [item["Alltags"] for item in smp_text_data]
        smp_media_type = [item["Mediatype"] for item in smp_text_data]
        smp_title = [item["Title"] for item in smp_text_data]

    # Create DataFrame
    smp_data = pd.DataFrame({
        "image_path": smp_image_path_files,
        "label": smp_labels,
        "category": smp_category,
        "subcategory": smp_subcategory,
        "Mediastatus": smp_Mediastatus,
        "Pathalias": smp_Pathalias,
        "Ispublic": smp_Ispublic,
        "Pid": smp_Pid,
        "Uid": smp_Uid,
        "Postdate": smp_Postdate,
        "Longitude": smp_Longitude,
        "Geoaccuracy": smp_Geoaccuracy,
        "Latitude": smp_Latitude,
        "photo_firstdate": smp_photo_firstdate,
        "photo_count": smp_photo_count,
        "ispro": smp_ispro,
        "timezone_offset": smp_timezone_offset,
        "photo_firstdatetaken": smp_photo_firstdatetaken,
        "timezone_id": smp_timezone_id,
        "user_description": smp_user_description,
        "location_description": smp_location_description,
        "concept": smp_concept,
        "all_tags": smp_all_tags,
        "media_type": smp_media_type,
        "title": smp_title,
    })

    smp_data["location_description"] = smp_data["location_description"].apply(leave_str_only_last_part_behind_comma)

    # Downsample if needed
    if configs.smp_subset_test:
        if not configs.smp_img_random:
            random.seed(42)
        random_images_indices = random.sample(range(len(smp_image_path_files)), 
                                            min(len(smp_image_path_files), configs.smp_img_num))
        random_images_indices = [idx for idx in random_images_indices 
                                if os.path.getsize(smp_image_path_files[idx]) > 0]
        smp_data = smp_data.iloc[random_images_indices].reset_index(drop=True)

    # print(smp_data.drop(columns=['image_path']).head(1))
    print(smp_data.columns)

    return smp_data

def data_normalization(data, column_name):
    # Normalization
    data_analysis = data.copy()

    # Keep only 'photo' in media_type
    data_analysis = data_analysis[data_analysis['media_type'] == 'photo']

    # Drop irrelevant columns
    for column in ['image_path', 'Pid', 'Uid', 'category', 'subcategory', 'Mediastatus', 'Pathalias', 'user_description', 'location_description', 'concept', 'all_tags', 'media_type', 'title', 'timezone_id', 'Longitude', 'photo_firstdate', 'photo_firstdatetaken']:
        if column in data_analysis.columns:
            data_analysis = data_analysis.drop(columns=[column])

    initial_line_count = len(data_analysis)

    # Count missing values for each column
    # for column in data_analysis.columns:
    #     print(f"Missing values in {column} column: {data_analysis[column].isna().sum()}")

    # Drop rows with NaN values
    data_analysis = data_analysis.dropna()
    data_analysis = data_analysis.reset_index(drop=True)

    # Convert 'timezone_offset' to minutes
    def time_str_to_minutes(time_str):
        time_obj = datetime.strptime(time_str, "%z")
        offset = time_obj.utcoffset()
        total_minutes = int(offset.total_seconds() / 60)
        return total_minutes

    data_analysis['timezone_offset'] = data_analysis['timezone_offset'].apply(time_str_to_minutes)
    # print("Line count after converting timezone_offset to minutes:", len(data_analysis))

    # Drop rows with NaN in 'timezone_offset' column
    data_analysis = data_analysis.dropna(subset=['timezone_offset'])

    # Print column values
    # for column in data_analysis.columns:
    #     print(column, data_analysis[column])

    # print('\n' * 3)

    # Drop rows with NaN values again
    data_analysis = data_analysis.dropna()
    # print(data_analysis.head())

    # Remove outliers based on the IQR method
    def remove_outliers_all_columns(df, exclude_columns=['label'], factor=1.5):
        df = df.dropna()
        df = df.reset_index(drop=True)
        df_cleaned = df.copy()
        columns_to_process = [col for col in df.columns if col not in exclude_columns]
        for column in columns_to_process:
            if np.issubdtype(df[column].dtype, np.number):
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                df_cleaned.loc[(df_cleaned[column] < lower_bound) | (df_cleaned[column] > upper_bound), column] = np.nan
        df_cleaned = df_cleaned.dropna()
        return df_cleaned

    # Normalize data using different scaling methods
    def sklearn_normalize(df, method='minmax', exclude_columns=['label']):
        df_normalized = df.copy()
        columns_to_normalize = [col for col in df.columns if col not in exclude_columns]
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'zscore':
            scaler = StandardScaler()
        elif method == 'maxabs':
            scaler = MaxAbsScaler()
        else:
            raise ValueError("Unsupported method. Choose 'minmax', 'zscore', or 'maxabs'.")
        
        df_normalized[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
        return df_normalized

    # Remove outliers and normalize data
    data_analysis = remove_outliers_all_columns(data_analysis)
    data_analysis = sklearn_normalize(data_analysis, method='zscore')

    # Check if columns can be converted to float
    for column in data_analysis.columns:
        if not np.issubdtype(data_analysis[column].dtype, np.number):
            data_analysis[column] = pd.to_numeric(data_analysis[column], errors='coerce')

    # print("[" + column_name + "] | line count:", initial_line_count, "->", len(data_analysis))

    return data_analysis

def show_subset_of_data(data, column_name, filter):
    subset_data = data[data[column_name] == filter]
    print(column_name, "show:")
    print(subset_data.head())
    print(f"\nTravel data volume: {len(subset_data)}")

    random_indices = random.sample(range(len(subset_data)), 20)
    fig, axes = plt.subplots(2, 10, figsize=(15, 5))
    for i, idx in enumerate(random_indices):
        img_path = subset_data.iloc[idx]['image_path']
        
        img = Image.open(img_path)
        
        row = i // 10
        col = i % 10
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        axes[row, col].set_title(f"travel images {i+1}")

    plt.tight_layout()
    plt.show()

def split_data(input, column_name, method="one-hot", intervals_num=None, other=0.01):
    data = input.copy()
    if method == "one-hot":
        # Split the data by each unique value in the specified column
        unique_values = data[column_name].unique()
        split_dataframes = [data[data[column_name] == value].reset_index(drop=True) for value in unique_values]
        if other > 1:
            raise ValueError("one_hot_thr must be less than or equal to 1.")
        split_dataframes = [split_dataframe for split_dataframe in split_dataframes if len(split_dataframe) >= len(data)*other]
    elif method == "quantile":
        # 1. Sort the DataFrame by the specified column
        # print(len(data))
        data[column_name] = pd.to_numeric(data[column_name], errors='coerce')
        # print(len(data))
        sorted_data = data.sort_values(by=column_name).reset_index(drop=True)
        # print(len(sorted_data))
        # # 2. drop data if more than 30% is the same
        # value_counts = sorted_data[column_name].value_counts(normalize=True)
        # filtered_values = value_counts[value_counts <= -1].index
        # if filtered_values.empty:
        #     filtered_values = value_counts.index[:1]
        # sorted_data = sorted_data[sorted_data[column_name].isin(filtered_values)].reset_index(drop=True)
        # 3. Calculate the 0.05 and 0.95 quantiles
        lower_quantile = sorted_data[column_name].quantile(0.05)
        upper_quantile = sorted_data[column_name].quantile(0.95)
        # 4. Filter the data to include only rows between the quantiles
        filtered_data = sorted_data[(sorted_data[column_name] >= lower_quantile) & (sorted_data[column_name] <= upper_quantile)]
        # 5. Split the filtered data into the specified number of intervals
        if intervals_num is None or intervals_num <= 0:
            raise ValueError("intervals_num must be a positive integer for quantile method.")
        split_dataframes = np.array_split(filtered_data, intervals_num)
        # 6. Return a list of DataFrames, resetting the index for each
        split_dataframes = [df.reset_index(drop=True) for df in split_dataframes]
    elif method == "texts-word-count":
        # 1. substitute None in the specified column with empty string
        data[column_name] = data[column_name].fillna("")
        data[column_name] = data[column_name].apply(lambda x: len(x.split()))
        # 2. Sort the DataFrame by the specified column
        sorted_data = data.sort_values(by=column_name).reset_index(drop=True)
        # 3. Calculate the 0.05 and 0.95 quantiles
        lower_quantile = sorted_data[column_name].quantile(0.05)
        upper_quantile = sorted_data[column_name].quantile(0.95)
        # 4. Filter the data to include only rows between the quantiles
        filtered_data = sorted_data[(sorted_data[column_name] >= lower_quantile) & (sorted_data[column_name] <= upper_quantile)]
        # 5. Split the filtered data into the specified number of intervals
        if intervals_num is None or intervals_num <= 0:
            raise ValueError("intervals_num must be a positive integer for quantile method.")
        split_dataframes = np.array_split(filtered_data, intervals_num)
        # 6. Return a list of DataFrames, resetting the index for each
        split_dataframes = [df.reset_index(drop=True) for df in split_dataframes]
    else:
        raise ValueError(f"Invalid method: {method}. Supported methods are 'one-hot', 'quantile', 'texts-word-count'.")
    first_rows = [df.iloc[0] for df in split_dataframes]
    first_rows = pd.concat(first_rows, axis=1).T.reset_index(drop=True)
    return first_rows, split_dataframes

def is_serializable(obj):
    """尝试序列化对象，若失败则返回False"""
    try:
        dill.dumps(obj)
    except (TypeError, dill.PicklingError):
        return False
    return True

def load_workspace_variables(filename):
    with open(filename, "rb") as file:
        workspace_variables = dill.load(file)
    globals().update(workspace_variables)

def analysis_features_using_configs(smp_data, config, data_path):
    valid_records = []
    if config["column_name"] == "total":
        smp_data_analysis = data_normalization(data=smp_data, column_name=config["column_name"])
        valid_records += analysis_feature(smp_data_analysis, data_path, "smp_total")
    else:
        smp_splited_data_first_rows, smp_splited_data = split_data(
                input = smp_data,
                ** config
            )
        smp_splited_normalized_data = [(data[config["column_name"]], \
            data_normalization(data, config["column_name"] + "|" + str(data[config["column_name"]].iloc[0]))) \
                for data in smp_splited_data]
        # KS test
        valid_records += ks_test_analysis(smp_splited_normalized_data, data_path, analysis_column='label', split_column_name=config["column_name"])
        for column, data in tqdm.tqdm(smp_splited_normalized_data):
            save_name = "smp_" + config["column_name"] + "_" + str(column.iloc[0])
            valid_records += analysis_feature(data, data_path, save_name)
    return valid_records

def smp_analysis_texts(smp_data, most_common_num=20):
    user_cter = Counter()
    for user in smp_data["user_description"]:
        if user is not None:
            user_cter.update(user.lower().split())
    user_cter["and"] = user_cter["I"] = user_cter["the"] = user_cter["href"] = user_cter["of"] = \
        user_cter["<a"] = user_cter["to"] = user_cter["in"] = user_cter["rel"] = user_cter["nofollow"] = \
        user_cter["class"] = user_cter["on"] = user_cter["for"] = user_cter["is"] = user_cter["-"] = \
        user_cter["m"] = user_cter["><img"] = user_cter["alt="] = user_cter["with"] = user_cter["/></a>"] = \
        user_cter["at"] = user_cter["are"] = user_cter["or"] = user_cter["as"] = user_cter["width="] = \
        user_cter["that"] = user_cter["height="] = user_cter["by"] = user_cter["s"] = user_cter["be"] = \
        user_cter["from"] = user_cter["title="] = user_cter["an"] = user_cter["&amp;"] = user_cter["am"] = \
        user_cter["if"] = user_cter["this"] = user_cter["de"] = user_cter["do"] = user_cter["will"] = \
        user_cter["src="] = user_cter["href="] = user_cter["a"] = user_cter["rel="] = user_cter["class="] = 0
    print("user_cter:", user_cter.most_common(most_common_num))
    tags_cter = Counter()
    for tag in smp_data["all_tags"]:
        if tag is not None:
            tags_cter.update(tag.lower().split())
    tags_cter['??'] = tags_cter['2015'] = tags_cter['????'] = 0
    print("tags_cter:", tags_cter.most_common(most_common_num))
    tit_cter = Counter()
    for title in smp_data["title"]:
        if title is not None:
            tit_cter.update(title.lower().split())
    tit_cter['-'] = tit_cter['the'] = tit_cter['2015'] = tit_cter['of'] = tit_cter['and'] = tit_cter['in'] = \
        tit_cter['&'] = tit_cter['2016'] = tit_cter['at'] = tit_cter['to'] = tit_cter['on'] = tit_cter['de'] = \
        tit_cter['|'] = tit_cter['with'] = tit_cter['?'] = tit_cter['????'] = tit_cter['1'] = tit_cter['a'] = \
        tit_cter['@'] = tit_cter['2'] = tit_cter['from'] = tit_cter['is'] = tit_cter['la'] = tit_cter['/'] = 0
    print("tit_cter:", tit_cter.most_common(most_common_num))
    return user_cter.most_common(most_common_num), tags_cter.most_common(most_common_num), \
        tit_cter.most_common(most_common_num)

def smp_analysis_texts_existence(smp_data, data_path, user_cter, tags_cter, tit_cter):
    for user_text in tqdm.tqdm(user_cter):
        text = user_text[0]
        smp_user_text_data = pd.concat([smp_data.copy(), smp_data["user_description"].apply(
            lambda x: text in x if x else False
        ).to_frame("user_description_" + text)], axis=1)
        config = {
            "column_name": "user_description_" + text,
            "method": "one-hot",
            "intervals_num": None,
            "other": 0.01,
        }
        analysis_features_using_configs(smp_user_text_data, config, data_path)
    for tag_text in tqdm.tqdm(tags_cter):
        text = tag_text[0]
        smp_tag_text_data = pd.concat([smp_data.copy(), smp_data["all_tags"].apply(
            lambda x: text in x if x else False
        ).to_frame("all_tags_" + text)], axis=1)
        config = {
            "column_name": "all_tags_" + text,
            "method": "one-hot",
            "intervals_num": None,
            "other": 0.01,
        }
        analysis_features_using_configs(smp_tag_text_data, config, data_path)
    for user_text in tqdm.tqdm(tit_cter):
        text = user_text[0]
        smp_title_text_data = pd.concat([smp_data.copy(), smp_data["title"].apply(
            lambda x: text in x if x else False
        ).to_frame("title_" + text)], axis=1)
        config = {
            "column_name": "title_" + text,
            "method": "one-hot",
            "intervals_num": None,
            "other": 0.01,
        }
        analysis_features_using_configs(smp_title_text_data, config, data_path)

def ks_test_analysis(splitted_data, data_path, analysis_column, split_column_name):
    columns = [data[0] for data in splitted_data]
    df_list = [data[1] for data in splitted_data]
    # 初始化 KS-stat 和 p-value 矩阵
    num_dfs = len(df_list)
    ks_stat_matrix = np.zeros((num_dfs, num_dfs))
    p_value_matrix = np.zeros((num_dfs, num_dfs))

    significant_results = []

    # 计算两两 DataFrame 的 KS-stat 和 p-value
    for i in range(num_dfs):
        for j in range(i, num_dfs):
            if i == j:
                # 当比较自身时，随机将数据分成两部分
                label_column = df_list[i][analysis_column]
                split_index = random.sample(range(len(label_column)), len(label_column) // 2)
                part1 = label_column.iloc[split_index]
                part2 = label_column.drop(label_column.index[split_index])
                ks_stat, p_value = ks_2samp(part1, part2)
            else:
                # 比较不同 DataFrame
                a_label = df_list[i][analysis_column]
                b_label = df_list[j][analysis_column]
                ks_stat, p_value = ks_2samp(a_label, b_label)

            # 填充矩阵
            ks_stat_matrix[i, j] = ks_stat
            ks_stat_matrix[j, i] = ks_stat
            p_value_matrix[i, j] = p_value
            p_value_matrix[j, i] = p_value

            if i <= j and p_value <= 0.05:
                # 构建可读的组名
                group_i_name = f"{split_column_name}_{columns[i].iloc[0]}" if hasattr(columns[i], 'iloc') else str(columns[i])
                group_j_name = f"{split_column_name}_{columns[j].iloc[0]}" if hasattr(columns[j], 'iloc') else str(columns[j])
                
                significant_results.append({
                    "Group1": group_i_name,
                    "Group2": group_j_name,
                    "KS-stat": ks_stat,
                    "P-value": p_value
                })

    # 将 KS-stat 和 p-value 矩阵转化为 DataFrame
    ks_stat_df = pd.DataFrame(ks_stat_matrix, columns=[split_column_name + "_" + \
        str(column.iloc[0]) for column in columns], index=[split_column_name + "_" + \
        str(column.iloc[0]) for column in columns])
    p_value_df = pd.DataFrame(p_value_matrix, columns=[split_column_name + "_" + \
        str(column.iloc[0]) for column in columns], index=[split_column_name + "_" + \
        str(column.iloc[0]) for column in columns])

    # 画 KS-stat 和 p-value 的色阶图
    plt.figure(figsize=(24, 10))

    # KS-stat 矩阵的色阶图
    plt.subplot(1, 2, 1)
    sns.heatmap(ks_stat_df, annot=True, cmap='Blues', square=True, cbar=True)
    plt.title('KS-stat Matrix')

    # p-value 矩阵的色阶图
    plt.subplot(1, 2, 2)
    sns.heatmap(p_value_df, annot=True, cmap='Reds', square=True, cbar=True)
    plt.title('p-value Matrix')

    save_path = os.path.join(data_path, "smp_" + split_column_name + "_ks_test")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.tight_layout()
    fig_filename = os.path.join(save_path, 'ks_pvalue_heatmaps.png')
    plt.savefig(fig_filename)
    ks_stat_df_filename = os.path.join(save_path, 'ks_stat_matrix.csv')
    p_value_df_filename = os.path.join(save_path, 'p_value_matrix.csv')
    ks_stat_df.to_csv(ks_stat_df_filename)
    p_value_df.to_csv(p_value_df_filename)

    return significant_results

def save_workspace_variables(filename, variables):
    # 过滤掉模块、函数、方法和不可序列化的对象
    workspace_variables = {key: value for key, value in variables.items() 
                           if not (key.startswith("__") 
                                   or isinstance(value, types.ModuleType) 
                                   or isinstance(value, types.FunctionType) 
                                   or isinstance(value, types.MethodType))
                           and is_serializable(value)}
    with open(filename, "wb") as file:
        dill.dump(workspace_variables, file)