import os
import torch
scripts_path = os.getcwd()
data_path    = os.path.abspath(os.path.join(scripts_path, '..', 'data'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
smp_subset_test = False
smp_img_num = 5000
smp_img_random = False
smp_analysis_configs = [
    {
        "column_name": "total",
        "method": None,
        "intervals_num": None, 
        "other": None,
    },
    {
        "column_name": "subcategory",
        "method": "one-hot",
        "intervals_num": None, 
        "other": 0.01,
    },
    {
        "column_name": "category",
        "method": "one-hot",
        "intervals_num": None, 
        "other": 0.01,
    },
    {
        "column_name": "Postdate",
        "method": "quantile",
        "intervals_num": 20, 
        "other": None,
    },
    {
        "column_name": "Longitude",
        "method": "quantile",
        "intervals_num": 20, 
        "other": None,
    },
    {
        "column_name": "Geoaccuracy",
        "method": "one-hot",
        "intervals_num": None, 
        "other": 0.01,
    },
    {
        "column_name": "Latitude",
        "method": "quantile",
        "intervals_num": 20, 
        "other": None,
    },
    {
        "column_name": "photo_count",
        "method": "quantile",
        "intervals_num": 20, 
        "other": None,
    },
    {
        "column_name": "ispro",
        "method": "one-hot",
        "intervals_num": None, 
        "other": 0.01,
    },
    {
        "column_name": "timezone_offset",
        "method": "one-hot",
        "intervals_num": None, 
        "other": 0.01,
    },
    {
        "column_name": "timezone_id",
        "method": "one-hot",
        "intervals_num": None, 
        "other": 0.01,
    },
    {
        "column_name": "red_mean",
        "method": "quantile",
        "intervals_num": 20, 
        "other": None,
    },
    {
        "column_name": "green_mean",
        "method": "quantile",
        "intervals_num": 20, 
        "other": None,
    },
    {
        "column_name": "blue_mean",
        "method": "quantile",
        "intervals_num": 20, 
        "other": None,
    },
    {
        "column_name": "red_std",
        "method": "quantile",
        "intervals_num": 20, 
        "other": None,
    },
    {
        "column_name": "green_std",
        "method": "quantile",
        "intervals_num": 20, 
        "other": None,
    },
    {
        "column_name": "blue_std",
        "method": "quantile",
        "intervals_num": 20, 
        "other": None,
    },
    {
        "column_name": "hue_mean",
        "method": "quantile",
        "intervals_num": 20, 
        "other": None,
    },
    {
        "column_name": "saturation_mean",
        "method": "quantile",
        "intervals_num": 20, 
        "other": None,
    },
    {
        "column_name": "value_mean",
        "method": "quantile",
        "intervals_num": 20, 
        "other": None,
    },
    {
        "column_name": "hue_std",
        "method": "quantile",
        "intervals_num": 20, 
        "other": None,
    },
    {
        "column_name": "saturation_std",
        "method": "quantile",
        "intervals_num": 20, 
        "other": None,
    },
    {
        "column_name": "value_std",
        "method": "quantile",
        "intervals_num": 20, 
        "other": None,
    },
    {
        "column_name": "contrast",
        "method": "quantile",
        "intervals_num": 20, 
        "other": None,
    },
    {
        "column_name": "homogeneity",
        "method": "quantile",
        "intervals_num": 20, 
        "other": None,
    },
    {
        "column_name": "energy",
        "method": "quantile",
        "intervals_num": 20, 
        "other": None,
    },
    {
        "column_name": "entropy",
        "method": "quantile",
        "intervals_num": 20, 
        "other": None,
    },
    {
        "column_name": "correlation",
        "method": "quantile",
        "intervals_num": 20, 
        "other": None,
    },
    {
        "column_name": "ASM",
        "method": "quantile",
        "intervals_num": 20, 
        "other": None,
    },
    {
        "column_name": "user_description",
        "method": "texts-word-count",
        "intervals_num": 20, 
        "other": None,
    },
    {
        "column_name": "all_tags",
        "method": "texts-word-count",
        "intervals_num": 20, 
        "other": None,
    },
    {
        "column_name": "title",
        "method": "texts-word-count",
        "intervals_num": 20, 
        "other": None,
    },
]