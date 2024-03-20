import os
import re
from osgeo import gdal

"""
        1.The following code updates the config with each call to facilitate reading the relevant data parameters.
        2.You can modify the newdata_path and use your own dataset, the other data-related parameters will be updated automatically.
        3.All data and label size must be exactly the same.
"""
config = {
    "newdata_path": "./data/",  # Users can modify the root directory where the data is stored and use their own data.
    "data_path": ["origin_data/aspect_10.tif",
                  "origin_data/duanceng_5.tif",
                  "origin_data/elevation_5.tif",
                  "origin_data/gcyz_3.tif",
                  "origin_data/gougumd_5.tif",
                  "origin_data/qifudu_5.tif",
                  "origin_data/river_5.tif",
                  "origin_data/road_5.tif",
                  "origin_data/slope_5.tif",
                  "origin_data/slope_5.tif",
                  ],
    "label_path": "origin_data/label1.tif",
    # The label TIF file must include 0 (training set landslide), 1 (test set landslide),
    # 2 (training set non-landslide), 3 (test set non-landslide), 0+1=2+3, and (0+2):(1+3)=7:3 or 8:2.
    "feature": 13,
    "width": 3368,
    "height": 2626,  # height * width
    "batch_size": 8196,  # training batch_size
    "epochs": 300,
    "size": 21,
    "Cutting_window": 40,  # patch size： Cutting window * height
    "device": "cuda:1",  # "cuda"  "cpu"
    "lr": 0.0001,    
    'weight_decay': 0.5,    
    "normalize": False,
    "normalize_to_0_1": True
}

data_path = []
for tif_data in os.listdir(config["newdata_path"]):
    if tif_data.endswith('tif'):
        if re.match('label', tif_data):
            config["label_path"] = config["newdata_path"] + tif_data
            continue
        temp = config["newdata_path"] + tif_data
        data_path.append(temp)

config["data_path"] = data_path
config["feature"] = len(data_path)
tif = gdal.Open(config["data_path"][0])
config["width"], config["height"] = tif.RasterXSize, tif.RasterYSize
