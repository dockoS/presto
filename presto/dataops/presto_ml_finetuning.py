r"""
Chesapeake CVPR Data Processing Script
======================================

This script processes GeoTIFF files from the Chesapeake CVPR dataset to create
image chips for segmentation tasks.

Dataset Source:
---------------
Chesapeake CVPR data from LILA:
https://lila.science/datasets/chesapeakelandcover

For this experiment, we will use images from NY.

Notes:
------
1. Only copy *_lc.tif & *_naip-new.tif files that we will use for our
segmentation downstream task.
   Using s5cmd for this: https://github.com/peak/s5cmd
   - Train:
   s5cmd cp \
        --no-sign-request \
        --include "*_lc.tif" \
        --include "*_naip-new.tif" \
        "s3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/lcmcvpr2019/cvpr_chesapeake_landcover/ny_1m_2013_extended-debuffered-train_tiles/*" \
        data/cvpr/files/train/
   - Val:
   s5cmd cp \
        --no-sign-request \
        --include "*_lc.tif" \
        --include "*_naip-new.tif" \
        "s3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/lcmcvpr2019/cvpr_chesapeake_landcover/ny_1m_2013_extended-debuffered-val_tiles/*" \
        data/cvpr/files/val/

2. We will create chips of size `224 x 224` to feed them to the model, feel
free to experiment with other chip sizes as well.
   Run the script as follows:
   python preprocess_data.py <data_dir> <output_dir> <chip_size>

   Example:
   python preprocess_data.py data/cvpr/files data/cvpr/ny 224
"""  # noqa E501
import rioxarray
from pyproj import Transformer
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
#from torch.utils.data import DataLoader, TensorDataset
import utils
import presto 
import xgboost as xgb
from loguru import logger
# this is to silence the xarray deprecation warning.
# Our version of xarray is pinned, but we'll need to fix this
# when we upgrade
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import os
import sys
from pathlib import Path
from presto import Presto
import numpy as np
import rasterio as rio

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from pathlib import Path

def process_pixel(id,pos_i,pos_j, data_dir,sensor_prefix="_S2_*.tif"):
    # Get the list of files and sort them
    globs=list((data_dir).glob(f"{id}{sensor_prefix}"))
    if len(globs)>1:
        
        filenames = sorted(list((data_dir).glob(f"{id}{sensor_prefix}")), key=lambda x: int(x.stem.split('_')[-1]))
    else:
        filenames = list((data_dir).glob(f"{id}{sensor_prefix}"))

    
    # Rest of your code...def process_images(id,data_dir):
    arrays = []
    for filename in (filenames):
        tif_file = rioxarray.open_rasterio(filename)
        crs = tif_file.rio.crs
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

        # firstly, get the latitudes and longitudes
        x, y = tif_file.x[pos_j], tif_file.y[pos_i]
        lon, lat = transformer.transform(x, y) 
      
        # then, get the eo_data, mask and dynamic world
        data_for_pixel = tif_file.values[:, pos_i, pos_j]
        arrays.append(data_for_pixel)


    return np.stack(arrays,axis=0) ,(lat,lon)
      
def check_nan_or_neg_inf(array, name):
        has_nan = np.isnan(array).any()
        has_neg_inf = np.isneginf(array).any()
        
        print(f"{name} :")
        if has_nan:
            print("  - Contient des valeurs NaN")
        else:
            print("  - Ne contient pas de valeurs NaN")

        if has_neg_inf:
            print("  - Contient des valeurs -inf")
        else:
            print("  - Ne contient pas de valeurs -inf")

def process_pixel_for_all_sensors(id,pos_i,pos_j, data_dir):
   
    # Get the list of files and sort them


    s2_data,_=process_pixel(id,pos_i,pos_j, data_dir,sensor_prefix="_S2_*.tif")
    s1_data,_=process_pixel(id,pos_i,pos_j, data_dir,sensor_prefix="_S1_*.tif")
    dynamic_world_data,_=process_pixel(id,pos_i,pos_j, data_dir,sensor_prefix="_dynamic_world_*.tif")
    srtm_data,_=process_pixel(id,pos_i,pos_j, data_dir,sensor_prefix="_srtm*.tif")
    srtm_data=np.repeat(srtm_data,s2_data.shape[0],axis=0)
    label_data,latlon=process_pixel(id,pos_i,pos_j, data_dir,sensor_prefix="_label.tif")

    if s2_data.shape[0]!=dynamic_world_data.shape[0]:
        last_row = dynamic_world_data[-1, :]

        # Calculate the number of times to repeat the last row
        repeat_count = s2_data.shape[0]-dynamic_world_data.shape[0]

        # Create a new array by repeating the last row and concatenate with Y
        dw_repeated = np.repeat(last_row[np.newaxis, :], repeat_count, axis=0)
        dynamic_world_data = np.concatenate((dynamic_world_data, dw_repeated), axis=0)



    # Vérifier chaque tableau
 
    x, mask, dynamic_world = utils.construct_single_presto_input(
                    s2=torch.from_numpy(s2_data).float(), s2_bands=['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10','B11', 'B12'],
                    s1=torch.from_numpy(s1_data).float(), s1_bands=['VV', 'VH'],
                    dynamic_world=torch.from_numpy(dynamic_world_data).float(),
                    #srtm=torch.from_numpy(srtm_data[:,:2]).float(),
                    #
                    # srtm_bands= ["elevation", "slope"]
                )
    


    return x, mask, dynamic_world, torch.tensor(latlon),label_data[0,0]

def process_id(id,data_dir):
    ff_path=list((data_dir).glob(f"{id}_S2_*.tif"))

    tif_file = rioxarray.open_rasterio(ff_path[0])
    _,h,w=tif_file.shape
    arrays, masks, dynamic_worlds, latlons, labels= [], [], [], [], []
    if len(ff_path)!=4:
        return arrays, masks, dynamic_worlds, latlons, labels
    for i in range(h):
        for j in range(w):
            x, mask, dynamic_world,latlon,label_data=process_pixel_for_all_sensors(id,i,j,data_dir)
            arrays.append(x)
            masks.append(mask)
            dynamic_worlds.append(dynamic_world)
            latlons.append(latlon)
            labels.append(label_data)
      
    return arrays, masks, dynamic_worlds, latlons, labels

def process_all_ids(ids,data_dir):
    arrays, masks, dynamic_worlds, latlons, labels= [], [], [], [], []
    for i,id in enumerate(ids):  
        print("processing id : ",id," ",i+1,"/",len(ids))
        xs, _masks, _dynamic_worlds, _latlons,_labels=process_id(id,data_dir)
        #not append it is a list ,  want to append the elements

        arrays.extend(xs)
        masks.extend(_masks)
        dynamic_worlds.extend(_dynamic_worlds)
        latlons.extend(_latlons)
        labels.extend(_labels)

    
    return (torch.stack(arrays, axis=0),
            torch.stack(masks, axis=0),
            torch.stack(dynamic_worlds, axis=0),
            torch.stack(latlons, axis=0),
            torch.tensor(labels),
        )


    
    







       

    



def main(train_features_path,train_labels_path,test_features_path,test_labels_path,classes=[8, 9, 11, 17]):

    

    def preprocess(features_np,labels_np):

        # concatenate features

    
        # get labels
        #print(features_np)

    
        logger.error('labels: ',np.unique(labels_np))

        # filter out instances where label is zero
    
        label_mapping = {label: idx for idx, label in enumerate(classes)}

        

    # Appliquer le mappage aux labels


        # filter out instances where label is not in [8, 9, 11, 17]

        selected_indices = np.isin(labels_np,classes )

        features_np_filtered = features_np[selected_indices]
        labels_np_filtered = labels_np[selected_indices]
        labels_np_filtered = np.array([label_mapping[label] for label in labels_np_filtered])
    
    
        unique_labels, counts = np.unique(labels_np_filtered, return_counts=True)

        for label, count in zip(unique_labels, counts):
            logger.success(f"Label {label}: {count} occurrences")
        return features_np_filtered,labels_np_filtered

    train_features_np = np.load(train_features_path)
    train_labels_np = np.load(train_labels_path)
    test_features_np = np.load(test_features_path)
    test_labels_np=np.load(test_labels_path)
    features_train,labels_train=preprocess(train_features_np,train_labels_np)
    features_test,labels_test=preprocess(test_features_np,test_labels_np)


   

    # train the model on the training set
    model = RandomForestClassifier(class_weight="balanced", random_state=42)
    model.fit(features_train, labels_train)

    # make predictions on the test set
    predictions = model.predict(features_test)

    # compute accuracy
    accuracy = accuracy_score(labels_test, predictions)
    print(f"Accuracy rf: {accuracy}")
    # Entraîner le modèle XGBoost sur l'ensemble d'entraînement
    model_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model_xgb.fit(features_train, labels_train)

    # Faire des prédictions sur l'ensemble de test avec XGBoost
    predictions_xgb = model_xgb.predict(features_test)

    # Calculer la précision de XGBoost
    accuracy_xgb = accuracy_score(labels_test, predictions_xgb)
    print(f"XGBoost Accuracy: {accuracy_xgb}")
    return model,model_xgb
  

import matplotlib.pyplot as plt
def predict_file(id,data_dir,rf_model,xgb_model):
    ff_path=list((data_dir).glob(f"{id}_S2_*.tif"))[0]
    _,h,w=rioxarray.open_rasterio(ff_path).shape
    
    arrays, masks, dynamic_worlds, latlons, labels=process_id(id,data_dir)
    month = torch.tensor([6] * len(arrays)).long()

    dl = DataLoader(
        TensorDataset(
            torch.stack(arrays, axis=0).float(),  # x
            torch.stack(masks, axis=0),  # mask
            torch.stack(dynamic_worlds, axis=0).long(),  # dynamic world
             torch.stack(latlons, axis=0).float(),  # latlons
            month
        ),

    batch_size=16,
    shuffle=False,
    )
    features_list = []
    for (x, mask, dw, latlons, month) in tqdm(dl):
        
       

        dw=dw[:,:,0]
     
        #print('iccii',dw.shape)

        with torch.no_grad():
            encodings = (
                pretrained_model.encoder(
                    x, dynamic_world=dw, mask=mask, latlons=latlons, month=month
                )
                .cpu()
                .numpy()
            )
     
            features_list.append(encodings)
   

    features_np = np.concatenate(features_list)   

    
    label_predicted_rf=rf_model.predict(features_np)
    label_predicted_xgb=xgb_model.predict(features_np)
    plt.figure(figsize=(10,10))
    plt.subplot(1, 3, 1)
    plt.title("RF")
    plt.imshow(np.array(label_predicted_rf).reshape(h,w), cmap="viridis")
    plt.subplot(1, 3, 2)
    plt.title("XGB")
    plt.imshow(np.array(label_predicted_xgb).reshape(h,w), cmap="viridis")
    plt.subplot(1, 3, 3)
    plt.title("True")
    # label where value not in [8, 9, 11, 17] to -1
    labels_filtered = np.where(np.isin(labels, [8, 9, 11, 17]), labels, -1)
    plt.imshow(labels_filtered.reshape(h,w), cmap="viridis")
    # the title of all the plot
    plt.suptitle(id)
    plt.show()
    



if __name__ == "__main__":


    train_features_path = Path(sys.argv[1])
    train_labels_path = Path(sys.argv[2])
    test_features_path = Path(sys.argv[3])
    test_labels_path = Path(sys.argv[4])
    classes=sys.argv[5:]
    classes=[int(c) for c in classes]
 



    main(train_features_path,train_labels_path,test_features_path,test_labels_path,classes)




