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


    
    







       

    



def save(ids,data_dir,output_dir,train_or_test="train"):
    os.makedirs(output_dir, exist_ok=True)
    # write the training ids to train_file.txt
    with open(os.path.join(output_dir,train_or_test+'.txt'), 'w') as f:
        for id in ids:
            f.write(f'{id}\n')


    train_data=process_all_ids(ids,data_dir)
    #np.save('train_data.npy', train_data)
    month = torch.tensor([6] * train_data[0].shape[0]).long()

    dl = DataLoader(
        TensorDataset(
            train_data[0].float(),  # x
            train_data[1].bool(),  # mask
            train_data[2].long(),  # dynamic world
            train_data[3].float(),  # latlons
            month
        ),

    batch_size=32,
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


    np.save(os.path.join(output_dir,train_or_test+'_features.npy'), features_np)
    np.save(os.path.join(output_dir,train_or_test+'_labels.npy'), train_data[4].numpy()) 
    




if __name__ == "__main__":


    data_dir = Path(sys.argv[1])
    pretrained_model = Presto.load_pretrained()
    pretrained_model.eval()


    ids = list(set([ str(filename).split("/")[-1].split("_")[0]
             for filename in list((data_dir).glob("*_S2_*"))]))
    random.shuffle(ids)
    split_frac=float(sys.argv[3])


    
    test_size = int(len(ids) * split_frac)

    # split the ids into training and test sets
    train_ids = ids[test_size:]  # 90% for training
    test_ids = ids[:test_size]
    save(train_ids,data_dir,sys.argv[2],"train")
    save(test_ids,data_dir,sys.argv[2],"test")






