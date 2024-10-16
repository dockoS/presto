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
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm


import utils
import presto 

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
      


def process_pixel_for_all_sensors(id,pos_i,pos_j, data_dir):
    # Get the list of files and sort them


    s2_data,_=process_pixel(id,pos_i,pos_j, data_dir,sensor_prefix="_S2_*.tif")
    s1_data,_=process_pixel(id,pos_i,pos_j, data_dir,sensor_prefix="_S1_*.tif")
    dynamic_world_data,_=process_pixel(id,pos_i,pos_j, data_dir,sensor_prefix="_dynamic_world_*.tif")
    srtm_data,_=process_pixel(id,pos_i,pos_j, data_dir,sensor_prefix="_srtm*.tif")
    srtm_data=np.repeat(srtm_data,s2_data.shape[0],axis=0)
    label_data,latlon=process_pixel(id,pos_i,pos_j, data_dir,sensor_prefix="_label.tif")
    x, mask, dynamic_world = utils.construct_single_presto_input(
                    s2=torch.from_numpy(s2_data).float(), s2_bands=['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10','B11', 'B12'],
                    s1=torch.from_numpy(s1_data).float(), s1_bands=['VV', 'VH'],
                    dynamic_world=torch.from_numpy(dynamic_world_data).float(),
                    #srtm=torch.from_numpy(srtm_data[:,:2]).float(),
                    #srtm_bands= ["elevation", "slope"]
                )


    return x, mask, dynamic_world, torch.tensor(latlon),label_data[0,0]

def process_id(id,data_dir):
    ff_path=list((data_dir).glob(f"{id}_S2_*.tif"))[0]
    tif_file = rioxarray.open_rasterio(ff_path)
    _,h,w=tif_file.shape
    arrays, masks, dynamic_worlds, latlons, labels= [], [], [], [], []
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

    
    







       


def main():
    """
    Main function to process files and create chips.
    Expects three command line arguments:
        - data_dir: Directory containing the input GeoTIFF files.
        - output_dir: Directory to save the output chips.
        - chip_size: Size of the square chips.
    """
    if len(sys.argv) != 4:  # noqa: PLR2004
        print("Usage: python script.py <data_dir> <output_dir> <chip_size>")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    chip_size = int(sys.argv[3])


    ids = list(set([ str(filename).split("/")[-1].split("_")[0]
             for filename in list((data_dir).glob("*_S2_*"))]))

    train_data=process_all_ids(ids,data_dir)
    #np.save('train_data.npy', train_data)
    

    # dataset = TensorDataset(
    #         train_data[0].float(),  # x
    #         train_data[1].bool(),  # mask
    #         train_data[2].long(),  # dynamic world
    #         train_data[3].float(),  # latlons,
    #         month,  # month,
    #         train_data[4].long(),  # labels

    #     )
        # Fonction pour remapper les étiquettes
    def remap_labels(labels):
        # Initialiser toutes les étiquettes à -1 (pour celles qu'on ne veut pas utiliser)
        binary_labels = torch.full_like(labels, -1)
        
        # Classe 11 devient 0
        binary_labels[labels == 11] = 0
        
        # Classes [8, 9, 17] deviennent 1
        binary_labels[(labels == 8) | (labels == 9) | (labels == 17)] = 1
        
        return binary_labels

    # Remappage des étiquettes dans ton dataset
    binary_labels = remap_labels(train_data[4].long())

    # Filtrer les données pour ne garder que les étiquettes 0 et 1
    mask = (binary_labels == 0) | (binary_labels == 1)

    month = torch.tensor([6] * train_data[0][mask].shape[0]).long()
    # Créer le dataset filtré avec les nouvelles étiquettes binaires
    dataset = TensorDataset(
        train_data[0][mask].float(),  # x
        train_data[1][mask].bool(),   # mask
        train_data[2][mask].long(),   # dynamic world
        train_data[3][mask].float(),  # latlons,
        month,                        # month,
        binary_labels[mask].long()    # étiquettes binaires
    )

    
    print("Dataset size:", len(dataset))


    pretrained_model = Presto.load_pretrained()
    #pretrained_model.eval()
    features_list = []
    # for (x, mask, dw, latlons, month) in tqdm(dl):
    #     dw=dw[:,:,0]
    #     #dw=torch.zeros_like(dw)+9
    #     #print('iccii',dw.shape)

    #     with torch.no_grad():
    #         encodings = (
    #             pretrained_model.encoder(
    #                 x, dynamic_world=dw, mask=mask, latlons=latlons, month=month
    #             )
    #             .cpu()
    #             .numpy()
    #         )
    #         features_list.append(encodings)
   

    # features_np = np.concatenate(features_list)   
    

    # np.save('./training_data/features.npy', features_np)
    # np.save('./training_data/labels.npy', train_data[4].numpy()) 

    finetuning_model=pretrained_model.construct_finetuning_model(num_outputs=1, regression=True)



    # create a TensorDataset
 

    # determine sizes of training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dl = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    )

    # create a DataLoader for the validation set
    val_dl = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
    )
  
   

    # Assuming PrestoFineTuningModel is imported or defined above

    # Replace with your dataset and DataLoader
                

    # Define model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = finetuning_model.to(device)
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
   
        correct_predictions = 0
        total_predictions = 0
        # Loop through batches
        for (x,mask,dw,latlon,month,labels) in tqdm(train_dl, desc=f"Epoch {epoch+1}/{num_epochs}"):
            #inputs, targets = inputs.to(device), targets.to(device)
            x=x.to(device)
            mask=mask.to(device)
            dw=dw[:,:,0].to(device)
            latlon=latlon.to(device)
            month=month.to(device)
            targets=labels.to(device)
            #print(x.shape,mask.shape,dw.shape,latlon.shape,month.shape,targets.shape)
            

            # Zero the gradient
            optimizer.zero_grad()
            print(x)
            # Forward pass
            outputs = model(x, dw, latlon, mask, month).squeeze()

            loss = criterion(outputs, targets.float())
          

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track loss
            running_loss += loss.item()
            predictions = torch.round(torch.sigmoid(outputs))
            correct_predictions += (predictions == targets).sum().item()
            total_predictions += targets.size(0)
        
        avg_loss = running_loss / len(train_dl)
        accuracy = correct_predictions / total_predictions * 100
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")


    print("Training complete.")
    model.eval()
    val_running_loss = 0.0
    val_correct_predictions = 0
    val_total_predictions = 0
    with torch.no_grad():
        for (x,mask,dw,latlon,month,labels) in tqdm(val_dl):
            x, mask, dw, latlon, month, val_targets = x.to(device), mask.to(device), dw[:,:,0].to(device), latlon.to(device), month.to(device), labels.to(device)
            # Forward pass
            val_outputs = model(x, dw, latlon, mask, month).squeeze()
            val_loss = criterion(val_outputs, val_targets.float())

            # Track validation loss
            val_running_loss += val_loss.item()

            # Compute validation accuracy
            val_predictions = torch.round(torch.sigmoid(val_outputs))
            val_correct_predictions += (val_predictions == val_targets).sum().item()
            val_total_predictions += val_targets.size(0)
    
    val_avg_loss = val_running_loss / len(val_dl)
    val_accuracy = val_correct_predictions / val_total_predictions * 100
    print(f"Validation - Loss: {val_avg_loss:.4f}, Accuracy: {val_accuracy:.2f}%")


    exit()
    features_np = np.load('./training_data/features.npy')
    labels_np = np.load('./training_data/labels.npy')
    # concatenate features
   
    # get labels
    #labels_np = train_data[4].numpy()
    print(np.unique(labels_np))

    # filter out instances where label is zero
    classes=[8, 9, 11, 17]

    # filter out instances where label is not in [8, 9, 11, 17]
    selected_indices = np.isin(labels_np,classes )
    features_np_filtered = features_np[selected_indices]
    labels_np_filtered = labels_np[selected_indices]
    #labels_np_filtered = np.where(labels_np_filtered == 11, 0, 1)

    # assuming 'labels_np' is your array
    unique_labels, counts = np.unique(labels_np_filtered, return_counts=True)

    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count} occurrences")
    # split data into training and test sets
    features_train, features_test, labels_train, labels_test = train_test_split(
        features_np_filtered, labels_np_filtered, test_size=0.2, random_state=42)

    # train the model on the training set
    model = RandomForestClassifier(class_weight="balanced", random_state=42)
    model.fit(features_train, labels_train)

    # make predictions on the test set
    predictions = model.predict(features_test)

    # compute accuracy
    accuracy = accuracy_score(labels_test, predictions)
    print(f"Accuracy: {accuracy}")

    # # train the model
    # model = RandomForestClassifier(class_weight="balanced", random_state=42)
    # model.fit(features_np_filtered, labels_np_filtered)

    # # make predictions on the training set
    # predictions = model.predict(features_np_filtered)

    # # compute accuracy
    # accuracy = accuracy_score(labels_np_filtered, predictions)
    # print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()


