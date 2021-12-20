# Loading Data into the Cloud

## Overview
To make data available in your AzureML workspace, it needs to be uploaded to an Azure Blob Storage container. This can be done using command line tooling (AzCopy) or using the Azure Storage Explorer application.

## Relevant Documents:
* [Migrate on-premises data to cloud storage with AzCopy](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-migrate-on-premises-data?tabs=windows) 
* [Azure Storage Explorer](https://azure.microsoft.com/en-us/features/storage-explorer/#overview), [Documentation](https://docs.microsoft.com/en-us/azure/vs-azure-tools-storage-explorer-blobs)
* [Create a storage account](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-create?tabs=azure-portal)
* [Create Blob Container](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-create?tabs=azure-portal)

## Prerequisites
 * Azure Account
 * Tools: 
    * [AzCopy](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10?toc=/azure/storage/blobs/toc.json) and/or
    * [Azure Storage Explorer](https://azure.microsoft.com/en-us/features/storage-explorer/#overview)

## Steps:
1. Download and extract PADCHEST
2. Create a storage account and a blob container
3. Upload data 


## Downloading and Extracting PADCHEST

We are going to be using a dataset of chest x-ray images called PADCHEST. The dataset is available for download [here](https://bimcv.cipf.es/bimcv-projects/padchest/) and comes in 52 separate zip files (0.zip trough 50.zip, 54.zip) each containing png files. These pngs are accompanied by a CSV containing labels and other metadata for each image. You will need to register and download PADCHEST manually from the link above.

> **Note:** PADCHEST is quite large (~ 1TB), so you may wish to only upload a portion of PADCHEST to save time.  See the appendix for more details.

This tutorial expects PADCHEST to be stored with a particular structure. The root of the directory should contain the main csv file and a folder `png` which contains all of the sub-image directories., e.g.:
```
root
├── PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv
└── png
    ├── 0
    │    ├── 100069103068753688347522093561206841448_7197k3.png
    │    ├── 100081820231385537397079729591266436694_8o3uj2.png
    │    ├── 100360992970443012139948853258191567510_orx7ef.png
    │    └── ...
    ├── 1
    │    ├── 100035238701184647172015593785663345624_vb6v1o.png
    │    ├── 100035238701184647172015593785663345624_veh2l3.png
    │    ├── 100162183276649430079588535037640068150_b9bb9n.png
    │    └── ...
    ...
    └── 54
        ├── 100015392272639845228940263464822056020_hpfqdw.png
        ├── 100127962438106734621148265078974688397_nqkpyr.png
        ├── 100127962438106734621148265078974688397_o5y9d9.png
        └── ...
```




## Creating a Storage account and a blob container
Creating a storage account and a blob container is easy. First, create the storage account following [this tutorial](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-create?tabs=azure-portal). Second, create a new blob container by accessing the newly storage account, selecting `Container` in the left hand menu, and clicking `Create`, and finally follow the prompts.

![Create Container](images/create-container-1.png)

You can also create the blob container using Azure CLI with [az storage account create](https://docs.microsoft.com/en-us/cli/azure/azure-cli-reference-for-storage), or using AzCopy using the [azcopy make](https://docs.microsoft.com/en-us/azure/storage/common/storage-ref-azcopy-make?toc=/azure/storage/blobs/toc.json) command.

## Uploading data to Azure
Next, we need to copy the extracted PADCHEST data into our newly created blob container.  This can be done easily using [Azure Storage Explorer](https://docs.microsoft.com/en-us/azure/vs-azure-tools-storage-explorer-blobs#managing-blobs-in-a-blob-container).  If you prefer to use CLI to copy the data this is accomplished with [AzCopy](https://docs.microsoft.com/en-us/azure/storage/common/storage-ref-azcopy-copy?toc=/azure/storage/blobs/toc.json). 

### Copying using AzCopy
Before we can copy using AzCopy, we need to create a SAS token.  To do this, navigate to the blob container, then click `Shared access tokens`.
![Create SAS 1](./images/create-sas-token-1.png)

Next, select `Permissions` and select the following options:

![Permissions](./images/create-sas-token-2.png)

Optionally, change the expiration date of the SAS token, we recommend ~24 hours to upload the entire dataset.

Finally, use the generated Blob SAS URL in an azcopy command:

```
azcopy copy "<local/path>" "<blob SAS url>" --recursive
```



## Next Steps
Now you are ready to [annotate your data](../2.Annotation/Readme.md)!


## Appendix - Uploading only a portion of PADCHEST

Given the size of PADCHEST, you can save time in this tutorial by only using a portion of the dataset. To do this, only download/extract as few as only one of the individual zip files (make sure to maintain the folder structure mention above), then edit or create another version of the PADCHEST csv, e.g.: 
```python
import pandas as pd
fn = "<path/to/root>/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
df_full = pd.read_csv(fn, low_memory=False, index_col=0)
df_full[df_full['ImageDir'].isin(['0'])].to_csv("<path/to/root>/PADCHEST_SMALL.csv")
```