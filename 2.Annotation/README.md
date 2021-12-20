# Data Annotation

## Relevant Links
* [Create Azure Machine Learning datasets (Python)](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-register-datasets)
* [Connect to data with the Azure Machine Learning studio](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-connect-data-ui)
* [Create a labeling project for multi-class image classification](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-labeling)

## Prerequisites
* Azure ML Workspace
* [Data loaded into blob storage](../1.Load%20Data/README.md)

## Steps
1. Create Datastore and Dataset
2. Create Labeling Project and Annotate Data
3. Export and use Labeled data

## Creating a Datastore and Dataset

This guide will briefly show you how to create a datastore and dataset through the azure portal. For a more complete tutorial please see the guide on creating dataset using [python sdk](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-register-datasets) or through the [azure portal](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-connect-data-ui).

### Obtaining Azure Storage Key
To obtain an azure storage key, navigate to the storage account that contains your blob container, and select `Access keys` from the menu on the left. In this tab, click `Show keys` at the top, then copy the desired key (either one will work).

![Access Keys](./images/access-keys.png)

### Creating a Datastore 
Next, navigate back to the AzureML Studio and select `Datastores` in the menu on the left, then click `New Datastore`.  This will bring up a prompt, fill out the form and click `Create`.

![Create Datastore](./images/create-datastore.jpg)

### Creating Dataset
After creating a Datastore you can create a Dataset which allows fast access to the files for training/testing.  Go to `Datasets` from the AzureML Studio menu, then click `Create dataset`, and choose `From datastore` from the dropdown menu.

![Create Dataset](./images/create-dataset.jpg)

Next, in the prompts that appear set the name to `padchest` (or any unique name), and Dataset type to `File` then click `Next`.  In the next menu, find and select the datastore that you previously created in the search bar and set the path to whereever you saved the files within the blob storage (probably `/`). Finally, Complete the remaining prompts to create the dataset.

![Create Dataset](./images/create-dataset-2.jpg)

### Example dataset usage

Once a dataset has been created they are easy to use.  Here is some example code:

```python
# azureml-core of version 1.0.72 or higher is required
from azureml.core import Workspace, Dataset
import pandas as pd
import os
os.environ["RSLEX_DIRECT_VOLUME_MOUNT"] = "true" # IMPORTANT for performance

# When running from inside an AzureML notebook, 
# workspace can be pulled from the environment
workspace = Workspace.from_config()

# Load workspace outside of azureML notebooks
#subscription_id, resource_group, workspace_name = '<sub_id>', '<rg_name>', '<ws_name>
workspace = Workspace(subscription_id, resource_group, workspace_name)

# Find specified dataset
dataset_name = "padchest"
dataset = Dataset.get_by_name(workspace, name=dataset_name)

# On AzureML mounting it very easy to mount and use files
mount = dataset.mount()
mount.start()
print(mount.mount_point)

pc_csv_file = os.path.join(mount.mount_point, "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv")
pc_df = pd.read_csv(pc_csv_file, low_memory=False, index_col=0)

# You can also download the dataset locally
# dataset.download(target_path='.', overwrite=False)
```

## Create Labeling Project and Annotating Data

> **Note:** This step is not required to complete this tutorial you can jump directly to the next steps to continue the tutorial.

Now that we have created a dataset, we can create a Labeling Project and start annotating data. For an indepth look at this process checkout the AzureML documentation on it [here](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-labeling). 

To create a labeling project, navigate to `Data Labeling` in from the Azure ML Studio menu, then select `Add project`. From the prompt give your labeling project a unique name, set the media type to `Image`, then select a Labeling task type. You can create a few different labeling projects of different types to experiment.

![Create Label Project](./images/create-label-project-1.jpg)

![Create Label Project](./images/create-label-project-2.jpg)


Go through the creation prompts until you reach the `Select or create dataset` page, on this menu select the dataset we create in previous steps (`padchest`).

![Create Label Project](./images/create-label-project-3.jpg)

Continue to through the prompts, until the end and click `Create project`. 

![Create Label Project](./images/create-label-project-4.jpg)

The project can take sometime to create, especially if there are numerous files within the dataset. Once the project has been created the `State` will show `Running`.

![Create Label Project](./images/create-label-project-5.jpg)

### Annotating Data

Now that we have created labeling projects, we can start annotating the data. A more complete guide to labeling can be found [here](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-labeling#start-labeling), here we show a brief introduction.

To start annotating images, navigate to the Data Labeling page by clicking `Data Labeling` within AzureML studio (the same page from which you created the Labeling Project).  From here select a Project name to view details.

![Dashboard](./images/labeling-dashboard.png)

From this page select `Label data`, review the instructions then click `Start labeling` at the bottom. 


![Instructions](./images/labeling-instr.png)


From this page you can start annotating data based on the Label data type. Note that Azure ML supports regular (PNG/JPEG) images for annotations as well as DICOMs (only 2D, Xray modalities at the moment of writing):

#### Segmentation
![Segmentation](./images/labeling-seg.png)

#### Bounding Box/Detection
![Detection](./images/labeling-box.png)

#### Whole Image Classification
![Segmentation](./images/labeling-class.png)

### Export Labeling Data

Once you have annotated a project, you can export these labels into various formats from the Labeling project Dashboard. To do this, navigate to the desired Labeling project Dashboard and click `Export` then select the appropriate option from the dropdown menu.

![Export Labels](./images/label-export-1.png)


By selecting, `Azure ML dataset`, this will create a versioned representation of your dataset that can be easily used throughout Azure ML by features such as `Automated ML`, `Notebooks`, and `Designer`.

![Labeled Dataset](./images/labeled-dataset.png)


## Next Steps
Now you are ready for to [build a model](../3.Build%20a%20model/Readme.md) in Azure ML!
