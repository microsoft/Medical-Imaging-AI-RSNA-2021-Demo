# Extended from: https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py

from skimage.io import imread
import os, os.path
import numpy as np
import pandas as pd
import torchxrayvision as xrv
from torchxrayvision.datasets import normalize, Dataset
import padchest_config

datapath = os.path.dirname(os.path.realpath(__file__))

class PC_Dataset_Custom(Dataset):
    """
    PadChest dataset
    Hospital San Juan de Alicante - University of Alicante
    
    PadChest: A large chest x-ray image dataset with multi-label annotated reports.
    Aurelia Bustos, Antonio Pertusa, Jose-Maria Salinas, and Maria de la Iglesia-Vayá. 
    arXiv preprint, 2019. https://arxiv.org/abs/1901.07441
    
    Dataset website:
    http://bimcv.cipf.es/bimcv-projects/padchest/
    """
    def __init__(self, imgpath, 
                 csvpath=os.path.join(datapath, "PADCHEST_chest_x_ray_images_labels_160K_01.02.19_DifferentialDiagnosis.csv"), 
                 views=["PA"],
                 transform=None, 
                 data_aug=None,
                 flat_dir=True, 
                 seed=0, 
                 unique_patients=True):

        super(PC_Dataset_Custom, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        
        self.pathologies = sorted(padchest_config.pathologies)
        
        mapping = dict()
        
        mapping["Infiltration"] = ["infiltrates",
                                   "interstitial pattern", 
                                   "ground glass pattern",
                                   "reticular interstitial pattern",
                                   "reticulonodular interstitial pattern",
                                   "alveolar pattern",
                                   "consolidation",
                                   "air bronchogram"]
        mapping["Pleural_Thickening"] = ["pleural thickening"]
        mapping["Consolidation"] = ["air bronchogram"]
        mapping["Hilar Enlargement"] = ["adenopathy",
                                        "pulmonary artery enlargement"]
        mapping["Support Devices"] = ["device",
                                      "pacemaker"]
        mapping["Tube'"] = ["stent'"] ## the ' is to select findings which end in that word
        
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.flat_dir = flat_dir
        self.csvpath = csvpath
        
        self.check_paths_exist()
        self.csv = pd.read_csv(self.csvpath, low_memory=False)

        # standardize view names
        self.csv.loc[self.csv["Projection"].isin(["AP_horizontal"]),"Projection"] = "AP Supine"
        
        # Keep only the specified views
        if type(views) is not list:
            views = [views]
        self.views = views
        
        self.csv["view"] = self.csv['Projection']
        self.csv = self.csv[self.csv["view"].isin(self.views)]

        # remove null stuff
        self.csv = self.csv[~self.csv["Labels"].isnull()]
        
        self.csv = self.csv[~self.csv["ImageID"].isin(padchest_config.missing_files)]
        
        if unique_patients:
            self.csv = self.csv.groupby("PatientID").first().reset_index()
            
        # filter out age < 10 (paper published 2019)
        self.csv = self.csv[(2019-self.csv.PatientBirth > 10)]
        
        # Get our classes.
        self.labels = []
        for pathology in self.pathologies:
            mask = self.csv["Labels"].str.contains(pathology.lower())
            if pathology in mapping:
                for syn in mapping[pathology]:
                    #print("mapping", syn)
                    mask |= self.csv["Labels"].str.contains(syn.lower())
            self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        
        ########## add consistent csv values
        
        # patientid
        self.csv["patientid"] = self.csv["PatientID"].astype(str)

        print('Pathologies:', self.pathologies)

    
        output_dir = './outputs'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file = output_dir + os.sep + 'class_names.txt'
        with open(output_file, 'w') as f:
            f.writelines(str(self.pathologies))


    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        if self.flat_dir:
            # Standard directory structure
            imgid = self.csv['ImageID'].iloc[idx]
        else:
            # Custom directory structure is folder / filename
            imgid = str(self.csv['ImageDir'].iloc[idx]) + os.sep + self.csv['ImageID'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path)

        sample["img"] = normalize(img, maxval=65535, reshape=True)

        if self.transform is not None:
            sample["img"] = self.transform(sample["img"])

        if self.data_aug is not None:
            sample["img"] = self.data_aug(sample["img"])

        return sample
