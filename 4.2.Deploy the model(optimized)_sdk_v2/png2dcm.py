# This script uses PyDicom library (https://pydicom.github.io/) to 
# generate a DICOM file from a supplied PNG image. 

import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from PIL import Image
import numpy as np
import zipfile
import io

# Read png from zip file. The code below assumes sample.zip which is a part of 
# the PADCHEST dataset
# zf = zipfile.ZipFile("./sample.zip")
# data = zf.read("255433269247415893224655601475580025849_j5s1kc.png")
data = 'sample.png'
image2d = np.array(Image.open(data)).astype(float)
image2d = (image2d/255).astype(np.uint16)

file_meta = FileMetaDataset()
file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
file_meta.MediaStorageSOPInstanceUID ='2.25.34327501276176110812231595851948283641'
file_meta.ImplementationClassUID = '1.3.6.1.4.1.30071.8'
file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

ds = Dataset()
ds.file_meta = file_meta

ds.Rows = image2d.shape[0]
ds.Columns = image2d.shape[1]
ds.NumberOfFrames = 1

ds.PixelSpacing = [1, 1] # in mm
ds.SliceThickness = 1 # in mm

ds.SeriesInstanceUID = pydicom.uid.generate_uid()
ds.StudyInstanceUID = pydicom.uid.generate_uid()

ds.PatientName = "Demo^RSNA2021"
ds.PatientID = "123456"
ds.Modality = "CR"
ds.StudyDate = '20211204'
ds.ContentDate = '20211204'

ds.BitsStored = 16
ds.BitsAllocated = 16
ds.HighBit = 15
ds.PixelRepresentation = 0
ds.PhotometricInterpretation = "MONOCHROME2"
ds.SamplesPerPixel = 1

ds.RescaleIntercept = 900
ds.RescaleSlope = 9
ds.WindowCenter = 2000
ds.WindowWidth = 2000

ds.is_little_endian = True
ds.is_implicit_VR = False

ds.PixelData = image2d.tobytes()

pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)
ds.save_as("sample_dicom.dcm", write_like_original=False)

