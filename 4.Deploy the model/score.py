from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse
import json, os, io
import numpy as np
import torch
import torchxrayvision as xrv
from torchvision import transforms
from torchxrayvision.datasets import normalize
import pydicom

def init():
    global modelx
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'pc-densenet-densenet-best.pt')
    # print(model_path)
    modelx = torch.load(model_path)

# TIP:  To accept raw data, use the AMLRequest class in your entry script and add the @rawhttp decorator to the run() function
#       more details in: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-advanced-entry-script
@rawhttp
def run(request):

    if request.method == 'GET':
        # For this example, just return the URL for GETs.
        respBody = str.encode(request.full_path)
        return AMLResponse(respBody, 200)

    elif request.method == 'POST':
        # For a real-world solution, you would load the data from reqBody
        # and send it to the model. Then return the response.
        try:

            # For labels definition see file: '3.Build a model/trainingscripts/padchest_config.py'
            pathologies_labels = ['Air Trapping', 'Aortic Atheromatosis', 'Aortic Elongation', 'Atelectasis',
             'Bronchiectasis', 'Cardiomegaly', 'Consolidation', 'Costophrenic Angle Blunting', 'Edema', 'Effusion',
             'Emphysema', 'Fibrosis', 'Flattened Diaphragm', 'Fracture', 'Granuloma', 'Hemidiaphragm Elevation',
             'Hernia', 'Hilar Enlargement', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
             'Pneumonia', 'Pneumothorax', 'Scoliosis', 'Tuberculosis']

            # Read DICOM and apply transformations
            def read_and_rescale_image( filepath):
                dcm = pydicom.read_file(filepath)
                image = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept

                def window_image(image, wc, ww):
                    img_min = wc - ww // 2
                    img_max = wc + ww // 2
                    image[image < img_min] = img_min
                    image[image > img_max] = img_max
                    return image

                image = window_image(image, dcm.WindowCenter, dcm.WindowWidth)
                # Scales 16bit to [-1024 1024]
                image = normalize(image, maxval=65535, reshape=True)
                return image

            file_bytes = request.files["image"]

            # Note that user can define this to be any other type of image
            input_image = read_and_rescale_image(file_bytes)

            preprocess = transforms.Compose([
                xrv.datasets.XRayCenterCrop(),
                xrv.datasets.XRayResizer(224)
            ])

            input_image = preprocess(input_image)
            input_batch =  torch.from_numpy( input_image[np.newaxis,...] )
            
            with torch.no_grad():
                output = modelx(input_batch)

            index = np.argsort( output.data.cpu().numpy() )
            probability = torch.nn.functional.softmax(output[0], dim=0).data.cpu().numpy()

            #Return the result
            return {"top_three_labels":  [pathologies_labels[index[0][-1]], pathologies_labels[index[0][-2]], pathologies_labels[index[0][-3]] ],
                "probability":[ round(probability[index[0][-1]]*100,2), round(probability[index[0][-2]]*100,2), round(probability[index[0][-3]]*100,2) ] }

        except Exception as e:
            result = str(e)
            # return error message back to the client
            return AMLResponse(json.dumps({"error": result}), 200)

    else:
        return AMLResponse("bad request", 500)
