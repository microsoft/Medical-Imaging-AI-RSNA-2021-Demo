# Deploy the model and model explainability  (bonus)
**Deployment scenario:** Submit a DICOM file (x-ray image) to the cloud and get a model prediction in real time.

To deploy the model that was trained in the previous section ([3.Build a model](../3.Build%20a%20model/Readme.md), [training.ipynb](../3.Build%20a%20model/training.ipynb)) as a **web service hosted on Azure Container Instances (ACI)**
, you need to open the [deploy.ipynb](./deploy.ipynb) Notebook in your Azure ML workspace and follow the steps below:

## Steps
1. Prepare an entry script.
2. Prepare an inference configuration.
3. Deploy the model you trained before to the cloud.
4. Test the resulting web service.

To simulate a realistic scenario:
* The model to deploy was trained from 16 bit gray scale PNG images from [PadChest](https://pubmed.ncbi.nlm.nih.gov/32877839/).
* The deployed model accepts DICOM images as inputs.

### 1. Prepare an entry script.
In order to use a model for inferencing, you need to create a scoring script first. In the notebook that sits beside this Readme we have such script embedded. 
The scoring script is only required to have two functions:
* The `init()` function, which typically loads the model into a global object. 
* The `run(input_data)` function uses the model to predict a value based on the input data. 
    * In our case, input_data will be a DICOM file forma. <br>

The output of the scoring script is the model prediction in the format of a JSON object that will be passed into an HTTP response.


### 2. Prepare an inference configuration.
We will create:
* A lightweight environment to deploy the model.
* Use [AMLRequest](https://docs.microsoft.com/en-us/python/api/azureml-contrib-services/azureml.contrib.services.aml_request?view=azure-ml-py) and [AMLResponse](https://docs.microsoft.com/en-us/python/api/azureml-contrib-services/azureml.contrib.services.aml_response.amlresponse?view=azure-ml-py) classes to access DICOM raw data.
* An inference configuration to deploy the model as a web service using the entry script or scoring script [score.py](./score.py).


### 3. Deploy the model to the cloud.
Then, we will deploy the model as an ACI web service which will be exposed as an HTTP endpoint. 
* Specify the **deployment configuration** of the compute resource (i.e., CPU or GPU, amount of RAM, etc.) required for your application.
* ***Deploy*** by bringing  all together: i) model, ii) environment, iii) inference configuration (script [score.py](./score.py)) and iv) deployment configuration.
* Then Azure ML, will automatically deploy the model in the cloud and you will be able to send the data to your model.

### 4. Test the resulting web service.
We will load a DICOM file, send it to the Webservice we have deploed and display the response.

## Bonus: eXplainable AI (XAI)
The notebook also includes a model usage scenario which we built around the use case of model explainability.

As the adoption of AI in Healthcare translates into clinical practice, there is an unmeet need in providing clinical meaningful insights to doctors that explain how AI algorithms work. While most the AI (Deep Learning) algorithms operate as a **black-box** (i.e., do not provide explanations), here we show how to use common XAI methods (e.g., 
[SHAP](https://shap-lrjball.readthedocs.io/en/latest/generated/shap.DeepExplainer.html) and [M3d-Cam](https://github.com/MECLabTUDA/M3d-Cam)) to verify that the **trained model** is using expected pixel information from the image. 

The [explain.ipynb](./explain.ipynb) Notebook demonstrates:

* How to use integrate [SHAP](https://shap-lrjball.readthedocs.io/en/latest/generated/shap.DeepExplainer.html) and [M3d-Cam](https://github.com/MECLabTUDA/M3d-Cam) from trained Deep Learning models.
* How to load a trained model from a run directly into your code
* How to access data directly from the datastore (after the [1.Load Data](../1.Load%20Data/README.md) step). 

![Explainability](./images/explainability_shap.png)