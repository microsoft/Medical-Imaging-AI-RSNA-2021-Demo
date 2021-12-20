# Build a model

To train the model using this demo section, you need to open the Notebook that sits beside this Readme file in your Azure ML workspace. For this, follow the steps below:

## Steps
1. Set Azure ML studio for Notebook execution
2. Clone git repository
4. Run Jupyter Notebook to train the model

## 1. Set up Azure ML studio for Notebook execution

### Set up compute resources 
Before running a python script, you will need to connect to a 
[compute instance](https://docs.microsoft.com/en-us/azure/machine-learning/concept-compute-instance). To setup your your compute instance, please see the following article:
* [How to create a compute instance in your workspace.](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-run-jupyter-notebooks#run-a-notebook-or-python-script) 

To train model, you will create a compute cluster. Please follow the following link:
* [Create an Azure Machine Learning compute cluster.](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=python)


## 2. Clone git repository
To clone this Git repository, follow the link below to access a compute instance terminal in your workspace. 
* [How to access a compute instance terminal in your workspace](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-access-terminal)

From the terminal, you get access to a shared workspace-wide storage space for your notebooks, and you can run git commands to clone the repository. 

## 3. Run a Jupyter Notebook to train a model
To train the model, open [training.ipynb](./training.ipynb) from Azure ML studio. In this example, we use [Azure Machine Learning SDK for Python](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py), using the kernel 'Python 3.8 - AzureML', corresponding to [Azure Machine Learning SDK for Python v1.36.0](https://docs.microsoft.com/en-us/azure/machine-learning/azure-machine-learning-release-notes#2021-11-08). The SDK and kernels are configured for you automatically. Note that you can also open and run the notebook in Visual Studio Code provided that you have Azure ML plugin installed. 

Note that to track and reproduce your Azure ML projects' software dependencies as they evolve, we use [Python environments](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-environments). We use the following YAML file [environment.yml](../environment.yml) and set environment's name is 'rsna_2021_demo'. While you can configure environment manually in the portal, we will set up and register the environment via Azure ML SDK in the notebook.


![training_sdk_script](./images/training_jupyter_notebook.png). 