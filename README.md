
> **DISCLAIMER**: This code is provided for research and development use only. This code is not intended for use in clinical decision-making or for any other clinical use and the performance of the code for clinical use has not been established.

# RSNA 2021 Demo

This repository contains code and instructions to recreate the medical imaging AI pipeline demo shown by Microsoft HLS team at RSNA 2021. 

# Demo scenario

As you navigate the demo, you will be playing several roles in a team that is aiming to build and deploy a chest radiograph classification system using cloud-based tools available in Microsoft Azure.

This demo will walk you through the steps that a data science team would typically undertake and describe the use of Microsoft tools as well as provide some custom code to take you through the steps. We chose a chest X-ray classification scenario since it is a well studied problem for which many great datasets are available. The focus of this demo is not on the algorithms used to build the best performing system, but rather the steps and tools that are needed to get one there. The same tools and principles could be applied to many other types of medical imaging datasets. 

> **Note**: In the real world you will likely deal with DICOM data. A follow-up to this demo will cover working with DICOM and the various aspects of handling real-world clinical data. In this demo we use real medical images converted to 16bit grayscale PNGs for simplicity. 

The following diagram represents the steps we will be modeling: 
![Flow diagram](./images/flow.png)

# Prerequisites

* [Azure account](https://azure.microsoft.com/en-us). Since this demo relies on the use of Azure cloud, you will need to create an account. 
* [Azure ML workspace](https://azure.microsoft.com/en-us/services/machine-learning). Azure Machine Learning is one of the many services provided on the Azure platform. A Workspace is a separate portal provided by the service with which you will be interacting through the demo. Azure ML Workspace provides controls that allow interacting with compute resources, writing code in shared Notebooks, annotating datasets, managing datasets and more. Once you create your Azure account, you will need to create the workspace. Follow instructions for creating a workspace on the [Azure ML Documentation page](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources#create-the-workspace) 

With that, you are ready to dive into the data science!

# Demo steps
The next steps include:
1. [Loading Data into the cloud](1.Load%20Data/README.md)  
    Here we will load a sample dataset into Azure Blob Storage and register it with Azure Machine Learning for further analysis
2. [Annotate the dataset](2.Annotation/README.md)  
Here, we cover how to annotate your medical imaging dataset using tools built into the Azure ML Studio. Note that in this demo, the actual training will use the labels available with our sample dataset so that you won't have to manually label all the images. But we will show you how to leverage existing labels and those that you create through the annotation tool. 
3. [Build a model from the annotated dataset](3.Build%20a%20model/Readme.md)  
In this portion we will show you how to set up your compute resources and then use Notebooks feature of the Azure ML Studio to submit an experiment, resulting in a machine learning model that is capable of analyzing XRay images.
4. [Deploy the model as a RESTful endpoint and perform explainability analysis](4.Deploy%20the%20model/Readme.md)<br>
 Finally, we will demonstrate how to deploy a model so that it's ready for inferencing, and then how to use this model to gain insights into how the model is making predictions. 

# Ask us about

## InnerEye - Advanced Machine Learning algorithms for medical imaging
If you are looking for advanced algorithms for medical image analysis that harness the power of Azure-based cloud computing - make sure to check out the amazing work done by the InnerEye team from Microsoft Research Cambridge and their open-source repositories: 
* [InnerEye-DeepLearning](https://github.com/microsoft/InnerEye-DeepLearning) for advanced medical image analysis algorithms 
* [HI-ML](https://github.com/microsoft/hi-ml) - building blocks for AI/ML scenarios for medical applications: 

## DICOM Service
A real world medical imaging AI setup would leverage DICOM data and would benefit from storing DICOM images in the cloud. You can explore the services that Microsoft provides to manage DICOM and FHIR data in the cloud: 
* [DICOM service](https://docs.microsoft.com/en-us/azure/healthcare-apis/dicom/dicom-services-overview)
* [FHIR service](https://docs.microsoft.com/en-us/azure/healthcare-apis/fhir/overview)

## Using this tutorial with DICOM files
As mentioned, this tutorial uses PNG images instead of DICOM images, however, everything in this tutorial can be done with DICOM images.  We are planning to amend this tutorial to use DICOM images, in the meantime please contact us if you are having any issues using DICOM images.

## Importing labels into Azure ML projects
Real world scenarios often come with datasets that are partially labeled, or weakly labeled, and need to be reviewed. Azure ML provides ability to import labels through the [API](https://docs.microsoft.com/en-us/rest/api/azureml/). We are planning to provide a tutorial on the best practices to import medical imaging labels, in the meantime contact us to learn more if you are interested. 

## Advanced annotation scenarios for 3D and 4D
We are working on an advanced annotation tool that provides more ways of annotating medical imaging datasets. Contact our team to learn more. 

![Advanced Medical Imaging Labeler](./images/amil.png)

# Extras

## Learning materials
Here are some learning materials if you would like to explore some of the Microsoft's AI tools further: 
* [A 30 day challenge](https://docs.microsoft.com/en-us/learn/challenges?id=8E1F62A7-99E3-48E4-9EC9-1FFFB99EE9AF&wt.mc_id=cloudskillschallenge_8E1F62A7-99E3-48E4-9EC9-1FFFB99EE9AF)  focusing on learning AI fundamentals
* [Interactive introduction into Azure ML](https://docs.microsoft.com/en-us/learn/modules/intro-to-azure-ml/)

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repositories using our CLA.

These are the ways to contribute:
* [Submit bugs](https://github.com/microsoft/Medical-Imaging-AI-RSNA-2021-Demo/issues) and help us verify fixes as they are checked in.
* Review the [source code changes](https://github.com/microsoft/Medical-Imaging-AI-RSNA-2021-Demo/pulls).
