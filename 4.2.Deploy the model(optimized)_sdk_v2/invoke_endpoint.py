import requests

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential

# enter details of your AML workspace
subscription_id = "<SUBSCRIPTION_ID>"
resource_group = "<RESOURCE_GROUP>"
workspace_name = "<AML_WORKSPACE_NAME>"

# get a handle to the workspace
ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)

online_endpoint_name = "padchest-pt-ipex-ov-sdk-v2"
endpoint_deployed = ml_client.online_endpoints.get(name=online_endpoint_name)


test_file = "./sample_dicom.dcm"
files = {'image': open(test_file, 'rb').read()}

# resp = requests.post(scoring_uri, input_data, headers=headers)
scoring_uri = endpoint_deployed.scoring_uri
auth_key = ml_client.online_endpoints.get_keys(online_endpoint_name).primary_key
print(f"Authkye:{auth_key}")

print(f"Sending request {test_file} to {scoring_uri}")
# Send the DICOM as a raw HTTP request and obtain results from endpoint.
response = requests.post(scoring_uri, headers={"Authorization": f"Bearer {auth_key}"},files=files, timeout=60)
print("output:", response.content)
