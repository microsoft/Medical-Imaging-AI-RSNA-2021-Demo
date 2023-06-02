from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse
import json, os, io
import numpy as np
import torch
import intel_extension_for_pytorch as ipex
import torchxrayvision as xrv
from torchvision import transforms
from torchxrayvision.datasets import normalize
import pydicom

import time
from openvino.runtime import Core
from openvino.runtime import get_version

def init():
    global bench_time
    bench_time = 10  # benchmark time in sec
    global target_device
    target_device = "CPU"

    # Initial PyTorch model
    global modelx
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'az-register-models', 'pc-densenet-densenet-best.pt')
    # print(model_path)
    modelx = torch.load(model_path)
    modelx.eval()

    # Initial PyTorch IPEX model
    global ipex_modelx
    global traced_model
    ipex_modelx = ipex.optimize(modelx)

    # Initialize OpenVINO Runtime.
    global ov_compiled_model
    ie = Core()
    ov_xml = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'az-register-models', 'pc-densenet-densenet-best.onnx')
    # Load and compile the OV model
    ov_model = ie.read_model(ov_xml)
    ov_compiled_model = ie.compile_model(model=ov_model, device_name=target_device)



# TIP:  To accept raw data, use the AMLRequest class in your entry script and add the @rawhttp decorator to the run() function
#       more details in: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-advanced-entry-script
# Note that despite the fact that we trained our model on PNGs, we would like to simulate
# a scenario closer to the real world here and accept DICOMs into our score script. Here's how:
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
            def benchmark_pt(test_image):
                latency_arr = []
                end = time.time() + int(bench_time)

                print(f"\n==== Benchmarking PyTorch inference with Fake Data for {bench_time}sec on CPU ====")
                print(f"Input shape: {test_image.shape}")

                while time.time() < end:
                    start_time = time.time()
                    pt_result = modelx(test_image)
                    latency = time.time() - start_time
                    latency_arr.append(latency)

                # Process output
                index = np.argsort( pt_result.data.cpu().numpy() )
                probability = torch.nn.functional.softmax(pt_result[0], dim=0).data.cpu().numpy()
                pt_result = get_top_predictions(index, probability)

                avg_latency = np.array(latency_arr).mean()
                fps = 1 / avg_latency

                print(f"PyTorch Avg Latency: {avg_latency:.4f} sec, FPS: {fps:.2f}")

                #Return the result
                pt_summary = {
                    "fwk_version": f"PyTorch: {torch.__version__}",
                    "pt_result": pt_result,
                    "avg_latency": avg_latency,
                    "fps": fps
                }
                return pt_summary

            def benchmark_ipex(test_image):
                latency_arr = []
                end = time.time() + int(bench_time)

                with torch.no_grad():
                    traced_model = torch.jit.trace(ipex_modelx, test_image)
                    traced_model = torch.jit.freeze(traced_model)

                print(f"\n==== Benchmarking IPEX inference with Fake Data for {bench_time}sec on CPU ====")
                print(f"Input shape: {test_image.shape}")

                while time.time() < end:
                    start_time = time.time()
                    with torch.no_grad():
                        ipex_result = traced_model(test_image)
                    latency = time.time() - start_time
                    latency_arr.append(latency)

                # Process output
                index = np.argsort( ipex_result.data.cpu().numpy() )
                probability = torch.nn.functional.softmax(ipex_result[0], dim=0).data.cpu().numpy()
                ipex_result = get_top_predictions(index, probability)

                avg_latency = np.array(latency_arr).mean()
                fps = 1 / avg_latency

                print(f"PyTorch Avg Latency: {avg_latency:.4f} sec, FPS: {fps:.2f}")

                #Return the result
                ipex_summary = {
                    "fwk_version": f"IPEX: {ipex.__version__}",
                    "ipex_result": ipex_result,
                    "avg_latency": avg_latency,
                    "fps": fps
                }
                return ipex_summary

            def benchmark_ov(test_image):
                # get the names of input and output layers of the model
                input_layer = ov_compiled_model.input(0)
                output_layer =ov_compiled_model.output(0)

                latency_arr = []
                end = time.time() + int(bench_time)
                print(f"\n==== Benchmarking OpenVINO {bench_time}sec on {target_device} ====")
                print(f"Input shape: {test_image.shape}")

                while time.time() < end:
                    start_time = time.time()
                    ov_output = ov_compiled_model([test_image])
                    latency = time.time() - start_time
                    latency_arr.append(latency)

                # Process output
                ov_output = ov_output[output_layer]
                index = np.argsort(ov_output)
                probability = torch.nn.functional.softmax(torch.from_numpy(ov_output[0]), dim=0).data.cpu().numpy()
                ov_result = get_top_predictions(index, probability)

                avg_latency = np.array(latency_arr).mean()
                fps = 1 / avg_latency

                print(f"OpenVINO Avg Latency: {avg_latency:.4f} sec, FPS: {fps:.2f}")

                ov_summary = {
                    "fwk_version": f"OpenVINO: {get_version()}",
                    "ov_result": ov_result,
                    "avg_latency": avg_latency,
                    "fps": fps
                }
                return ov_summary

            # Read DICOM and apply photometric transformations
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

            # Decode output and get predictions
            def get_top_predictions(index, probability, num_predictions=3):
                # For labels definition see file: '3.Build a model/trainingscripts/padchest_config.py'
                pathologies_labels = ['Air Trapping', 'Aortic Atheromatosis', 'Aortic Elongation', 'Atelectasis',
                    'Bronchiectasis', 'Cardiomegaly', 'Consolidation', 'Costophrenic Angle Blunting', 'Edema', 'Effusion',
                    'Emphysema', 'Fibrosis', 'Flattened Diaphragm', 'Fracture', 'Granuloma', 'Hemidiaphragm Elevation',
                    'Hernia', 'Hilar Enlargement', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
                    'Pneumonia', 'Pneumothorax', 'Scoliosis', 'Tuberculosis']

                top_labels = []
                top_probs = []
                for i in range(num_predictions):
                    top_labels.append(pathologies_labels[index[0][-1-i]])
                    top_probs.append(round(probability[index[0][-1-i]] * 100, 2))

                result = {"top_labels": top_labels, "top_probabilities": top_probs}
                return result

            # Get System information
            def get_system_info():
                import subprocess


                # Run lscpu command and capture output
                lscpu_out = subprocess.check_output(['lscpu']).decode('utf-8')
                print(lscpu_out)
                # Run free -g command and capture output
                mem_out = subprocess.check_output(['free', '-g']).decode('utf-8')
                print(mem_out)
                os_out = subprocess.check_output(['cat', '/etc/os-release']).decode('utf-8')
                kernal_out = subprocess.check_output(['uname', '-a']).decode('utf-8')
                pyver_out = subprocess.check_output(['which', 'python']).decode('utf-8')
                os_out = os_out + " \n" + kernal_out + "\n" + pyver_out
                print(os_out)
                fwk_versions = {"PyTorch": torch.__version__,"IPEX" : ipex.__version__ , "OpenVINO": get_version()}
                print(fwk_versions)

                return_data = {
                    "lscpu_out": lscpu_out,
                    "mem_out_gb": mem_out,
                    "fwk_versions": fwk_versions,
                    "os": os_out
                }
                return return_data

            #
            # Start Processing
            #
            file_bytes = request.files["image"]

            # Note that user can define this to be any other type of image
            input_image = read_and_rescale_image(file_bytes)

            preprocess = transforms.Compose([
                xrv.datasets.XRayCenterCrop(),
                xrv.datasets.XRayResizer(224)
            ])

            input_image = preprocess(input_image)
            input_batch =  torch.from_numpy( input_image[np.newaxis,...] )

            #Benchmark PyTorch
            pt_summary = benchmark_pt(input_batch)
            print(f"PyTorch Output: {pt_summary}")

            #Benchmark IPEX
            ipex_summary = benchmark_ipex(input_batch)
            print(f"IPEX Output: {ipex_summary}")

            # Benchmark OpenVINO
            ov_summary = benchmark_ov(input_batch)
            print(f"OpenVINO Output: {ov_summary}")

            sys_info = get_system_info()

            return_data = {"pt_summary": pt_summary,
            "ipex_summary" : ipex_summary,
            "ov_summary": ov_summary,
            "system_info": sys_info}

            return return_data

        except Exception as e:
            result = str(e)
            # return error message back to the client
            return AMLResponse(json.dumps({"error": result}), 200)

    else:
        return AMLResponse("bad request", 500)

