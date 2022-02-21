import argparse
import argparse
import os
import numpy as np
import torch
import torch.backends as b
import torch.nn as nn
import torchvision

backend = "qnnpack" # arm
#backend = 'fbgemm' #x86

def torch2script(torch_model_path, script_module_path):
    loaded_model = None
    if torch_model_path:
        if not os.path.exists(torch_model_path):
            raise('Torch model path does not exist')
        print('Loading model from .pth')
        loaded_dict = torch.load(torch_model_path)
    else:
        #TODO: install pytorch_qunatisation (trt)
        print('Loading model from hub')
        loaded_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd').cpu()
        utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
    print('Completed loading model')

    if not loaded_model:
        raise('No model has been loaded')

    loaded_model.eval()
    
    with torch.no_grad():
        dummy_input = torch.ones(1, 3, 300, 300)
        print('Started tracing')
        traced_script_module = torch.jit.trace(loaded_model, dummy_input)
        print('Finished tracing and saving')
        output_path = os.path.join(script_module_path, 'ssd_300_300_coco_fp32_traced.pt')
        traced_script_module.save(output_path)
        print(f'Generated \"{output_path}\" of size {round(os.path.getsize(output_path) * 1e-6, 2)} MB')

        dummy_input = torch.ones(1, 3, 300, 300).type(torch.uint8)
        loaded_model.qconfig = torch.quantization.get_default_qconfig(backend)
        b.quantized.engine = backend
        loaded_model_static_quantized = torch.quantization.prepare(loaded_model, inplace=False)
        loaded_model_static_quantized = torch.quantization.convert(loaded_model_static_quantized, inplace=False)
        traced_script_module = torch.jit.trace(loaded_model_static_quantized, dummy_input)
        output_quantized_path = os.path.join(script_module_path, 'ssd_300_300_coco_int8_traced.pt')
        traced_script_module.save(output_quantized_path)
        print(f'Generated \"{output_quantized_path}\" of size {round(os.path.getsize(output_quantized_path) * 1e-6, 2)} MB')

        print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_model_path', type=str, default='')
    parser.add_argument('--script_module_path', type=str, default=os.getcwd())

    args = parser.parse_args()

    torch2script(args.torch_model_path, args.script_module_path)
