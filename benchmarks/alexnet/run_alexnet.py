import tvm
from tvm import relay
from tvm.contrib import graph_runtime as runtime

# Import this add stonne as an x86 co-processor
import bifrost
from bifrost.stonne.simulator import config_simulator
from bifrost.runner.run import run_torch_stonne

# Download an example image from the pytorch website
import urllib
from PIL import Image
from torchvision import transforms
import torch


url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model


