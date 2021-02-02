import torch
from torchvision import transforms
from torchvision.models import vgg11


import urllib
from PIL import Image



vgg11 = vgg11(pretrained = True)

url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)



from torchvision import transforms
from torchvision.models import vgg11
from bifrost.runner.run import run_torch_stonne
z
# Instansiate torch model
vgg11 = vgg11(pretrained = True)

# Get input
input_image = Image.open("dog.jpg")
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# Run
out = run_torch_stonne(vgg11, input_batch)