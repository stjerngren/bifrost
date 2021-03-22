from os import pathsep
import tvm
from tvm import relay
from tvm.contrib import graph_runtime as runtime

# Import this add stonne as an x86 co-processor
import bifrost
from bifrost.stonne.simulator import architecture
from bifrost.runner.run import run_torch_stonne
from alexnet import alex_model

################################################################################
# Choose eval settings here
################################################################################

# choose maeri or sparse
architecture_setting = "maeri"

# chosoe sparsity ratio (ignored if not sigma)
sparsity_ratiio = 0

# choose tile config: performance, opt, stonne_paper, basic
tiles = "performance"


#################################################################################
# Do not change anything after this
architecture.ms_size = 128
architecture.dn_bw=64
architecture.rn_bw=64
if architecture == "sparse":
    architecture.controller_type = "SIGMA_SPARSE_GEMM"
    architecture.sparsity_ratio = 0
if sparsity_ratiio:
    if sparsity_ratiio == 50:
        from weight_pruning import model as alex_model
    architecture.sparsity_ratio = 0


architecture.create_config_file()


conv_paths = [
    "/Users/axelstjerngren/uni/Year4/ProjectLevel4/level-4-project/benchmarks/alexnet/tiles/%s/conv_1.txt" % tiles,
    "/Users/axelstjerngren/uni/Year4/ProjectLevel4/level-4-project/benchmarks/alexnet/tiles/%s/conv_2.txt" % tiles,
    "/Users/axelstjerngren/uni/Year4/ProjectLevel4/level-4-project/benchmarks/alexnet/tiles/%s/conv_3.txt" % tiles,
    "/Users/axelstjerngren/uni/Year4/ProjectLevel4/level-4-project/benchmarks/alexnet/tiles/%s/conv_4.txt" % tiles,
    "/Users/axelstjerngren/uni/Year4/ProjectLevel4/level-4-project/benchmarks/alexnet/tiles/%s/conv_5.txt" % tiles
]
fc_paths = [
    "/Users/axelstjerngren/uni/Year4/ProjectLevel4/level-4-project/benchmarks/alexnet/tiles/%s/fc_1.txt" % tiles,
    "/Users/axelstjerngren/uni/Year4/ProjectLevel4/level-4-project/benchmarks/alexnet/tiles/%s/fc_2.txt" % tiles,
    "/Users/axelstjerngren/uni/Year4/ProjectLevel4/level-4-project/benchmarks/alexnet/tiles/%s/fc_3.txt" % tiles, 
]
architecture.load_tile_config(
    conv_cfg_paths = conv_paths,
    fc_cfg_paths = fc_paths
    )

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

import time 
start = time.time()

run_torch_stonne(alex_model, input_batch)

end = time.time()
print(end - start)