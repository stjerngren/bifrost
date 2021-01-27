import tvm
from tvm import relay, runtime
alex_model = None

# Import this add stonne as an x86 co-processor
import bifrost
from bifrost.stonne.simulator import config_simulator

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



test = torch.jit.trace(alex_model, input_batch).eval()
mod, params = relay.frontend.from_pytorch(test, [("test", input_batch.shape)])

config_simulator(
    ms_size=16,
    reduce_network_type="ASNETWORK",
    dn_bw=8,
    rn_bw=8,
    controller_type="MAERI_DENSE_WORKLOAD",
)

target = "llvm -libs=stonne"
lib = relay.build(mod, target=target, params=params)

ctx = tvm.context(target, 0)
module = runtime.GraphModule(lib["default"](ctx))
module.set_input("test", input_batch)
module.run()

out = module.get_output(0)
print(out)