import json

ms_size_64_cycles  = list()
ms_size_64_psums = list()

with open("/Users/axelstjerngren/uni/Year4/ProjectLevel4/level-4-project/bifrost/bifrost_temp/test_2_layer.log", "r") as f:
    for line in f:
        config = json.loads(line)
        if config["result"][2] != 1000000000:
            size = config["config"]["entity"][8][2]
            if size == 64:
                ms_size_64_cycles.append(config["result"][2])

with open("/Users/axelstjerngren/uni/Year4/ProjectLevel4/level-4-project/bifrost/bifrost_temp/test_2_layer_psums.log", "r") as f:
    for line in f:
        config = json.loads(line)
        if config["result"][2] != 1000000000:
            size = config["config"]["entity"][8][2]
            if size == 64:
                ms_size_64_psums.append(config["result"][2])


import matplotlib
import matplotlib.pyplot as plt

plt.plot(ms_size_64_cycles, label = "Cycles")
plt.plot(ms_size_64_psums, label = "Psums")
plt.title("STONNE conv2d tuning", fontsize=16, fontweight='bold')
plt.suptitle("Improvement", fontsize=10)
plt.legend(loc='upper right')
plt.ylabel("Cycles")
plt.show()