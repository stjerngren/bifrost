import json

ms_size_32  = list()
ms_size_64  = list()
ms_size_128 = list()

with open("test.log", "r") as f:
    for line in f:
        config = json.loads(line)
        if config["result"][2] != 1000000000:
            size = config["config"]["entity"][8][2]
            if size == 32:
                ms_size_32.append(config["result"][2])
            if size == 64:
                ms_size_64.append(config["result"][2])
            if size == 128:
                ms_size_128.append(config["result"][2])

ms_size_32.sort(reverse=True)
ms_size_64.sort(reverse=True)
ms_size_128.sort(reverse=True)

print(ms_size_32 )
print(ms_size_64 )
print(ms_size_128)

import matplotlib
import matplotlib.pyplot as plt

plt.plot(ms_size_32, label = "32")
plt.plot(ms_size_64, label = "64")
plt.plot(ms_size_128, label = "128")
plt.title("STONNE conv2d tuning", fontsize=16, fontweight='bold')
plt.suptitle("Improvement", fontsize=10)
plt.legend(loc='upper right')
plt.ylabel("Cycles")
plt.show()