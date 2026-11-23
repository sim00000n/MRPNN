from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#FILE = r"C:\CODE\MRPNN-main\training_pipeline\logs\train_20260105_165622.log"
#FILE = r"C:\CODE\MRPNN-main\training_pipeline\logs\train_20260105_181507.log"
#FILE = r"C:\CODE\MRPNN-main\training_pipeline\logs\train_20260105_193135.log"
FILE = None#r"C:\CODE\MRPNN-main\training_pipeline\logs\train_20260222_160608.log"#r"C:\CODE\MRPNN-main\training_pipeline\logs\train_20260106_162129.log"

if FILE is None:
    log_dir = Path(r"C:\CODE\MRPNN-main\training_pipeline\logs")
    FILE = max(log_dir.iterdir(), key=lambda f: f.stat().st_mtime)

# Step 1: Collect all filenames
all_filenames = set()
with open(FILE) as f:
    for line in f:
        line = line.strip()
        if line and line.startswith(("CLOUD", "MODEL", "Avg", "Val Avg")):
            name, _ = line.rsplit(" ", 1)
            all_filenames.add(name)

all_filenames = sorted(all_filenames)

# Step 2: Initialize storage
example_losses = {name: [] for name in all_filenames}

# Step 3: Parse file
last_values = {name: np.nan for name in all_filenames}  # forward-fill

current_block = {}

with open(FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        if line.startswith("Base lr:"):
            # Commit block with forward-fill
            for name in all_filenames:
                if name in current_block:
                    last_values[name] = current_block[name]
                example_losses[name].append(last_values[name])

            current_block.clear()
            continue

        elif line.startswith(("CLOUD", "MODEL", "Avg", "Val Avg")):
            # Normal data line
            name, val = line.rsplit(" ", 1)
            current_block[name] = float(val)

        else:
            continue


hard_cases = []#["CLOUD0_C0.416063_0.191247_0.417250_L0.145794_0.458824_0.876484_A1.350192.png", "CLOUD0_C0.459367_0.419116_0.306151_L0.899402_0.187153_0.395031_A1.739639.png", "CLOUD0_C-0.429370_0.406946_-0.323467_L-0.573236_0.513591_0.638455_A1.529147.png", "CLOUD0_C-0.018360_0.430869_0.384959_L-0.845140_0.158514_0.510502_A0.730777.png", "CLOUD0_-0.629501_0.649585_-0.508684_L-0.099236_0.716329_-0.690670_A1.227424.png",
#               "CLOUD1_C0.058434_0.093086_0.109452_L-0.912422_-0.010247_-0.409123_A3.326212.png", "CLOUD1_C0.111274_0.031224_0.263431_L-0.827468_0.106621_0.551297_A0.649283.png", "CLOUD1_C-0.333920_-0.338320_0.054831_L0.194386_0.911442_-0.362611_A0.649845.png", "CLOUD1_C-0.106258_0.348746_0.043382_L0.549003_-0.817488_-0.174096_A1.073008.png", "CLOUD1_C0.274434_-0.302850_-0.118316_L0.025041_0.988429_0.149600_A1.426336.png", "CLOUD1_C0.219995_0.216839_0.378951_L-0.419090_-0.885986_0.198475_A1.374002.png",
#               "MODEL0_C0.048008_-0.214250_-0.304825_L0.376944_-0.758858_0.531081_A1.145751", "MODEL0_C0.153836_-0.298650_0.808849_L-0.243864_0.823592_-0.512081_A0.446764.png", "MODEL0_C0.677334_0.266187_-0.210111_L-0.900335_-0.433937_0.033107_A1.491007.png", "MODEL0_C0.688876_0.721127_0.368055_L-0.675847_-0.648882_-0.349546_A1.385542.png", "MODEL0_C0.694409_-4.856205_6.184069_L-0.024256_0.309454_-0.950605_A1.185063.png", "MODEL0_C7.840495_3.821144_-1.289281_L-0.770715_-0.582003_0.259365_A1.009125.png",
#               "MODEL1_C-0.045797_0.073078_-0.013943_L0.489629_-0.437446_0.754257_A2.637765.png", "MODEL1_C0.133278_-0.243804_0.002983_L-0.947979_-0.085091_-0.306748_A1.239971.png", "MODEL1_C2.899275_0.435656_2.909296_L-0.709297_0.422897_-0.563965_A3.541405.png", "MODEL1_C-2.944424_-2.651136_2.635145_L0.527350_0.844058_0.097311_A1.744594.png", "MODEL1_C-1.674745_0.886623_1.286366_L0.778243_-0.404415_-0.480402_A4.643472.png", "MODEL1_C-1.058841_0.054630_0.145098_L0.514398_0.508809_0.690296_A4.947528.png"]


#filtered_losses = dict(list(example_losses.items())[:20] + list(example_losses.items())[20::10])
# 1. Separate "Avg" lines from "Regular" lines
avg_items = [(n, l) for n, l in example_losses.items() if n.startswith(("Avg", "Val Avg"))]
regular_items = [(n, l) for n, l in example_losses.items() if not n.startswith(("Avg", "Val Avg")) and n not in hard_cases]
hard_items = [(n, l) for n, l in example_losses.items() if not n.startswith(("Avg", "Val Avg")) and n in hard_cases]

# 2. Sort only the regular lines by their last loss value
sorted_regular = sorted(
    regular_items,
    key=lambda item: item[1][-1],
    reverse=True
)

# 3. Apply the slice to the sorted regular lines
# (20 worst + every 10th of the remaining)
filtered_regular = sorted_regular[:20] + sorted_regular[20::10]

T = len(filtered_regular[0][1])
x = np.arange(T)

plt.figure(figsize=(14, 8))
for name, losses in filtered_regular:
    plt.plot(x, losses, alpha=0.5)

    plt.text(
        x[-1] + 0.5,
        losses[-1],
        name,
        fontsize=7,
        alpha=0.8,
        verticalalignment="center"
)

for name, losses in avg_items:
    plt.plot(x, losses, color= ("black" if name.startswith("Avg") else "blue"), linewidth=1.5, alpha=1.0)

    plt.text(
        x[-1] + 0.5,
        losses[-1],
        name,
        fontsize=9,
        alpha=0.8,
        verticalalignment="center",
        weight = "bold",
        color="red"
    )

for name, losses in hard_items:
    plt.plot(x, losses, color="brown", linewidth=1.5, alpha=1.0)

    plt.text(
        x[-1] + 0.5,
        losses[-1],
        name,
        fontsize=9,
        alpha=0.8,
        verticalalignment="center",
        weight="bold",
        color="blue"
    )

#plt.plot(x, avg_losses, color="black", linewidth=1.5)
#plt.text(x[-1] + 0.5, avg_losses[-1], "Avg / 30", fontsize=9, weight="bold")

plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Per-example loss progression")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()
