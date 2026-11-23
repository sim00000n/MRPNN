import faulthandler
import sys

from sympy import count_roots

import defines

faulthandler.enable()

if len(sys.argv) > 1:
    if sys.argv[1] == "inference_only":
        defines.TRAINING = False
        defines.DEBUG_IMG = False
    elif sys.argv[1] == "debug_img":
        defines.TRAINING = False
        defines.DEBUG_IMG = True

import torch
from pathlib import Path
import matplotlib.pyplot as plt
import mrpnn_helpers
from radiance_predictor_exact import RadiancePredictorExact
import numpy as np
from PIL import Image
import time

import os
import re

import logging
from datetime import datetime
import threading
from itertools import islice

def fetch_ground_truth_data(root_dir, is_training_set, gt_set):
    """
    Scans the ground_truth subfolder and parses PNG filenames of the format:
        NAME_CX_Y_Z_LX_Y_Z_AX.png

    Returns:
        dict where:
        {
            NAME: [
                [filename, (camX, camY, camZ), (lightX, lightY, lightZ), alpha],
                ...
            ],
            ...
        }
    """

    count_in_set = 0

    # Regex to match the filename format
    pattern = re.compile(
        r"^(.*?)_C([-.\d]+)_([-.\d]+)_([-.\d]+)_L([-.\d]+)_([-.\d]+)_([-.\d]+)_A([-.\d]+)\.png$"
    )

    for f in os.listdir(root_dir):
        if f.lower().endswith(".png"):
            match = pattern.match(f)
            if not match:
                continue  # Skip files that don't match the expected pattern

            (
                name,
                cx, cy, cz,
                lx, ly, lz,
                a
            ) = match.groups()

            cam_pos = (float(cx), float(cy), float(cz))
            light_dir = (float(lx), float(ly), float(lz))
            alpha = float(a)

            entry = [f, cam_pos, light_dir, alpha, is_training_set]

            if name not in gt_set:
                gt_set[name] = []
            gt_set[name].append(entry)

            count_in_set += 1

    return gt_set, count_in_set


def random_three_vector():
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    phi = np.random.uniform(0,np.pi*2)
    costheta = np.random.uniform(-1,1)

    theta = np.arccos( costheta )
    x = np.sin( theta) * np.cos( phi )
    y = np.sin( theta) * np.sin( phi )
    z = np.cos( theta )
    return (x,y,z)

stop_requested = False

dropout_pattern = re.compile(
            r"^d ([\d]+)$"
        )

def input_listener():
    global stop_requested
    while True:
        try:
            cmd = input().strip().lower()
        except EOFError:
            return
        if cmd in ("q", "quit", "exit"):
            print("Graceful stop requested")
            stop_requested = True
            return
        match = dropout_pattern.match(cmd)
        if match:
            d_p = match.group(1)
            wrapper.set_dropout_val(float(d_p)/100.0)


threading.Thread(
    target=input_listener,
    daemon=True
).start()


Path("logs").mkdir(exist_ok=True)
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

if defines.TRAINING:
    handler = logging.FileHandler(f"logs/train_{run_id}.log", mode="w")
else:
    handler = logging.StreamHandler()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        handler
    ],
)

def handle_exception(exc_type, exc_value, exc_traceback):
    """Catches any exception that isn't caught by a try/except block"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    for handler in logging.root.handlers:
        handler.flush()

# Register the hook
sys.excepthook = handle_exception

device = torch.device("cuda")

if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.synchronize()

# 1) Build the predictor from weights
#weight_dict = torch.load("model_best.pt")
weights_np   = mrpnn_helpers.load_npy_as_dict("NNWeight.npy")
weight_dict  = mrpnn_helpers.build_weight_dict(weights_np)
predictor    = RadiancePredictorExact(weight_dict, 2 ** 16, device=device).eval()

# compile only the model
#predictor = torch.compile(predictor, dynamic=True, fullgraph=False, mode="max-autotune")

gt_set = {}

gt_set, count_in_set = fetch_ground_truth_data(defines.SET_PATH, True, gt_set)
if defines.TRAINING:
    gt_set, _ = fetch_ground_truth_data("val_set", False, gt_set)

    #if defines.FORCE_EVAL:
    #    count_in_set = _count_in_set

if defines.TRAINING and not defines.FORCE_EVAL:
    gt_set_full = {}
    gt_set_full, _ = fetch_ground_truth_data("ground_truth_extend", True, gt_set_full)
    for name, entries in gt_set_full.items():
        np.random.shuffle(entries)

# 2) Wrap with the C++ data fetcher
width, height = 512, 512  # or whatever your C++ extension uses
items_iterator = iter(gt_set.items())
first_name, _ = next(items_iterator)
todo_set = {}

def milestone_callback():
    global count_in_set, gt_set_full

    c = int(float(count_in_set) * 0.005)
    for name, entries in gt_set.items():
        new_entries = gt_set_full[name][:c]

        for e in new_entries:
            source = Path(f"ground_truth_extend\\{e[0]}")
            destination = Path(f"ground_truth\\{e[0]}")
            source.rename(destination)

        start_offset = len(gt_set[name])
        todo_set[name].extend(list(range(start_offset, start_offset + len(new_entries))))
        gt_set[name].extend(new_entries)
        gt_set_full[name] = gt_set_full[name][c:]
    print(f"Inserting {c*4} new examples")
    count_in_set += c*4
    return count_in_set

def extend_examples(name):
    global count_in_set, gt_set_full

    newly_insert_pt = len(gt_set[name])
    
    #return newly_insert_pt

    c = min(int(float(count_in_set) * 0.01), len(gt_set_full[name])) if name in gt_set_full else 0

    if c > 0:
        new_entries = gt_set_full[name][:c]

        for e in new_entries:
            source = Path(f"ground_truth_extend\\{e[0]}")
            destination = Path(f"ground_truth\\{e[0]}")
            source.rename(destination)


        gt_set[name].extend(new_entries)
        gt_set_full[name] = gt_set_full[name][c:]
        count_in_set += c

    return newly_insert_pt

wrapper = mrpnn_helpers.MRPNNWrapper(predictor, width, height, "..\\build\\TestCase\\" + first_name,  count_in_set, milestone_callback, device=device)

# 3) Define g and scatterrate (global for this frame)
g_value = 0.857
scatter = (1.0, 1.0, 1.0)   # example isotropic; use your actual triplet
alpha = 1.0

lastExampleCount = 0    # debug

hard_cases = {}
hard_cases_names = {}
hard_cases_names["CLOUD0"] = []#["CLOUD0_C0.416063_0.191247_0.417250_L0.145794_0.458824_0.876484_A1.350192.png", "CLOUD0_C0.459367_0.419116_0.306151_L0.899402_0.187153_0.395031_A1.739639.png", "CLOUD0_C-0.429370_0.406946_-0.323467_L-0.573236_0.513591_0.638455_A1.529147.png", "CLOUD0_C-0.018360_0.430869_0.384959_L-0.845140_0.158514_0.510502_A0.730777.png", "CLOUD0_-0.629501_0.649585_-0.508684_L-0.099236_0.716329_-0.690670_A1.227424.png"]
hard_cases_names["CLOUD1"] = []#["CLOUD1_C0.058434_0.093086_0.109452_L-0.912422_-0.010247_-0.409123_A3.326212.png", "CLOUD1_C0.111274_0.031224_0.263431_L-0.827468_0.106621_0.551297_A0.649283.png", "CLOUD1_C-0.333920_-0.338320_0.054831_L0.194386_0.911442_-0.362611_A0.649845.png", "CLOUD1_C-0.106258_0.348746_0.043382_L0.549003_-0.817488_-0.174096_A1.073008.png", "CLOUD1_C0.274434_-0.302850_-0.118316_L0.025041_0.988429_0.149600_A1.426336.png", "CLOUD1_C0.219995_0.216839_0.378951_L-0.419090_-0.885986_0.198475_A1.374002.png"]
hard_cases_names["MODEL0"] = []#["MODEL0_C0.150954_-0.130681_-0.050226_L-0.867706_-0.191197_0.458835_A0.460956.pngMODEL0_C0.150954_-0.130681_-0.050226_L-0.867706_-0.191197_0.458835_A0.460956.png", "MODEL0_C-0.022396_0.074058_-0.167301_L-0.061475_-0.368093_0.927755_A0.644374.png"]
hard_cases_names["MODEL1"] = []#["MODEL1_C-0.045797_0.073078_-0.013943_L0.489629_-0.437446_0.754257_A2.637765.png", "MODEL1_C0.133278_-0.243804_0.002983_L-0.947979_-0.085091_-0.306748_A1.239971.png", "MODEL1_C2.899275_0.435656_2.909296_L-0.709297_0.422897_-0.563965_A3.541405.png", "MODEL1_C-2.944424_-2.651136_2.635145_L0.527350_0.844058_0.097311_A1.744594.png", "MODEL1_C-1.674745_0.886623_1.286366_L0.778243_-0.404415_-0.480402_A4.643472.png", "MODEL1_C-1.058841_0.054630_0.145098_L0.514398_0.508809_0.690296_A4.947528.png"]

num_itr = {}
num_itr["CLOUD0"] = 10
num_itr["CLOUD1"] = 10
num_itr["MODEL0"] = 10
num_itr["MODEL1"] = 10

for name, entries in gt_set.items():
    hard_cases[name] = [entry for entry in entries if entry[0] in hard_cases_names[name]]
    entries[:] = [entry for entry in entries if entry[0] not in hard_cases_names[name]]
    todo_set[name] = list(range(len(entries)))
    #if defines.TRAINING:
    #    np.random.shuffle(todo_set[name])
    np.random.shuffle(todo_set[name])
    #todo_set[name] = todo_set[name][:500]

num_itr_per_vol = 10

curr = None

do_hard_case = False#True

set_indirection = list(range(len(gt_set)))
while True:
    np.random.shuffle(set_indirection)
    for i in set_indirection:
        name, entries = next(islice(gt_set.items(), i, None))
        wrapper.swap_data("..\\build\\TestCase\\" + name)
        for x in range(num_itr[name]) if (not defines.DEBUG_IMG or defines.TRAINING) else range(len(entries)):
            if(not todo_set[name]):
                if defines.TRAINING:
                    newly_insert_pt = extend_examples(name)
                    todo_set[name] = wrapper.batch_tick(entries, newly_insert_pt)
                else:
                    if defines.DEBUG_IMG:
                        continue
                    todo_set[name] = list(range(len(entries)))
                    #np.random.shuffle(todo_set[name])

            if defines.TRAINING:
                do_hard_case = False#len(hard_cases[name]) > 0 and np.random.uniform(0,1) < 0.3
                if do_hard_case:
                    curr = hard_cases[name][np.random.randint(0,len(hard_cases[name]))]
                else:
                    curr = entries[todo_set[name].pop()]

                path = defines.SET_PATH if curr[4] else "val_set"
                wrapper.train_step_one_image(curr[1], curr[2], curr[3], g_value, scatter, f"{path}\\{curr[0]}", curr[4])
            else:
                #currTime = time.perf_counter()
                #genNew = nextGen <= currTime

                curr = entries[todo_set[name].pop()]
                print("GT: " + curr[0])

                while True:
                    wrapper.inference(curr[1], curr[2], curr[3], g_value, scatter, f"{defines.SET_PATH}\\{curr[0]}")

                    #nextGen = currTime + 5.0

                    wrapper.tick_gui()

                    if wrapper.should_got_to_next_example() or defines.DEBUG_IMG:
                        break

            if (wrapper.should_stop_training() or stop_requested):
                #if (defines.TRAINING):
                    #mrpnn_helpers.export_state_dict_to_cuh(
                    #    pt_path="model_best.pt",
                    #    out_cuh_path="weights_generated.cuh",
                    #    expected_shapes=mrpnn_helpers.EXPECTED_SHAPES,  # strongly recommended
                    #)

                # plt.ioff()
                # plt.show()

                wrapper.save_checkpoint("checkpoint_last.pt")

                quit()

