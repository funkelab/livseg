#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:47:39 2023

@author: ananyasista
"""
from pathlib import Path
from natsort import natsorted

c1 = []
c2 = []
c3 = []
c4 = []

path = Path(
    "/Volumes/public/Feliciano_Lab/cnt_liver3/lobule1/mito_dendra2_phall555_Lipitox_R_580_Ms_PMP70_647/")

filenames = [file.name for file in path.iterdir() if file.is_file()]

for name in natsorted(filenames):
    if name[-6:-4] == "00":
        c1.append(name)
    elif name[-6:-4] == "01":
        c2.append(name)
    elif name[-6:-4] == "02":
        c3.append(name)
    else:
        c4.append(name)
