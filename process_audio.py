#!/usr/bin/env python3
"""
Extract acoustic features using egemaps (OpenSmile)

Copyright 2025, J.S. Gómez-Cañón
Licensed under GNU AFFERO GENERAL PUBLIC LICENSE
"""


import glob
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

import opensmile
import pdb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# smile = opensmile.Smile(
#     feature_set=opensmile.FeatureSet.eGeMAPSv02,
#     feature_level=opensmile.FeatureLevel.Functionals,
# )
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

def process_egemaps(fn, out_fn):
    dir_path = os.path.dirname(out_fn)
    os.makedirs(dir_path, exist_ok=True)

    this_feat = smile.process_file(fn)

    this_feat.to_csv(out_fn)


if __name__ == '__main__':
    base_path = os.path.abspath('.')
    path_audio = os.path.join(base_path, 'audio', 'recordings')
    path_tegg = os.path.join(base_path, 'audio', 'tegg')

    all_reco = glob.glob(os.path.join(path_audio, '*.wav'))
    all_tegg = glob.glob(os.path.join(path_tegg, '*.wav'))

    for f in tqdm(all_reco, f'Processing egemaps features recordings...'):
        out_f = f.replace('/audio/', '/feats/').replace('.wav', '.csv')
        if os.path.exists(out_f):
            continue
        process_egemaps(f, out_f)

    for f in tqdm(all_tegg, f'Processing egemaps features tegg...'):
        out_f = f.replace('/audio/', '/feats/').replace('.wav', '.csv')
        if os.path.exists(out_f):
            continue
        process_egemaps(f, out_f)
