# TAVA dataset

In this paper we present the TAVA, a novel dataset for advancing research in Speech Emotion Recognition (SER) by disentangling paralinguistic and linguistic information in affective speech signals. 
The dataset includes 352 audio recordings of emotionally expressive English speech, each paired with a corresponding transformed electroglottographic (tEGG) version---a signal designed to preserve affective cues while systematically suppressing phonetic content. 
In addition, we provide over 120,000 crowd-sourced ratings of valence, arousal, and dominance for both the original and transformed signals. 
These ratings support fine-grained comparisons of affect perception across modalities. 
Building on prior work showing that tEGG signals can effectively isolate vocal affect, this dataset offers a unique resource for evaluating sensitivity to vocal affect in clinical populations with language or communication difficulties, as well as in studies aimed at dissociating linguistic and affective processing in individuals without these impairments.
By contributing a phoneme-reduced, affect-rich signal representation to the SER community, we aim to enable more robust modelling of vocal affect and broaden the applicability of SER systems to diverse user populations. 

### Installation 
This repository was created using Python 3.11.10. Install the required dependencies:
```
python3.11 -m venv .venv
pip install -r requirements.txt
```

### Usage
The speech, EGG and tEGG audio files are in `audio` directory. The egemaps features are in the `feats` directory.

1. Extract acoustic features:
```
python process_audio.py -audio [speech/tEGG] 
```

2. Run prediction script to obtain classification and regression metrics. 
```
python lld_predict.py 
```

3. Run statistics from dataset. 
```
python stats.py 
```


### Reference:
```
@inproceedings{GomezCanonn2025WASPAA,
	title        = {{The Test of Auditory-Vocal Affect (TAVA) dataset}},
	author       = {Gómez-Cañón, Juan Sebastián and Noufi, Camille and Berger, Jonathan and Parker, Karen J. and Bowling, Daniel},
	year         = 2025,
	booktitle    = {IEEE Workshop on Applications of Signal Processing to Audio and Acoustics},
	month        = oct,
	publisher    = {WASPAA},
	pages        = {},
	doi          = {},
	url          = {},
	address      = {Tahoe City, USA}
}
```

