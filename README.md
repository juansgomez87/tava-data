# TAVA dataset

### Installation 
This repository was created using Python 3.11.10. Install the required dependencies:
```
python3.11 -m venv .venv
pip install -r requirements.txt
```

### Usage
1. Extract acoustic features
```
python process_audio.py
```


2. Run prediction script to obtain classification and regression metrics. 
```
python classifier.py -clf [phase/playlist] -algo [egemaps/maest] -mean [y/n]
```


### Reference:
```
@inproceedings{GómezCañón2025WASPAA,
	title        = {{The Test of Auditory-Vocal Affect (TAVA) dataset}},
	author       = {Gómez-Cañón, Juan S. and Noufi, Camille and Berger, Jonathan and Parker, Karen J. and Bowling, Daniel},
	year         = 2025,
	booktitle    = {2025 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
	organization = {IEEE},
    address      = {Lake Tahoe, USA}
}
```

