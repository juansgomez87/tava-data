# TAVA dataset

### Installation 
This repository was created using Python 3.11.10. Install the required dependencies:
```
python3.11 -m venv .venv
pip install -r requirements.txt
```

### Usage
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
	author       = {Gómez-Cañón, Juan S. and Noufi, Camille and Berger, Jonathan and Parker, Karen J. and Bowling, Daniel},
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

