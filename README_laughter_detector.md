# laughter-detection


This library contains code and models to segment  regions of laughter from an audio file. The models/ folder contains models trained on the [Switchboard](https://catalog.ldc.upenn.edu/ldc97s62) data set.

Please cite the following paper if you use this software for research:

Kimiko Ryokai, Elena Durán López, Noura Howell, Jon Gillick, and David Bamman (2018), "Capturing, Representing, and Interacting with Laughter," CHI 2018

## Installation

```sh
git clone https://github.com/jrgillick/laughter-detection.git
cd laughter-detection
pip install -r requirements.txt
```

## Usage
- To train a new model on Switchboard, see the __main__ function in `compute_features.py` and `train_model.py`
- To run the laugh detector from the command line, use `python segment_laughter.py <input_audio_path> <stored_model_path> <output_folder> <threshold>(optional) <min_length>(optional) <save_to_textgrid>(optional)`
- e.g. ` 1`

  #### Parameters
  - The threshold parameter adjusts the minimum probability threshold for classifying a frame as laughter. The default is 0.5, but you can  experiment with settings between 0 and 1 to see what works best for your data. Lower threshold values may give more false positives but may also recover a higher percentage of laughs from your file.

  - The min_length parameter sets the minimum length in seconds that a laugh needs to be in order to be identified. The default value is 0.2.

  - If set to True, the save_to_textgrid parameter toggles writing the identified laughter intervals to a [Praat](http://www.fon.hum.uva.nl/praat/) TextGrid annotation file rather than saving them in separate WAV files. The default value is False.


  #### Output
  - The segmenter prints out a list of time segments in seconds of the form (start, end) and saves them as a list of wav files into a folder at `<output_folder>`
  
## Dependencies
- [Librosa](http://librosa.github.io/librosa/) - tested with version 0.5.0
- [Keras](https://keras.io/) - tested with version 2.0.0
