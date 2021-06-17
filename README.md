# NYCU-2021Spring-Introduction_to_AI
NYCU 2021 Spring Introduction to Artificial Intelligence Final project

# Requirements:
Python packages that you need.

See requirements.txt

# File description
### Demo-classify-youtube.py: 
Use trained model to classify youtube video.

### Hyper_parameters.py: 
Hyperparameter settings.

### audio_augmentation.py: 
Augments dataset.

### feature_extraction.py: 
Extracts feature from augmented dataset.

### main.py: 
Train and test on GTZAN dataset.

# Usage
### Train and test on GTZAN dataset:
1. Download GTZAN dataset and place genre folders under dataset/gtzan
2. Augment data by running "$ python audio_augmentation.py"
3. Extracts feature by running "$ python feature_extraction"
4. Starts train by running "$ python main.py"

### Use trained model to classify youtube video:
1. Make sure you have "Trained_model_wonorm.pth" under the same directory.
2. Run "$ python Demo-classify-youtube.py"

**Note:** YouTube tends to block python crawler, if the process stucks, please enter ^C and restart the program.
