Meta-Transformer Black Box - Usage Guide

This script, meta_blackbox.py, wraps the vit_main (Meta-Transformer) model from the paper into a simple, easy-to-use "black box" classifier.

It is designed to be run from the root of the meta-transformer-amc repository.

How It Works

Initialization: When you create a MetaBlackBoxClassifier object, it automatically:

Loads the config.yaml and model_params.yaml files.

Overrides the config to ensure it loads the vit_main model.

Loads the pre-trained weights for vit_main from the checkpoint/learning/vit_main/ directory.

Initializes the ViT encoder.

Classification: The .classify_signal() method is different from a standard classifier:

It requires two arguments:

query_signal: A single (2, 1024) NumPy array (the signal you want to identify).

support_set_dict: A Python dictionary.

Keys: The class indices (e.g., 0, 2, 4 for 'OOK', '8ASK', 'QPSK').

Values: A list of example signals (as (2, 1024) NumPy arrays) for that class.

Process:

It first runs all signals in the support_set_dict through the encoder.

It calculates the average "prototype" vector for each class.

It runs the query_signal through the encoder.

It calculates the distance from the query signal to each class prototype.

It converts these distances into a probability list.

It returns the top guess, its confidence, and the probabilities for all classes that were in the support set.

How to Use

Make sure you have followed the main README.md to install all requirements and download the dataset and pre-trained models.

Run the script directly from your terminal:

python meta_blackbox.py


Example Output

When you run the script, it will automatically load a sample "episode" (a 5-shot support set and one query signal) from the dataset and classify it as a demonstration. You should see a much more confident and accurate result:

Initializing Meta-Transformer black box classifier...
Running on GPU: cuda:0
Classifier ready.

Loading demo episode...
Loaded demo episode for classes: ['OOK', '8ASK', 'QPSK', '16PSK', '16QAM']
Selected query signal with actual label: 16QAM (SNR=20dB)

--- Classification Result ---
Actual Class:    16QAM
Predicted Class: 16QAM
Confidence:      99.98%

--- Probabilities (for this episode) ---
16QAM     : 99.98%
QPSK      : 0.01%
OOK       : 0.00%
8ASK      : 0.00%
16PSK     : 0.00%


Integrating Into Your Own Code

import numpy as np
from meta_blackbox import MetaBlackBoxClassifier, get_demo_episode

# 1. Create the classifier (loads model)
classifier = MetaBlackBoxClassifier()

# 2. Get a query signal and a support set
# (Here we load a demo one, but you would provide your own)
query_signal, actual, support_set = get_demo_episode(classifier.config)

# 3. Classify
if query_signal is not None:
    guess, prob, all_probs = classifier.classify_signal(query_signal, support_set)
    print(f"The model predicts: {guess} (Confidence: {prob:.2f})")
