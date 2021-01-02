# Music Classification

## Using Spectrogram Images

When initially thinking about music classification, the first thing that came to mind was turning audio into images. Creating spectrograms using the **LibROSA** package does just that. The full process is as follows:

### Dataset

The first thing to do was find a relevant dataset that contained both music and labeled genres. I managed to find a few datasets, but the one that peaked my interest the most was the *Free Music Archive* dataset available here:

[FMA GitHub](https://github.com/mdeff/fma)

The music dataset used was *fma_medium*, which contains 25,000 tracks that are 30s long. *fma_metadata* and *utils<span>.</span>py* are also provided, which make it easy to access track metadata.

### Feature Visualization

**Relevant Files:**
* spectrogram_image/Feature Visualization.ipynb
* spectrogram_image/convert.py


*Feature Visualization<span>.</span>ipynb* provides an overview of the metadata contents, genre ratios, etc. It also provides examples of audio for each track, as well as visual examples of the tracks being converted to spectrograms. All spectrogram processing functions were created and stored in *convert<span>.</span>py*.

### Feature Extraction

**Relevant Files:**
* spectrogram_image/Feature Extraction.ipynb
* spectrogram_image/convert.py
* spectrogram_image/spectrogram_generation.py

*Feature Extraction<span>.</span>ipynb* uses the power of Python's **multiprocessing** package for parallelization, **LibROSA** for spectrogram generation, and **Matplotlib** for image conversion. It stores the spectrograms in a file and creates metadata files for use by **Tensorflow's** dataframe generator function.

### TFRecord Generation

**Relevant Files:**
* spectrogram_image/TFRecord Conversion.ipynb
* spectrogram_image/TFRecord_generation.py

Inital testing with the extracted spectrogram features showed that retrieving input was an immense bottleneck when training **Tensorflow** models. As a result, it became necessary to speed up retrieval, which was the reason for converting the the spectrograms into TFRecord files. 

### Training

**Relevant Files:**
* spectrogram_image/CNN.ipynb
* spectrogram_image/training.py
* spectrogram_image/spec_augment.py

CNN testing can be found in *CNN.ipynb*. Since general augmentation models can't be used by spectrograms, researchers at Google came up with a simple method called SpecAugment. The paper can be found here:

[SpecAugment Paper](https://arxiv.org/abs/1904.08779)

In essence, spectrogram augmentation involves time shifting data and covering up random segments of the input to prevent overfitting. 

 The model created for training was simplistic, but training found that training on 4 different spectrogram types reached only 55% accuracy. While this is better than the 6.25% baseline, a few problems came to light:

* After generating spectrogram data through LiBROSA, conversion to a spectrogram image is unwanted. This creates two unwanted channels in the dataset.

* The spectrogram sizes are immense, with the largest spectrogram having a shape of (2480, 1232, 3).

* The dataset is severly unbalanced and is most likely causing the model to only learn a few classifications.

This lead to redefining the data pipeline.

----

## Using Spectrogram Features

### Feature Extraction

**Relevant Files:**
* spectrogram_feature/Full Data Pipeline.ipynb
* spectrogram_feature/generate_records.py


The new redefined pipeline involves directly using the generated spectrogram parameters for training. A few modifications were made:

* The input data was restricted to being 10 seconds long. Spectrogram representations of 30 second-length audio clips require too many parameters and can have high variance.

* Genres that have less tracks had 3 samples extracted from them, resulting in 3x the number of samples. Genres that had a large number of tracks would only receive a single sample from each track.

* There are many different spectrogram representations of audio. Because no specific spectrogram outperformed the other during image testing, the *Constant-Q Transform* method was used. This is mainly due to the transform being designed specifically for accurately representing musical audio data.

When processing a spectrogram using Constant Q-Transform (a spectrogram processing method) with 84 bins, the resulting spectrogram dimensions are (480, 84). These set of input parameters would normally be used to create the spectrogram images, but it's better to directly feed them as input. In fact, this representation has 250x less parameters than the largest spectrogram. The full transition from raw audio data to TFRecords can be seen by analyzing the **Relevant Files** above.

### Training

**Relevant Files:**
* spectrogram_feature/Training.ipynb
* spectrogram_feature/training.py
* spectrogram_feature/spec_augment.py

Various models were defined during training. This involved CNNs, ResNets, RNNs, and a combination of these layer models. Unfortunately, the highest accuracy reached was still around 55%. Despite increasing model parameters and layer counts, the model wasn't able to even overfit on the training data. 

While my model implementation could definitely be improved, my results show how genre classification isn't discrete between audio tracks and can be a confusing concept. A few examples of confusion between genre classification that are present today are:

* Folk v.s. Country
* Jazz v.s. Blues
* What is considered Pop?

Maybe a neural network trained on manually discretized audio tracks can automatically redefine and classify tracks for artists in the future.