# --------------------- IMPORTS --------------------- #

# Metadata Reading
import pandas as PD

# TFRecord Conversion
import tensorflow as TF
from PIL import Image
from io import BytesIO
import os

# -------------------- CONSTANTS -------------------- #

# Default Spectrogram Size (To prevent unwanted resizing operations).
DEFAULT_SIZE = (2480, 1232)

# Training & Testing Metadata
TRAIN_metadata = PD.read_csv('./TRAIN_metadata.csv')
TEST_metadata = PD.read_csv('./TEST_metadata.csv')

# Converting the categorical string labels to integer labels.
TRAIN_metadata['Genre'] = TRAIN_metadata['Genre'].astype('category').cat.codes
TEST_metadata['Genre'] = TEST_metadata['Genre'].astype('category').cat.codes

# --------------------- FUNCTION --------------------- #

def worker(queue):
    while True:
        # Attempts to retrieve job information. Quits if no job is available.
        try:
            spectrogram_dir, record, size, train = queue.get()
        
            # Determining which metadata file to use & record name.
            metadata = TRAIN_metadata if train else TEST_metadata
            file_addon = 'train' if train else 'test'
            filename = f'{record}_{size[0]}x{size[1]}_{file_addon}.tfr'

            # Creating the record.
            with TF.io.TFRecordWriter(filename) as writer:
                for index, element in metadata.iterrows():
                    # Opening the image with 3 channels.
                    image = Image.open(os.path.join(spectrogram_dir, element['Pathname'])).convert('RGB')

                    # Resizes the image if needed.
                    if size != DEFAULT_SIZE:
                        image = image.resize(size)

                    # Creating a byte representation of the image.
                    image_bytes = BytesIO()
                    image.save(image_bytes, format = 'PNG')

                    # Storing the feature in the TFRecord file.
                    feature = TF.train.Example(features = TF.train.Features(feature = {
                        'image': TF.train.Feature(bytes_list = TF.train.BytesList(value = [image_bytes.getvalue()])),
                        'label': TF.train.Feature(int64_list = TF.train.Int64List(value = [element['Genre']]))
                    }))
                    writer.write(feature.SerializeToString())
                
        except Exception as E: return