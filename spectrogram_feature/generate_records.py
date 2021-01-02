import librosa
import tensorflow as TF
import numpy as NP
import sys

DURATION = 10
DATA_DIM = (430, 84)

def worker(args):
    pathname, label, offset = args
    
    try:
        # Loading raw audio data.
        data, _ = librosa.load(
            pathname, 
            offset = offset, 
            duration = DURATION
        )

        # Constant-Q Transformation
        data = librosa.cqt(data)
        
        # Additional Preprocessing
        data = NP.abs(data)
        data = librosa.util.normalize(data)
        data = data.T
        data = data[:DATA_DIM[0]]

        # Ignoring data that doesn't meet shape requirements.
        if data.shape != DATA_DIM:
            raise Exception(f'WRONG DIMENSIONS {data.shape} AT {offset}: {pathname}') 

        # Flattening the data so it can be parsed into a TFRecord feature.
        data = data.flatten()

        # Creating the TFrecord feature.
        feature = TF.train.Example(features = TF.train.Features(feature = {
            'parameters': TF.train.Feature(float_list = TF.train.FloatList(value = data)),
            'label': TF.train.Feature(int64_list = TF.train.Int64List(value = [label]))
        }))
        
        return feature.SerializeToString()
        
    except Exception as E:
        return E
        
        
        
    