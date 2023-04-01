#!/usr/bin/env python3
"""
Module Docstring
"""

__author__ = "Your Name"
__version__ = "0.1.0"
__license__ = "MIT"




import torch
import random
from glob import glob
from omegaconf import OmegaConf
from src.silero.utils import (init_jit_model, 
                       split_into_batches,
                       read_batch,
                       prepare_model_input)
from IPython.display import display, Audio

import torch
import zipfile
import pydub
from glob import glob
import urllib

device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU

from silero import silero_stt, silero_tts, silero_te

# Load the STT model
model_stt, decoder_stt, utils_stt = silero_stt('en')

# Load the TTS model
#model_tts, utils_tts = silero_tts('ru')

# Load the TE model
#model_te, utils_te = silero_te('ru')

# download a single file in any format compatible with TorchAudio
file_path = './George_W._Bush_first_inaugural_address,_January_20,_2001.ogg'

input = prepare_model_input(read_batch([file_path]),
                            device=device)
# urllib.request.urlretrieve("https://opus-codec.org/static/examples/samples/speech_orig.wav", "speech_orig.wav")
# test_files = glob('speech_orig.wav')
# batches = split_into_batches(test_files, batch_size=10)
# input = prepare_model_input(read_batch(batches[0]),
#                             device=device)

output = model_stt(input)


def main():
  for example in output:
    print(decoder_stt(example.cpu()))

if __name__ == "__main__":
    main()
