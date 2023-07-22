import hydra
import torch
import transformers
import nltk
import accelerate
import bitsandbytes
import peft 
import xformers
import evaluate
import scipy
import numpy
import pandas


@hydra.main('../conf', 'conf.yaml', version_base='1.2')
def main(cfg):
    pass

if __name__=='__main__':
    main()
