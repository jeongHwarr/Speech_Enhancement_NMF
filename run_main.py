"""
main program (Source separation, speech enhancement).

@author: Jeonghwa Yoo
"""

import os
import argparse
from work_module import make_training_audio, make_test_audio, run_nmf

WORKSPACE = os.getcwd() # current workspace 
TR_SPEECH_PATH =  os.path.join(WORKSPACE,"dataset/train/speech") # directory path containing speech files for training. (training/speech)
TR_NOISE_PATH = os.path.join(WORKSPACE,"dataset/train/noise") #directory path containing noise files for training.(training/noise)
TE_SPEECH_PATH = os.path.join(WORKSPACE,"dataset/test/speech") # directory path containing speech files for test. (test/speech)
TE_NOISE_PATH = os.path.join(WORKSPACE,"dataset/train/noise") # directory path containing noise files for test. (test/noise)
OUTPUT_PATH = os.path.join(WORKSPACE,"output") # path for output directory

def get_args():
    parser = argparse.ArgumentParser(description="NMF based source separation.")
    
    #-----------------Algorithm----------------- 
    parser.add_argument('--nmf_mode', default='NMF_e', choices=['NMF', 'NMF_g', 'NMF_e'], type=str,
                         help="Choose a NMF algorithm. [NMF: standard, NMF_g: NMF using gamma distribution, NMF_e: NMF using exponential distribution")  
    #-----------------Data preprocessing----------------- 
    parser.add_argument('-sr', '--sampling_rate', default=16000, type=int,
                     help="target sampling rate")
    parser.add_argument('--fft', default=512, type=int,
                         help="FFT size")
    parser.add_argument('--window', default="hamming", type=str,
                         help="type of window (hamming, boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann, kaiser..)")
    parser.add_argument('--win_size', default=256, type=int,
                         help="window size")
    parser.add_argument('--overlap', default=192, type=int,
                     help="overlap size of spectrogram")
    #----------------- NMF parameters ----------------- 
    parser.add_argument('-i_train', '--max_iter_train', default=100, type=int,
                         help="max number of iterations ")
    parser.add_argument('-i_test', '--max_iter_test', default=30, type=int,
                         help="max number of iterations ") 
    parser.add_argument('-b_speech', '--num_basis_speech', default=128, type=int,
                         help="number of basis of speech") 
    parser.add_argument('-b_noise', '--num_basis_noise', default=128, type=int,
                         help="number of basis of noise")     
    parser.add_argument('-e', '--epsilon', default=0.5, type=float,
                         help="threshold to check convergence.") 
    parser.add_argument('--penalty', default=0.005, type=float,
                         help="Penalty rate for the penalty term.") 
    #----------------- Additional parameters about separation ----------------- 
    parser.add_argument('--p', default=2, type=int,
                         help="power of wiener gain.")
    #----------------- User parameter ----------------- 
    parser.add_argument('-v', '--visualize', default=1, type=int, choices=[0,1],
                        help="If value is 1, visualization of result of enhancement") 
    parser.add_argument('--plot_save', default=1, type=int, choices=[0,1],
                        help="If value is 1, save plot result as image") 

    args = parser.parse_args() 
    return args

if __name__ == '__main__':
    args = get_args()
    
    DIRECTORY = {}   
    DIRECTORY['WORKSPACE'] = WORKSPACE
    DIRECTORY['TR_SPEECH_PATH'] = TR_SPEECH_PATH
    DIRECTORY['TR_NOISE_PATH'] = TR_NOISE_PATH
    DIRECTORY['TE_SPEECH_PATH'] = TE_SPEECH_PATH 
    DIRECTORY['TE_NOISE_PATH'] = TE_NOISE_PATH
    DIRECTORY['OUTPUT_PATH'] = OUTPUT_PATH
    
    make_training_audio.main(DIRECTORY, args) # to make training data
    make_test_audio.main(DIRECTORY, args) # to make test data (noisy audio)
    run_nmf.main(DIRECTORY, args) # run NMF alogorithm for speech enhancement



