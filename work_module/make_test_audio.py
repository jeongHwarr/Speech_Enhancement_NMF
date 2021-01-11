"""
This file creates the test audio (noisy, SNR: 0dB).

@author: Jeonghwa Yoo
"""

import os
import numpy as np

from util.util import makedirs, read_audio, write_audio

def rms(input):
    rms = np.sqrt(np.mean(input**2))
    return rms

def equalizingRMS(ref,target):
    target = rms(ref)/rms(target)*target
    return target


def addNoise(speech, noise, snr):
    if len(speech) > len(noise):
        speech = speech[0:len(noise)]
    else :
        noise = noise[0:len(speech)]
    noisy = speech + np.sqrt(np.sum(np.abs(speech)**2))/ np.sqrt(np.sum(np.abs(noise)**2) * np.power(10,snr*0.1)) * noise
    return noisy

    
def main(DIRECTORY, args):
    
    print("[Start] making audio for the test.")
    
    TE_SPEECH_PATH = DIRECTORY['TE_SPEECH_PATH']
    TE_NOISE_PATH = DIRECTORY['TE_NOISE_PATH']
    OUTPUT_PATH = DIRECTORY['OUTPUT_PATH']
    
    # set output path
    output_path = os.path.join(OUTPUT_PATH,'test_noisy_audio')
   
    # make output path
    makedirs(output_path)
    
    speech_names = [na for na in os.listdir(TE_SPEECH_PATH) if na.lower().endswith(".wav")]
    noise_names = [na for na in os.listdir(TE_NOISE_PATH) if na.lower().endswith(".wav")]
    
    for speech_na in speech_names:
        speech_path = os.path.join(TE_SPEECH_PATH, speech_na)
        speech,_ = read_audio(speech_path, target_fs=args.sampling_rate)
        
        for noise_na in noise_names:
            noise_path = os.path.join(TE_NOISE_PATH, noise_na)
            noise,_ = read_audio(noise_path, target_fs=args.sampling_rate)
            noise = equalizingRMS(ref=speech, target=noise) 
            noisy = addNoise(speech = speech, noise = noise, snr=0)
            noisy_na = os.path.join("%s_%s.wav" % 
                                    (os.path.splitext(speech_na)[0], os.path.splitext(noise_na)[0]))
            noisy_path = os.path.join(output_path, noisy_na)
            write_audio(noisy_path, noisy, args.sampling_rate)
    
    print("[Finish] making audio for the test.")
    
    

if __name__ == '__main__':
    main()