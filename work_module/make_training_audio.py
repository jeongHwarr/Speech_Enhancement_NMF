"""

This file creates the training audio (speech/noise).
The speech and noise to be used for training are merged and write them to 'OUTPUT_PATH'.

OUTPUT:
    OUTPUT_PATH\merged_audio\merged_train_speech_audio.wav
    OUTPUT_PATH\merged_audio\merged_train_noise_audio.wav    
    
@author: Jeonghwa Yoo

"""
import os
import glob
import numpy as np

from util.util import makedirs, read_audio, write_audio

def rms(input):
    rms = np.sqrt(np.mean(input**2))
    return rms

def equalizingRMS(source,target):
    target = rms(source)/rms(target)*target
    return target

    
def get_merged_audio(path_read_folder, audio_type, args):
    merged_audio = []   
    for filename in glob.glob(os.path.join(path_read_folder, '*.wav')):
        # wav, rate = read_wav(filename , sr=args.sampling_rate)
        wav, rate = read_audio(filename, target_fs=args.sampling_rate)
        if audio_type=='speech' : merged_audio.append(wav)
        elif audio_type=='noise': 
            end = int (args.sampling_rate * 25) 
            merged_audio.append(wav[0:end]) # just using first 25 seconds data
    merged_audio=np.hstack(merged_audio)
    return merged_audio,rate


    
def main(DIRECTORY, args):
    
    print("[Start] making audio for the training.")
    
    TR_SPEECH_PATH = DIRECTORY['TR_SPEECH_PATH']
    TR_NOISE_PATH = DIRECTORY['TR_NOISE_PATH']
    OUTPUT_PATH = DIRECTORY['OUTPUT_PATH']
    
    # set output path
    merged_output_path = os.path.join(OUTPUT_PATH,'merged_audio')
   
    # make output path
    makedirs(merged_output_path)
    
    # make merged speech audio using librosa
    speech_merged_audio,rate = get_merged_audio(TR_SPEECH_PATH, audio_type='speech', args=args)
    
    # make merged noise audio using librosa
    noise_merged_audio,rate = get_merged_audio(TR_NOISE_PATH, audio_type='noise', args=args )
   
    # make RMS of speech and noise equal
    noise_merged_audio = equalizingRMS(speech_merged_audio, noise_merged_audio)
    
    # write wav files
    write_audio(merged_output_path+'/merged_train_speech_audio.wav', speech_merged_audio, args.sampling_rate)
    write_audio(merged_output_path+'/merged_train_noise_audio.wav', noise_merged_audio, args.sampling_rate)
    

    print("[Finish] making audio for the training.")
    
    

if __name__ == '__main__':
    main()
