"""

Run the NMF algorithm. (Train/Test)

@author: Jeonghwa Yoo
"""

import os
import scipy
import librosa
import numpy as np
from util.util import makedirs, read_audio, write_audio
from work_module.nmf.nmf import nmf_train, nmf_test
import matplotlib.pyplot as plt


def audio_to_spectrogram(audio_path, fs, window, fft_size, hop_size):
    audio_wav,_ = read_audio(audio_path, fs)
    stft_audio = librosa.stft(audio_wav, n_fft=fft_size, hop_length=hop_size, window=window)  
    return stft_audio
    
def spectrogram_to_audio(magnitude, phase, window, hop_size):
    complx = magnitude * np.exp(1j*phase)
    audio = librosa.istft(complx, hop_length=hop_size, window=window)
    return audio


def wienner_filtering(V_noisy, W_hat, H_hat, num_basis_speech, p):
    clean_W = W_hat[:,:num_basis_speech]
    noise_W = W_hat[:,num_basis_speech:]
    
    clean_H = H_hat[:num_basis_speech,:]
    noise_H = H_hat[num_basis_speech:,:]
    
    speech_hat = np.matmul(clean_W, clean_H)
    noise_hat = np.matmul(noise_W, noise_H)
    
    wiener_gain = np.power(speech_hat,p) / (np.power(speech_hat, p)+np.power(noise_hat,p))
    
    enhanced_V = wiener_gain*V_noisy
    
    return enhanced_V

    
    
def main(DIRECTORY, args):
    
    OUTPUT_PATH = DIRECTORY['OUTPUT_PATH']
    
    hop_size = int(args.win_size - args.overlap) # hop size to make spectrogram 
    
    print("[Start] trainning.. NMF algoritm.")
    
    
    #-------------------train------------------
    target_dir = os.path.join(OUTPUT_PATH,'merged_audio')
     
    speech_path = os.path.join(target_dir, "merged_train_speech_audio.wav")
    noise_path = os.path.join(target_dir, "merged_train_noise_audio.wav") 
    
    # get magnitude of spectrograms 
    V_speech = abs(audio_to_spectrogram(speech_path, args.sampling_rate, args.window, args.fft, hop_size))
    V_noise = abs(audio_to_spectrogram(noise_path, args.sampling_rate, args.window, args.fft, hop_size))

    # do NMF (V ~= WH)
    V = np.concatenate((V_speech, V_noise), axis=1)
    num_basis_train = args.num_basis_speech + args.num_basis_noise
    W_train, H_train = nmf_train(V, args.max_iter_train, args.epsilon, num_basis_train)
    print("[End] trainning.. NMF algoritm.")
    
    
    #-------------------test-------------------
    print("[Start] Test.. NMF algoritm (output: enhanced speech).")
    test_dir = os.path.join(OUTPUT_PATH,'test_noisy_audio') 
    noisy_names = [na for na in os.listdir(test_dir) if na.lower().endswith(".wav")]
        
    output_path = os.path.join(OUTPUT_PATH,'enhanced_audio',str(args.nmf_mode))
    makedirs(output_path)
    
    for noisy_na in noisy_names:
        print("%s"%(noisy_na))
        noisy_path = os.path.join(test_dir, noisy_na)
        
        stft_noisy = audio_to_spectrogram(noisy_path, args.sampling_rate, args.window, args.fft, hop_size) # spectrogram for noisy
        
        V_noisy = abs(stft_noisy) # magnitude spectrogram for noisy 
        H_noisy = nmf_test(V_noisy, W_train, H_train, args.max_iter_test, args.epsilon, penalty=args.penalty, algorithm=args.nmf_mode) # new encoding vector obtained through nmf algorithm
        
        enhanced_V = wienner_filtering(V_noisy, W_train, H_noisy, args.num_basis_speech, args.p) # magnitude spectrogram for enhanced 
        reconstructed_audio = spectrogram_to_audio(enhanced_V, np.angle(stft_noisy), args.window, hop_size) # reconstruct audio 
        
        # write enhanced audio
        out_audio_path = os.path.join(output_path,"enhanced_%s"%noisy_na)
        write_audio(out_audio_path, reconstructed_audio, args.sampling_rate)
        
        #-------------------plot-------------------      

        if args.visualize:
            
            # spectrogram for clean speech
            clean_speech_na = ("_").join(noisy_na.split("_")[0:-1])+os.path.splitext(noisy_na)[-1]
            clean_speech_path = os.path.join(DIRECTORY['TE_SPEECH_PATH'],clean_speech_na)
            stft_clean = audio_to_spectrogram(clean_speech_path, args.sampling_rate, args.window, args.fft, hop_size)
            
            # visualize 
            fig, axs = plt.subplots(3,1, sharex=False)
            axs[0].matshow(np.log10(np.abs(stft_noisy)**2), origin='lower', aspect='auto', cmap='jet')
            axs[1].matshow(np.log10(np.abs(stft_clean)**2), origin='lower', aspect='auto', cmap='jet')
            axs[2].matshow(np.log10(enhanced_V**2), origin='lower', aspect='auto', cmap='jet')
            axs[0].set_title("0 db mixture log spectrogram (%s)" % noisy_na)
            axs[1].set_title("Clean speech log spectrogram")
            axs[2].set_title("Enhanced speech log spectrogram")
            for j1 in range(3):
                axs[j1].xaxis.tick_bottom()
            plt.tight_layout()
            fig = plt.gcf()
            plt.show()
            
            if args.plot_save:
                
                plot_path = os.path.join(OUTPUT_PATH,"plot",str(args.nmf_mode))
                makedirs(plot_path)
                
                seg_result_path = os.path.join(plot_path,"%s.png"%(os.path.splitext(noisy_na)[0]))
                fig.savefig(seg_result_path, dpi=300)

            
    print("[End] Test.. NMF algoritm (output: enhanced speech).")
        


if __name__ == '__main__':
    main()


