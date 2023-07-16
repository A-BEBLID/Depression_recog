import librosa
from librosa import feature
import matplotlib
from matplotlib import pyplot as plt
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

########################################################
path = './Nd_cut/'
files = os.listdir(path)
files = [path + f for f in files if f.endswith('.wav')]


##############w###########n##################wn#####WN'Code
def Extract_features():
        for i in range(len(files)):
                FileName = files[i]

                id = FileName[8:15]
                print(id)
                audio,sr = librosa.load(FileName, sr=48000, mono=True, offset=0.0, duration=None)
                audio = librosa.effects.preemphasis(audio)

                mel = librosa.feature.melspectrogram(y=audio, sr=sr)
                print(mel.shape)
                plt.figure(figsize=(10,4))
                librosa.display.specshow(librosa.power_to_db(mel,ref=np.max),y_axis='mel', x_axis='time')
                plt.colorbar(format='%+2.0f dB')
                plt.title('Mel spectrogram')
                plt.xlabel('Time')
                plt.ylabel('Hz')
                plt.savefig('./Nd_cut_diagrams/'+id+'_mel.png')
                plt.show()
                ##############w###########n##################wn#####WN'Code
                mfcc = librosa.feature.mfcc(y=audio,sr=sr, n_mfcc=13)
                print(mfcc.shape)
                plt.figure(figsize=(10,4))
                librosa.display.specshow(mfcc,x_axis='time', sr=sr)
                plt.colorbar(format='%+2.f')
                plt.title('MFCC')
                plt.ylabel('Coefficient')
                plt.xlabel('Time')
                plt.savefig('./Nd_cut_diagrams/'+id+'_mfcc.png')
                plt.show()
                ##############w###########n##################wn#####WN'Code
                chroma = librosa.feature.chroma_stft(y=audio,sr=sr,)
                print(chroma.shape)
                plt.figure(figsize=(10,4))
                librosa.display.specshow(chroma,y_axis='chroma',x_axis='time')
                plt.colorbar(format='%+2.f')
                plt.title('Chroma')
                plt.ylabel('Pitch class')
                plt.xlabel('Time')
                plt.savefig('./Nd_cut_diagrams/'+id+'_chroma.png')
                plt.show()
                ##############w###########n##################wn#####WN'Code
                zcr = librosa.feature.zero_crossing_rate(y=audio)
                print('ZCR\'s shape is: ', zcr.shape)
                print(zcr.shape[1])
                x_axis= np.arange(0, zcr.shape[1])
                y_axis=np.array(zcr[0])
                plt.plot(x_axis,y_axis, color='#4169E1', alpha=0.8, linewidth=1)
                plt.title('ZCR')
                plt.ylabel('Count')
                plt.xlabel('Time')
                plt.savefig('./Nd_cut_diagrams/'+id+'_zcr.png')
                plt.show()


if __name__ == '__main__':
        Extract_features()



