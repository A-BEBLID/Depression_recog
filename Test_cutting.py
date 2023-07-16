import os, re
import wave
import pydub
import numpy as np
import matplotlib as plt
from pydub import AudioSegment
from pydub.utils import make_chunks

size = 10000
path = './Nd/'
files = os.listdir(path)
files = [path + f for f in files if f.endswith('.wav')]
cuts_path = './Nd_cut/'

def cut_waves():
    for i in range(len(files)):
        FileName = files[i]

        id = FileName[5:9]
        print(id)
        if FileName:
            wave = AudioSegment.from_file(FileName, "wav")  # 打开wav文件
            chunks = make_chunks(wave, size)  # 将文件切割为10s一块
            for i, chunk in enumerate(chunks):
                chunk_name = id + str(i)+'.wav'  # 也可以自定义名字
                print(chunk_name)
                chunk.export('./Nd_cut/'+ chunk_name, format="wav")  # 新建的保存文件夹

if __name__ == '__main__':
    cut_waves()