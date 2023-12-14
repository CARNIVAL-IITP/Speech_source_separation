import json
from glob import glob
import soundfile as sf
import os
from tqdm import tqdm
if __name__ == '__main__':

    data_path = '/home/bjwoo/PycharmProjects/data/asteroid_generate_sitec/2spk'

    training_type = ["train","test","valid"]
    channels = ["4mic","6mic","8mic"]
    prefixs = ['mix','s1','s2']
    for training in training_type:
        for channel in channels:
            wav_path = os.path.join(data_path,training,channel,"mix")
            wav_list = glob(wav_path+"/*.wav")
            mix_file = []
            s1_file = []
            s2_file = []
            print(training,channel)
            for mixed_wav in tqdm(wav_list):
                mixed , sr = sf.read(mixed_wav)
                s1_wav = mixed_wav.replace("mix","s1")
                s2_wav = mixed_wav.replace("mix","s2")
                
                mix_file.append((mixed_wav,len(mixed)))
                s1_file.append((s1_wav, len(mixed)))
                s2_file.append((s2_wav, len(mixed)))
                
            with open("./"+training+"/"+channel+"/"+"mix.json","w") as f:
                json.dump(mix_file, f,indent=4)
            with open("./"+training+"/"+channel+"/"+"s1.json","w") as f:
                json.dump(s1_file, f, indent=4)
            with open("./"+training+"/"+channel+"/"+"s2.json", "w") as f:
                json.dump(s2_file, f, indent=4)

