import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from glob import glob
import random
import json
class Data_Mixer(nn.Module):
    def __init__(self,data_list, sample_rate=16000, segment=4.0, pad=True):
        super(Data_Mixer,self).__init__()
        self.sample_rate = sample_rate
        self.segment = segment
        self.data_list = data_list
        with open("/home/bjwoo/PycharmProjects/Speech_source_separation/src/data/sitec_mixer/mix.json", "r") as st_json:
                mix_json = json.load(st_json)
        self.activlev = dict(mix_json)
    def forward(self,spk_num=2):
        assert spk_num == 2
        rand_num = self.get_random_num(spk_num)

        wav_info = []
        len_info = []
        activlev_info = []
        for idx, num in enumerate(rand_num):
            y, sr = torchaudio.load(self.data_list[num])
            wav_name = self.data_list[num].split("/")[-2]+"/"+ self.data_list[num].split("/")[-1]
            activlev_info.append(self.activlev[wav_name])
            wav_info.append(y)
            len_info.append(y.size(-1))
        max_len = torch.max(torch.Tensor(len_info))
        if self.segment != None:
            data_len = int(self.segment * self.sample_rate)
            if max_len<data_len:
                max_len=data_len
                for idx, length in enumerate(len_info):
                    if length < max_len:
                        dif_len = int(abs(length - max_len))
                        wav_info[idx] = F.pad(wav_info[idx], (0, dif_len), 'constant', 0)
            else:
                for idx, length in enumerate(len_info):
                    start_crop = random.randint(0, max_len - data_len)
                    if length<max_len:
                        dif_len = int(abs(length-max_len))
                        wav_info[idx] = F.pad(wav_info[idx],(0,dif_len),'constant',0)
                    wav_info[idx] = wav_info[idx][:,start_crop:start_crop+data_len]
        rand_snr = random.uniform(0,2.5)
        wav_info = torch.stack(wav_info)
        for idx, wav in enumerate(wav_info):
            if idx%2 ==0:
                wav_info[idx] = wav_info[idx]/torch.sqrt(torch.as_tensor(activlev_info[idx])) * 10 ** (torch.as_tensor(rand_snr/20))
            else:
                wav_info[idx] = wav_info[idx]/torch.sqrt(torch.as_tensor(activlev_info[idx])) * 10 ** (torch.as_tensor(-rand_snr/20))
        mixture = torch.sum(wav_info,dim=0)
        gain = torch.mul(torch.max(torch.max(torch.abs(mixture)), torch.max(torch.abs(wav_info))),1.11)
        if gain<1.11:
            gain = torch.Tensor([1.11])

        return mixture/gain, wav_info/gain
    def get_random_num(self,spk_num):
        result = []
        for i in range(spk_num):
            get_randn = random.randint(0,len(self.data_list)-1)
            if i!=0:
                while abs(get_randn-result[-1])<30:
                    get_randn = random.randint(0, len(self.data_list)-1)
            result.append(get_randn)
        return result

if __name__ == '__main__':
    # x = torch.randn(size=(8, 7999, 256))
    # SKIM = SkiMSeparator(256,causal=True)
    import json
    with open("/home/bjwoo/PycharmProjects/Speech_source_separation/src/data/sitec_mixer/mix.json", "r") as st_json: \
        mix_json = json.load(st_json)
    data_path = "/home/bjwoo/PycharmProjects/data/sitec"
    data_list = glob(data_path+"/**/*.wav",recursive=True)
    x = torch.randn([6,64000])
    SkiM = Data_Mixer(data_list)
    a,b = SkiM(2)
    from tqdm import tqdm
    print(a.size())
    import torchaudio
    torchaudio.save("mixed.wav",a,16000)
    torchaudio.save("s1.wav", b[0], 16000)
    torchaudio.save("s2.wav", b[1], 16000)
