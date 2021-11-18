import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
from mpl_toolkits import mplot3d
import librosa
from glob import glob
import io
import os
import math
from tqdm import tqdm
import soundfile as sf
import sys
from multiprocessing import Pool
from functools import partial
import random
random.seed(1)
def get_random_wav(wav_list):
    num = np.random.randint(len(wav_list))
    wav = wav_list[num]
    spk_id = wav.split('/')[-2]
    file_name = wav.split('/')[-1]
    return [wav, spk_id ,file_name.split('.')[0]]

def zero_pad_wav(a,b):
    dif = abs(len(a)-len(b))
    if len(a)>len(b):
        b = np.pad(b,(0,dif))
    else:
        a = np.pad(a,(0,dif))
    return a,b

def room_simulate(num_mic,mic_array,room_type):
    room_list= {'star_3':[8.3, 3.4, 2.5],'room_819': [7.9, 7.0, 2.7], 'room_409':[7.0 , 4.2, 2.7] }
    room = room_list[room_type]
    dim_x, dim_y, dim_z = room[0],room[1],room[2]
    sr = 16000
    rt60 = 0.3
    e_absorption, max_order = pra.inverse_sabine(rt60, [dim_x, dim_y, dim_z])
    print(e_absorption,max_order)
    num_direction = 12
    mic_radius = 0.04 #0.03231 testing
    #mic_radius =  np.random.uniform(low=0.025,high=0.035)
    mic_x_radius = 0.0637  # ?�??
    mic_y_radius = 0.0484
    mic_lin = 0.04  # ?�형
    
    room = pra.ShoeBox(room,
                      fs = sr,
                      materials = pra.Material(e_absorption),
                      max_order = max_order)

    mic_center = np.array([dim_x/2, dim_y/2, 0.69])
    thetas = np.arange(num_mic)/num_mic * 2 * np.pi
    theta_source = np.arange(num_direction) / num_direction * 2 * np.pi
    if mic_array == 'circle':
        center_to_mic = np.stack([np.cos(thetas), np.sin(thetas), np.zeros_like(thetas)], 0) * mic_radius
    elif mic_array == 'ellipse':
        center_to_mic = np.stack([mic_x_radius*np.cos(thetas), mic_y_radius*np.sin(thetas), np.zeros_like(thetas)], 0) #?�?�형
    elif mic_array =='linear':
        linear = np.arange(num_mic)*mic_lin
        linear = linear - np.max(linear)/2
        center_to_mic = np.stack([linear,np.zeros_like(linear),np.zeros_like(linear)],0) #?�형
    mic_positions = mic_center[:, None] + center_to_mic
    room.add_microphone_array(mic_positions)
    far_field_distance = 1 #?�원 source ?�치?�에 ?�???�정: ?�기???�성 ?�일 ?�어준 ???�래 block???�행?�켜주면 채널??맞게 ?��??�이??가??    thetas = np.arange(num_direction) / num_direction * 2 * np.pi
    center_to_source = np.stack([np.cos(theta_source), np.sin(theta_source), np.zeros_like(theta_source)], -1) * far_field_distance
    source_positions = mic_center[None, :] + center_to_source

    return room,source_positions

def search(dirname,wav_list):
    try:
        count =0
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                wav_list = search(full_filename,wav_list)
            else:
                ext = os.path.splitext(full_filename)[-1]
                if ext == '.wav':
                    wav_list.append(full_filename)
        return wav_list
    except PermissionError:
        pass
def generate_mix_data(wav_list,output_path,num_mic,mic_array,room_type,sample_num,seed_num):
    print("random seed:" ,seed_num)
    np.random.seed(seed_num+6)
    for i in tqdm(range(sample_num)):
        one = get_random_wav(wav_list)
        two = get_random_wav(wav_list)
        wav_a, spk_a, file_a = one[0], one[1], one[2]
        wav_b, spk_b, file_b = two[0], two[1], two[2]
        if spk_a == spk_b:
            two = get_random_wav(wav_list)
            wav_b, spk_b, file_b = two[0], two[1], two[2]
            
        wav_a, sr = librosa.load(wav_a, sr=16000)
        wav_a, index_a = librosa.effects.trim(wav_a , top_db=30)
        wav_b, sr = librosa.load(wav_b, sr=16000)
        wav_b, index_b = librosa.effects.trim(wav_b , top_db=30)

        wav_a, wav_b = zero_pad_wav(wav_a, wav_b)
        for i in range(1):
            for j in range(1):
                a_num = np.random.randint(12)
                b_num = np.random.randint(12)
                room, source_positions = room_simulate(num_mic=num_mic,mic_array=mic_array,room_type=room_type)
                room.add_source(source_positions[a_num], signal=wav_a)
                room.add_source(source_positions[b_num], signal=wav_b)
                room.simulate()
                # name = spk_a + '_' + file_a + '_' + str(a_num) + '_' + spk_b + '_' + file_b + '_'+str(
                #     b_num) + '_' + mic_array + '.wav'
                #name = "sitec_sample.wav"
                name = room_type +'_'+ str(num_mic)+'mic-'+ mic_array+'_'+str(len(wav_a)%100)+'.wav'
                room.mic_array.to_wav(
                    output_path  + 'mix/' + name,
                    norm=True,
                    bitdepth=np.int16,
                )
                sf.write(output_path + 's1/' + name, wav_a, sr)
                sf.write(output_path + 's2/' + name, wav_b, sr)
                del (room)
                del (source_positions)

def main():
    mode = "t"
    sample_num = 3 # training -> 270000 wav samples / test -> 30000 wav samples
    repeat = np.arange(1)
    data_path = '/home/bjwoo/PycharmProjects/data/sitec/SiTEC_Dict01_reading_sentence/'
    #output_path = '/home/bjwoo/PycharmProjects/data/multi_channel_sitec/'
    output_path = '/home/bjwoo/PycharmProjects/data/test_sample/8mic/'
    wav_list = []
    wav_list = search(data_path,wav_list)
    train_list= []
    test_list = []
    for i in wav_list:
        if i.split('/')[-3] == "set181-200":
            test_list.append(i)
        else:
            train_list.append(i)
            
    num_mic = 8
    mic_array = 'circle'
    room_type = 'room_409' #'star_3':[8.3, 3.4, 2.5],'room_819': [7.9, 7.0, 2.7], 'room_409':[7.0 , 4.2, 2.7]

    divided_list = train_list
    if mode == 'train':
        divided_list = train_list
        output_path += 'train/'
    elif mode == 'test':
        divided_list = test_list
        output_path +='test/'

    print(mode)
    print(output_path)
    pool = Pool(processes=6)
    func = partial(generate_mix_data, divided_list,output_path,num_mic,mic_array,room_type,sample_num)
    pool.map(func,repeat)
    pool.close()
    pool.join()
if __name__ =="__main__":
    main()
