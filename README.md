# Multi channel Speech separation

Carnival system을 구성하는 Speech source separation 모델입니다.
과학기술통신부 재원으로 정보통신기획평가원(IITP) 지원을 받아 수행한
"원격 다자간 영상회의에서의 음성 품질 고도화 기술개발" 과제 공개 코드입니다.
(2021.05~2024.12)

본 과제의 version 1과 version 2의 멀티 채널 음성 분리 모델은 [Cone of Silence](https://github.com/vivjay30/Cone-of-Silence) 기반으로 2개 이상의 마이크에 입력되는 음성 정보를 이용하여 화자의 위치를 찾고, 해당 위치의 목소리만 추출하는 형태의 speech separation입니다.
Cone of Silence 모델을 기반으로 하여 아래 부분들을 개선하였습니다.

Version 3은 새로 encoder - masknet - decoder 기반의 구조로 구성되어있습니다.

Done

* Speaker network의 화자 임베딩을 이용하여 target speaker separation 성능 개선
* Feature wise linear modulation을 통한 speaker embedding 본 네트워크에 conditioning
* Bottleneck Layer를 Conformer encoder 사용
* SI-SNR loss 추가
* Low delay speeech separation 알고리즘 추가

To do

* Streaming speech separation
* Speaker Loss 추가(코드정리)
* 3~4명의 입력에도 분리할 수 있도록 기능 추가
  
Requirements
-------------
python==3.7.10     
torch==1.6.0    
torchaudio==0.6.0        
hydra-core==1.0.3           
hydra-colorlog==1.0.0      
omegaconf==2.1.1         
librosa==0.7.1         
soundifle==0.10.3          
pystoi==0.3.3       
pesq==0.0.2         
numpy==1.19.4       

Preprocessing
-------------
Version1 & Version2 Dataset:

To generate the dataset run this command:

    python preprocess/generate_sitec_dataset.py ./SiTEC_DB_path ./output_path
    
you can change settings at preprocess/generate_sitec_dataset.py using argparse

Version3 & Version4 Dataset:

To generate the dataset run this command:

    python preprocess/iitp_rir.py

you can change settings in the main() of iitp_rir.py

Training
-------------
To train the model run this command:

    python main.py
    
1차년도 세팅은 conf/conf.yaml 의 model에서 version_1
2차년도 세팅은 conf/conf.yaml 의 model에서 version_2(speaker_model_checkpoint 경로도 수정)
3차년도 세팅은 conf/conf.yaml 의 model에서 version_3

Evaluation
-------------
To be added

Reference code:
-------------
* Cone of Silence: https://github.com/vivjay30/Cone-of-Silence
* ECAPA-TDNN : https://github.com/TaoRuijie/ECAPA-TDNN 
* Conformer: https://github.com/jaketae/conformer 
