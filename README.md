# Multi channel Speech separation

Carnival system을 구성하는 Speech source separation 모델입니다.
과학기술통신부 재원으로 정보통신기획평가원(IITP) 지원을 받아 수행한
"원격 다자간 영상회의에서의 음성 품질 고도화 기술개발" 과제 공개 코드입니다.
(2021.05~2024.12)

본 멀티 채널 음성 분리 모델은 [Cone of Silence](https://github.com/vivjay30/Cone-of-Silence) 기반으로 2개 이상의 마이크에 입력되는 음성 정보를 이용하여 화자의 위치를 찾고, 해당 위치의 목소리만 추출하는 형태의 speech separation입니다. 본 실험은 SiTEC 한국어 음성 DB를 사용하여 진행되었습니다.

Cone of Silence 모델을 기반으로 하여 아래 부분들을 개선하였습니다.

Done

* Speaker network의 화자 임베딩을 이용하여 target speaker separation 성능 개선
* Feature wise linear modulation을 통한 speaker embedding 본 네트워크에 conditioning
* Bottleneck Layer를 Conformer encoder 사용
* SI-SNR loss 추가

To do

* Streaming speech separation(low delay)
* Multi-input Single-output 을 위한 beamformer 추가

Requirements
-------------

Preprocessing
-------------
To generate the dataset run this command:

    python preprocess/generate_sitec_dataset.py ./SiTEC_DB_path ./output_path
    
you can change settings at preprocess/generate_sitec_dataset.py using argparse

Training
-------------
To train the model run this command:

    python main.py
    
you can change settings at conf/conf.yaml

Evaluation
-------------
To be added
