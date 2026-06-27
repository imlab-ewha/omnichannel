## Installation 
```shell
$ python3.9 -m venv .venv_ex
$ . .venv_ex/bin/activate
$ pip install --upgrade pip

# 유틸 모듈 설치
$ pip install pandas
$ pip install emoji==0.6.0
$ pip install soynlp
$ pip install SQLAlchemy
$ pip install pymysql

# 병렬처리 모듈 설치
$ python -m pip install "dask[complete]"

# Pytorch for CUDA11.3 (NVIDIA GeForce RTX 3090)기반, 이전 gpu 버전 혹은 cpu 버전 설치는 pytorch 공식 홈페이지 참고
$ pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# 리뷰 분석 모델을 위한 모듈 설치
$ pip install sentencepiece
$ pip install transformers
$ pip install kiwipiepy
```

## Run
```shell
$ python sub.py
```