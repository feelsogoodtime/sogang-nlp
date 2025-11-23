<div align="center">
  <div>&nbsp;</div>
  <img src="logo.png" width="300"/> <br>
  <a href="https://trendshift.io/repositories/8133" target="_blank"><img src="https://trendshift.io/api/badge/repositories/8133" alt="myshell-ai%2FMeloTTS | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

##
conda create -n melotts python=3.11
conda activate melotts

python app.py --host 0.0.0.0
http://192.168.12.190:7860/


$env:HF_HUB_OFFLINE=1
python -m melo.app --host 0.0.0.0
python app.py --host 0.0.0.0
Remove-Item Env:\HF_HUB_OFFLINE
tensorboard --logdir=melo/logs/TEST --host 0.0.0.0
http://127.0.0.1:6006


python preprocess_text.py --metadata data/KR/metadata.list
전처리하면 config.json 생성됨

pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python preprocess_text.py --metadata data/KR/metadata.list
python preprocess_text.py --metadata data/KR/metadata_copy.list --val-per-spk 100 --max-val-total 400
화자 4명 val데이터 100개 

전처리는 data/KR/metadata.list, melo/configs/config.json 파일 참조해서
data/KR/config.json, data/KR/metadata.list.cleaned, data/KR/train.list, data/KR/val.list 생김
훈련돌리면 logs/KR/config.json 생김 (복사뜸)



$env:USE_LIBUV=0; $env:MASTER_ADDR="localhost"; $env:MASTER_PORT="10902"; $env:WORLD_SIZE="1"; $env:RANK="0"; $env:LOCAL_RANK="0"; python train.py --c data/KR/config.json --model KR --pretrain_G logs/KR/G_KR.pth --pretrain_D logs/KR/D.pth --pretrain_dur logs/KR/DUR.pth
$env:USE_LIBUV=0; $env:MASTER_ADDR="localhost"; $env:MASTER_PORT="10902"; $env:WORLD_SIZE="1"; $env:RANK="0"; $env:LOCAL_RANK="0"; python train.py --c data/KR/config.json --model KR --pretrain_G logs/KR/G_KR.pth --pretrain_D logs/KR/D.pth --pretrain_dur logs/KR/DUR.pth
$env:USE_LIBUV=0; $env:MASTER_ADDR="localhost"; $env:MASTER_PORT="10902"; $env:WORLD_SIZE="1"; $env:RANK="0"; $env:LOCAL_RANK="0"; python train.py --c data/KR/config.json --model KR --pretrain_G logs/KR/G_803000.pth --pretrain_D logs/KR/D_803000.pth --pretrain_dur logs/KR/DUR_803000.pth


음성 합성 시 최초 실행이 오래 걸리는 이유는 다음과 같습니다:
모델 로딩:
kr_default_model과 kr_trained_model 두 개의 모델을 동시에 로드합니다
각 모델은 약 1GB 정도의 크기를 가집니다
모델 파일을 디스크에서 메모리로 로드하는 과정이 필요합니다
BERT 모델 초기화:
kykim/bert-kor-base 모델을 처음 로드할 때 다운로드가 필요합니다
이는 한 번만 다운로드되고 이후에는 캐시된 버전을 사용합니다
CUDA 초기화:
GPU를 사용하는 경우 CUDA 컨텍스트 초기화가 필요합니다
이는 첫 실행 시에만 발생합니다
이러한 초기화 과정은 한 번만 발생하고, 이후 음성 합성은 훨씬 빠르게 실행됩니다. 이는 정상적인 동작이며, 대부분의 딥러닝 모델에서 공통적으로 발생하는 현상입니다.

1. preprocess_text.py 에서 configs/config.json 파일 참조해서 새롭게 생성함 
2. tensorboard --logdir=melo/logs/KR/eval

파인튜닝

# 2. 최신 체크포인트로 재시작
python train.py --c data/KR/config.json --model KR \
    --pretrain_G logs/KR/G_105000.pth \
    --pretrain_D logs/KR/D_105000.pth \
    --pretrain_dur logs/KR/DUR_105000.pth

python melo/preprocess_audio_quality.py -i data/KR/wavs/sample.wav -o data/KR/wavs_enhanced/sample.wav --all




## Introduction
MeloTTS is a **high-quality multi-lingual** text-to-speech library by [MIT](https://www.mit.edu/) and [MyShell.ai](https://myshell.ai). Supported languages include:

| Language | Example |
| --- | --- |
| English (American)    | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-US/speed_1.0/sent_000.wav) |
| English (British)     | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-BR/speed_1.0/sent_000.wav) |
| English (Indian)      | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN_INDIA/speed_1.0/sent_000.wav) |
| English (Australian)  | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-AU/speed_1.0/sent_000.wav) |
| English (Default)     | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-Default/speed_1.0/sent_000.wav) |
| Spanish               | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/es/ES/speed_1.0/sent_000.wav) |
| French                | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/fr/FR/speed_1.0/sent_000.wav) |
| Chinese (mix EN)      | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/zh/ZH/speed_1.0/sent_008.wav) |
| Japanese              | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/jp/JP/speed_1.0/sent_000.wav) |
| Korean                | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/kr/KR/speed_1.0/sent_000.wav) |

Some other features include:
- The Chinese speaker supports `mixed Chinese and English`.
- Fast enough for `CPU real-time inference`.

## Usage
- [Use without Installation](docs/quick_use.md)
- [Install and Use Locally](docs/install.md)
- [Training on Custom Dataset](docs/training.md)

The Python API and model cards can be found in [this repo](https://github.com/myshell-ai/MeloTTS/blob/main/docs/install.md#python-api) or on [HuggingFace](https://huggingface.co/myshell-ai).

**Contributing**

If you find this work useful, please consider contributing to this repo.

- Many thanks to [@fakerybakery](https://github.com/fakerybakery) for adding the Web UI and CLI part.

## Authors

- [Wenliang Zhao](https://wl-zhao.github.io) at Tsinghua University
- [Xumin Yu](https://yuxumin.github.io) at Tsinghua University
- [Zengyi Qin](https://www.qinzy.tech) (project lead) at MIT and MyShell

**Citation**
```
@software{zhao2024melo,
  author={Zhao, Wenliang and Yu, Xumin and Qin, Zengyi},
  title = {MeloTTS: High-quality Multi-lingual Multi-accent Text-to-Speech},
  url = {https://github.com/myshell-ai/MeloTTS},
  year = {2023}
}
```

## License

This library is under MIT License, which means it is free for both commercial and non-commercial use.

## Acknowledgements

This implementation is based on [TTS](https://github.com/coqui-ai/TTS), [VITS](https://github.com/jaywalnut310/vits), [VITS2](https://github.com/daniilrobnikov/vits2) and [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2). We appreciate their awesome work.
