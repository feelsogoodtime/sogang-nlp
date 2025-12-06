# WebUI by mrfakename <X @realmrfakename / HF @mrfakename>
# Demo also available on HF Spaces: https://huggingface.co/spaces/mrfakename/MeloTTS
import gradio as gr
import os
import torch
import io
import ssl
import re
import json
import numpy as np
import librosa
import soundfile as sf
import tempfile
import click
import nltk
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from melo.api import TTS
import time
from melo.text.english import text_normalize

# EfficientSpeech imports
# Temporarily commented out due to g2pk/mecab dependency issues
# import yaml
# import torch.nn.functional as F
# from es_model import EfficientSpeech
# from synthesize import get_lexicon_and_g2p, text2phoneme

def resource_path(relative_path):
    base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

# ===== 환경 설정 =====
ssl._create_default_https_context = ssl._create_unverified_context

# SSL 인증서 검증 비활성화
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'

# Hugging Face SSL 검증 비활성화
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

print("Make sure you've downloaded unidic (python -m unidic download) for this WebUI to work.")

# ===== NLTK 데이터 다운로드 =====
def download_nltk_data():
    """필요한 NLTK 데이터를 다운로드합니다."""
    required_data = [
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('tokenizers/punkt', 'punkt')
    ]
    
    for data_path, data_name in required_data:
        try:
            nltk.data.find(data_path)
        except LookupError:
            print(f"Downloading {data_name}...")
            nltk.download(data_name)

download_nltk_data()

# ===== 상수 정의 =====
DEVICE = 'auto'
DEFAULT_SPEED = 1.0
SILENCE_DURATION = 0.3  # 문장 사이 무음 길이 (초)
KOREAN_DIGITS = ['영', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구']

DEFAULT_TEXT = '최근 텍스트 음성 변환 분야가 급속도로 발전하고 있습니다.'

# 화자별 기본 텍스트
SPEAKER_DEFAULT_TEXTS = {
    '김민석': """저희 팀이 이 프로젝트를 시작한 가장 큰 이유는 매우 단순했습니다. 

프로젝트는 잘 만들었는데, 또는 기획안은 너무 좋은데, 발표가 너무 어려웠던 적은 없으신가요? 

프로젝트에서는 발표자 선정, 발표 피로감, 시간 제약 등 여러 문제가 생기는데 

‘이 발표 자체를 TTS로 대신할 수 있다면 어떨까?” 라는 문제의식에서 출발했습니다. 

그래서 발표의 부담을 줄이고, 누구나 자연스러운 목소리로 발표할 수 있는 한국어 TTS 발표 시스템을 만들어보기로 결정했습니다. 

## #4

이러한 계기를 바탕으로 두가지 핵심 목표를 세웠습니다.
첫째, 발표 환경에서 바로 사용 가능한 실시간 한국어 경량 TTS의 모델 구축과 음질 및 속도 개선

둘째, 팀원 음성을 학습한 다중 화자 발표 시스템의 가능성을 실제 데이터와 데모로 검증하는 것입니다.

이를 위해 MELO TTS를 기반으로 한국어 전처리 규칙을 적용하고, Whisper STT와 Zero-shot을 조합해 팀원별 7천 문장 이상을 확보해 실험했습니다.

## #5

한국어 TTS는 영어보다 훨씬 어렵습니다. 

 철자 발음 불일치, 받침과 연음, 외래어 발음처럼 음운 변동이 많고, 텍스트 만으로는 문장 억양이나 호흡 정보를 알 수 없습니다.

또한 숫자 - 날짜 단위 같은 표현이 너무 다양하고, 영문 약어가 섞일때 Prosody도 크게 흔들립니다.

마지막으로 TTS는 구조상 무거워서 모바일이나 CPU 환경에서 실시간 합성이 쉽지 않습니다. 이 문제들을 해결하기 위한 프로젝트를 진행했습니다.""",

    '김철희': """저희 팀은 한국어 경량 TTS의 성능을 높이기 위해 크게 네가지 방향으로 접근했습니다.  

첫째,MeloTTS 기반 Zero-shot 모델을 사용해 팀원 개개인의 목소리를 임베딩하고 , 둘째, Whisper 기반 STT를 활용해 녹음 데이터의 품질을 자동으로 검증했습니다. 셋째, 4문장에서 7000문장으로 데이터를 확장하여 학습 효과를 높였고, 마지막으로 실시간 데모를 위해 전체 파이프라인을 스트리밍 구조로 구성했습니다. 

전체 구조는 텍스트가 음소로 변환되고 Generator가 파형을 만들고, Discriminator가 이를 평가하는 End-to-End TTS 구조입니다.

또한 GAN방식을 활용하여 Generator 와 Discriminator 가 경쟁하여 학습하도록 하였고, 마지막으로 5가지 손실 함수를 통해 다양한 측면에서 최적화 할 수 있게 구성했습니다.

## #8

Melo TTS는 3가지 핵심 기술이 있습니다. 

모델 구조와 GAN 방식, Mel spectrogram 비교 입니다.

모델 구조는 Generator, Discriminator, Duration 모델의 세 부분으로 이루어져 있습니다.  Generator는 텍스트와 음소 정보를 입력받아 멜 스펙트로그램으로 생성하고, Discriminator는 생성된 음성을 판단하여 품질을 높이는 역할을 합니다. 한국어의 특징상 발음 길이와 호흡이 자연스러워야 하기 때문에 Duration 모델도 필수적으로 포함했습니다. Duration 구조가 안정되지 않으면 단어 길이가 짧게 끊기거나, 반대로 늘어지는 문제가 발생하기 때문입니다. 

## #9

또한 TTS는 GAN 방식으로 학습되는데 매 스텝마다 generator가 음성을 만들고, Discriminator가 평가한뒤 Generator가 다시 개선되는 구조입니다.

## #10

MELO TTS 모델의 학습 성능을 평가하기 위해, 생성된 음성과 실제 음성의 Mel 스펙트로그램을 비교해야 하며, 이를 통해 모델이 얼마나 자연스러운 음성을 생성했는지를 직관적으로 확인할 수 있습니다. 코드에서는 mel_spectrogram_torch 함수를 사용해 두 음성의 스펙트로그램을 시각화합니다.

## #11~13

한국어 TTS 모델을 성공적으로 학습시키는데 중요한 5가지 손실함수가 있습니다.

MEL LOSS는 음질과 음색을 결정하며 가장 큰 비중을 가집니다.

Duration Loss는 한국어 발음 길이를 학습시키는데 중요합니다.

KL Loss는 분포를 안정화시키고, 

Feature Matching Loss는 음성의 세부 특징까지 정교하게 학습하게 합니다.

그리고 Adversarial Loss는 GAN 구조에서 자연스러움을 높이는 역할을 합니다""",

    '이기쁨': """TTS 모델 훈련의 핵심 목표는 원래 화자의 목소리 특성을 유지하면서도 새로운 텍스트를 자연스럽게 읽어주기 위한 3가지의 핵심 요소가 필요합니다.

첫째, 화자 유사도 분석입니다. 만약 모델이 텍스트는 잘 읽지만 완전히 다른 사람의 목소리로 나온다면 실패한 모델일 것입니다. 그렇기에 훈련 후에도 같은 사람의 목소리로 들리는지, 화자 고유의 음색과 특징이 보존되는지 검증이 필요했습니다.

둘째, 음질 왜곡도 분석입니다. 제로샷 TTS원본과 훈련 후 합성음을 비교하여 동일한 음질을 재현하며, 자연스러운 목소리로, 노이즈나 왜곡은 없는지 검증이 필요합니다. 목표는 8 데시벨보다 낮게 나오면 원본과 동일한 품질이라고 기준을 잡았습니다.

마지막으로 운율 특성 분석입니다. 목소리는 화자의 특성을 살렸지만 감정이 없이 단조롭다면 불쾌한 골짜기처럼 청중의 피로도만 증가할 것 입니다. 이에 말하는 속도와 리듬이 자연스러운지, 억양과 감정은 적절히 표현되는지, 문장에 따른 강세와 쉼이 적절한지 확인하고자 하였습니다.

## #16

zero-shot 음성 분석을 통해 원본 음성와 zero-shot으로 생성한 음성의 차이를 보고자 하였습니다.
파형의 유형이 비슷하게 나타나나기는 하지만 두 음성간 차이를 확연하게 확인하기는 어려워서 이외의 추가적인 평가 지표를 활용해 정량적으로 분석 하고자 하였습니다.

## #17~19

학습 초기에 Loss 값은 약 2.45로 비교적 높은 편이었지만, 학습이 진행될수록 안정적으로 감소해 최종적으로 2.31 수준까지 떨어졌습니다. 56,000 스텝까지 안정적으로 감소하였습니다.

학습률 역시 거의 일정하게 유지되며 학습이 잘 된것을 볼 수 있습니다. 

멜 스펙트로그램 분석에서도 학습 효과가 뚜렷했습니다. 초기에는 저주파 영역만 흐릿하게 활성화되었지만, 중반부터 포먼트가 명확하게 형성되고, 후반에는 고주파와 저주파의 패턴이 균형 있게 나타나며 자연스러운 음성 구조를 학습한 것을 확인할 수 있었습니다.

또한 원본 음성과 Zero-shot 출력의 파형을 비교한 결과, 전체 에너지 분포와 발성 리듬이 상당히 비슷하게 나타났습니다. 물론 파형만으로는 음질을 완전히 비교하기 어렵기 때문에, 추가적으로 MFCC 기반 음색 유사도, STFT 주파수 차이, 그리고 STT 기반 발음 정확도(WER) 등을 함께 검토했습니다. 대부분의 지표에서 원본 대비 자연스러운 유사도를 얻었습니다. 

## #20~22

TTS 모델의 품질은 세가지 요소로 평가했습니다.

화자, 음질, 자연스럼입니다.

모델 A는 화자 유사도, 피치, 템포는 목표를 달성했지만 MCD가 높아 음질 개선이 필요했습니다. 이를 보완하기 위해 모델 A가 생성한 음성으로 다시 데이터셋을 구성하여 모델 B를 재학습한 결과, Loss 가 평균 76% 감소했고, MCD도 2.8dB 이상 개선되었습니다. 이를 통해 학습된 음성을 재학습함으로써 기존 모델 A보다 모델 B에서 높은 품질의 합성 음성을 만들수 있다는 것을 확인하였습니다.

## #23

화자 유사도 분석은 각 음성 파일에서 20개의 MFCC 계수를 추출한 뒤, 시간 축을 따라 평균을 내어 20차원의 화자 벡터인 vec1과 vec2를 생성합니다. 그 후  이 두 벡터 간의 코사인 유사도를 계산하여 두 화자의 목소리가 얼마나 유사한지 정량적으로 분석하였습니다.

## #24

MCD, 음질 왜곡도 분석의 경우 원본 음성과 합성된 음성의 MFCC 벡터 사이의 평균 유클리디안 거리를 측정합니다. 정확한 비교를 위해 먼저 DTW를 사용하여 두 음성의 벡터를 정렬하고, 최종적으로 이 거리를 계산하여 값이 낮을수록 왜곡이 적고 음질이 좋음을 의미합니다.

## #25

피치 추출은 리브로사 피치트랙 함수를 이용하며, 두 음성 간의 평균 피치 차이를 계산하여 운율의 높낮이 변화를 분석합니다. 또한, 템포 비율은 두 음성 파일의 길이 비율로 정의되어, 음성 속도의 상대적인 변화를 정량적으로 평가하였습니다.

## #26

실제 원본 음성 , zeroshot 음성, 모델 A/B가 생성한 음성을 비교한 결과, 

최종 모델인 B로 갈수록,  음질, 억양 발화 길이 등 전반적으로 더 자연스러운 합성을 보여주었습니다.

생성한 모델을 통하여 시연을 진행하도록 하겠습니다.""",

    '이도훈': """마지막으로 프로젝트 결과입니다.

저희는 화자당 4문장만으로 Zero-shot을 통해 총 28,000개의 학습 데이터를 생성했고, 모델을 600MB에서 200MB까지 경량화했습니다.

한국어 음운, 억양, Prosody 처리를 위한 전처리 규칙을 구축해 자연성을 높였고, RTF 0.5로 CPU에서도 실시간 합성이 가능한 수준까지 도달했습니다.

또한 zero shot 데이터를 학습하여 나온 합성 음성을 토대로 재학습하는 자기 개선형 학습구조를 구축하여 더 나은 품질의 데이터를 스스로 만들어내는 파이프라인을 구현했습니다.

이번 프로젝트를 통해 세 가지 중요한 인사이트를 얻었습니다.

**첫째, Zero-shot TTS도 한국어 전처리와 데이터 정제가 충분히 갖춰지면 실제 발표 수준의 품질을 확보할 수 있다는 점입니다.** 

**둘째, Whisper 기반의 STT 자동 필터링과 초기 모델의 학습 데이터 셋을 재학습시키는 것은 데이터 품질을 획기적으로 높여 TTS의 자연스러움을 향상 시키는 핵심 요소였습니다.** 

**셋째, 현대의 TTS 모델은 경량화만 잘 이루어진다면, 실시간 발표, 강의, 안내 시스템 등 다양한 분야에서 실사용이 가능하다는 가능성을 직접 확인했습니다.**   

물론 한국어의 발음 규칙, 억양, 코드스위칭 문제는 여전히 남아 있으며 특히 Prosody 안정화와 모바일 환경 최적화는 앞으로 개선해야 할 부분입니다.

그러나 이번 프로젝트를 통해 한국어 경량 TTS가 실제로 활용될 수 있는 충분한 잠재력을 확인할 수 있었습니다. 

이번 프로젝트를 수행하면서 zero shot 재합성을 몇번이나 반복하면 어디까지 좋아지고, 몇단계까지, 화자유지와 음질유지가 가능한지 궁금증이 생겼습니다. 이는 학습 시간과 리소스 문제로 추가적으로 더 진행해보지는 못했지만 향후 TTS모델의 누적 합성 안정성, 즉 합성 음성의 재사용 가능성을 탐구할 수 있는 흥미로운 방향이라는 점에서 의미 있는 시사점을 남겼습니다.

## #end

이상으로 발표를 마치겠습니다.

프로젝트 관련하여 궁금하신 부분이 있으시다면 말씀 부탁드립니다."""
}

# 영어 단어 -> 한글 발음 변환 딕셔너리
ENGLISH_TO_KOREAN = {
    'MELO': '멜로',
    'Melo': '멜로',
    'melo': '멜로',
    'MeloTTS': '멜로티티에스',
    'Whisper': '휘스퍼',
    'whisper': '휘스퍼',
    'WHISPER': '휘스퍼',
    'Whissper': '휘스퍼',
    'STT': '에스티티',
    'stt': '에스티티',
    'TTS': '티티에스',
    'tts': '티티에스',
    'Zero-shot': '제로샷',
    'zero-shot': '제로샷',
    'Zero-Shot': '제로샷',
    'ZERO-SHOT': '제로샷',
    'zeroshot': '제로샷',
    'Zeroshot': '제로샷',
    'zero shot': '제로샷',
    'Zero shot': '제로샷',
    'Zero Shot': '제로샷',
    'ZERO SHOT': '제로샷',
    'End-to-End': '엔드 투 엔드',
    'end-to-end': '엔드 투 엔드',
    'END-TO-END': '엔드 투 엔드',
    'Discriminator': '디스크리미네이터',
    'discriminator': '디스크리미네이터',
    'MEL LOSS': '멜 로스',
    'Mel Loss': '멜 로스',
    'mel loss': '멜 로스',
    'Mel loss': '멜 로스',
    'MEL': '멜',
    'LOSS': '로스',
    'Mel spectrogram': '멜 스펙트로그램',
    'mel spectrogram': '멜 스펙트로그램',
    'Mel Spectrogram': '멜 스펙트로그램',
    'Mel spectogram': '멜 스펙토그램',
    'mel spectogram': '멜 스펙토그램',
    'mel_spectrogram_torch': '멜 스펙트로그램 토치',
    'GAN': '쥐에이엔',
    'gan': '쥐에이엔',
    # 단위 및 기술 용어
    'dB': '데시벨',
    'DB': '데시벨',
    'db': '데시벨',
    'MB': '메가바이트',
    'mb': '메가바이트',
    'GB': '기가바이트',
    'gb': '기가바이트',
    'KB': '킬로바이트',
    'kb': '킬로바이트',
    # 변수명
    'vec1': '벡터 일',
    'vec2': '벡터 이',
    'Vec1': '벡터 일',
    'Vec2': '벡터 이',
    # 라이브러리/함수
    'librosa.piptrack': '리브로사 닷 핍트랙',
    'librosa.feature': '리브로사 닷 피처',
    # 문법 교정
    '것 입니다': '거십니다',
    '것입니다': '것입니다',
}

# 숫자+단위 -> 한글 변환 패턴
NUMBER_UNIT_PATTERNS = {
    '1문장': '한 문장',
    '2문장': '두 문장',
    '3문장': '세 문장',
    '4문장': '네 문장',
    '5문장': '다섯 문장',
    '6문장': '여섯 문장',
    '7문장': '일곱 문장',
    '8문장': '여덟 문장',
    '9문장': '아홉 문장',
    '10문장': '열 문장',
    '1가지': '한 가지',
    '2가지': '두 가지',
    '3가지': '세 가지',
    '4가지': '네 가지',
    '5가지': '다섯 가지',
    '6가지': '여섯 가지',
    '7가지': '일곱 가지',
    '8가지': '여덟 가지',
    '9가지': '아홉 가지',
    '10가지': '열 가지',
}

# ===== 모델 관리 클래스 =====
class ModelManager:
    """TTS 모델을 관리하는 클래스"""

    def __init__(self, model_name, config_file, ckpt_file):
        print(f"{model_name} 모델 로딩 중...")
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_name = model_name
        self.model = TTS(language='KR', device=DEVICE,
                        config_path=os.path.join(base_dir, 'model', config_file),
                        ckpt_path=os.path.join(base_dir, 'model', ckpt_file))
        print(f"{model_name} 모델 로딩 완료!")
        self.speaker_ids = None

    def get_model(self):
        """모델을 가져옵니다."""
        return self.model

    def get_speaker_ids(self):
        """스피커 ID를 가져옵니다."""
        if self.speaker_ids is None:
            if hasattr(self.model, 'hps') and hasattr(self.model.hps, 'data') and hasattr(self.model.hps.data, 'spk2id'):
                self.speaker_ids = self.model.hps.data.spk2id
            else:
                self.speaker_ids = {}
        return self.speaker_ids

# ===== 스피커 정보 관리 =====
class SpeakerInfoManager:
    """스피커 정보를 관리하는 클래스"""
    
    def __init__(self):
        self.cache = {}
        
    def get_speaker_info(self, speaker_name, language=None):
        """스피커의 상세 정보를 가져옵니다."""
        cache_key = f"{speaker_name}_{language}" if language else speaker_name

        if cache_key in self.cache:
            return self.cache[cache_key]

        result = {
            'name': speaker_name,
            'label': speaker_name,
            'size': 0
        }
        self.cache[cache_key] = result
        return result
    

# ===== 텍스트 처리 함수 =====
def preprocess_text(text):
    """텍스트를 전처리합니다."""
    # 영어 단어를 한글 발음으로 변환
    for eng, kor in ENGLISH_TO_KOREAN.items():
        text = text.replace(eng, kor)
    # 숫자+단위 패턴을 한글로 변환
    for pattern, replacement in NUMBER_UNIT_PATTERNS.items():
        text = text.replace(pattern, replacement)
    # 줄바꿈으로 분리된 숫자 처리
    text = re.sub(r'(\d+,)\s*\n\s*(\d+)', r'\1\2', text)
    # 숫자 내의 모든 쉼표를 제거
    text = re.sub(r'(\d{1,3}(?:,\d{3})+)', lambda m: m.group(0).replace(',', ''), text)
    # 소수점 숫자를 한글로 변환
    text = re.sub(r'\d+\.\d+', convert_decimal_to_korean, text)
    return text

def convert_decimal_to_korean(match):
    """소수점 숫자를 한글로 변환합니다."""
    number = match.group(0)
    parts = number.split('.')
    result = ''
    
    # 정수 부분
    for digit in parts[0]:
        result += KOREAN_DIGITS[int(digit)]
    
    result += '점'
    
    # 소수 부분
    for digit in parts[1]:
        result += KOREAN_DIGITS[int(digit)]
        
    return result

def parse_speaker_text(text, default_speaker):
    """텍스트에서 화자 태그를 파싱합니다."""
    pattern = r'\[([^\]]+)\](.*?)(?=\[|$)'
    matches = re.findall(pattern, text, re.DOTALL)

    if not matches:
        # 태그가 없는 경우 전체 텍스트를 기본 화자로 처리
        return [(default_speaker, text)]

    return [(speaker, sentence.strip()) for speaker, sentence in matches if sentence.strip()]

def parse_text_with_silence(text):
    """
    텍스트에서 # 패턴을 파싱하여 (텍스트, 무음시간) 튜플 리스트로 반환합니다.
    예: "안녕하세요#다음 문장" -> [("안녕하세요", 5), ("다음 문장", 0)]
    #, #1, #2, ... #100 모두 5초 무음을 삽입합니다.
    """
    # # 또는 #숫자 패턴으로 분리 (1~100까지)
    pattern = r'#(\d*)'
    parts = re.split(pattern, text)

    result = []
    i = 0
    while i < len(parts):
        text_part = parts[i].strip()
        silence_duration = 0

        # 다음 파트가 숫자(무음 시간)인지 확인
        if i + 1 < len(parts):
            silence_str = parts[i + 1]
            if silence_str == '':
                # # 만 있는 경우 3초
                silence_duration = 3
            else:
                # #숫자 형태인 경우 (1~100)도 모두 3초로 고정
                num = int(silence_str)
                if 1 <= num <= 100:
                    silence_duration = 3
                else:
                    # 100 초과면 그 숫자 초 사용
                    silence_duration = num
            i += 2
        else:
            i += 1

        if text_part:
            result.append((text_part, silence_duration))

    return result if result else [(text, 0)]

def get_silence_audio(duration_seconds, target_sample_rate):
    """
    지정된 시간(초)의 무음 오디오 배열을 반환합니다.
    5초 무음 파일을 기반으로 정확한 길이의 무음을 생성합니다.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    silence_file = os.path.join(base_dir, 'silence_5s.wav')

    # 5초 무음 파일 로드 (원본 샘플레이트 유지)
    silence_5s, file_sr = librosa.load(silence_file, sr=None)

    # 타겟 샘플레이트와 다르면 리샘플링
    if file_sr != target_sample_rate:
        silence_5s = librosa.resample(silence_5s, orig_sr=file_sr, target_sr=target_sample_rate)

    # 필요한 샘플 수 계산
    needed_samples = int(target_sample_rate * duration_seconds)

    # 5초 무음의 샘플 수
    samples_per_5s = len(silence_5s)

    if duration_seconds <= 5:
        # 5초 이하면 필요한 만큼만 잘라서 반환
        return silence_5s[:needed_samples]
    else:
        # 5초 초과면 반복해서 붙이기
        repeats = int(np.ceil(duration_seconds / 5))
        extended_silence = np.tile(silence_5s, repeats)
        return extended_silence[:needed_samples]


def generate_silence_samples(duration_seconds, sample_rate):
    """지정된 시간(초)의 무음 샘플 배열을 생성합니다. (파일 기반)"""
    return get_silence_audio(duration_seconds, sample_rate)

# ===== 오디오 처리 함수 =====
def combine_audio_segments(audio_segments, sample_rate):
    """여러 오디오 세그먼트를 하나로 합칩니다.

    audio_segments: [(audio_bytes, silence_duration), ...] 형태의 리스트
    silence_duration이 0보다 크면 해당 오디오 뒤에 무음 추가
    """
    if not audio_segments:
        return None

    if len(audio_segments) == 1 and audio_segments[0][1] == 0:
        return audio_segments[0][0]

    print("\n오디오 데이터 합치는 중...")
    audio_arrays = []

    for i, (audio_data, silence_duration) in enumerate(audio_segments):
        # 오디오를 numpy 배열로 변환 (sr=None으로 원본 샘플레이트 유지)
        temp_bio = io.BytesIO(audio_data)
        audio_array, sr = librosa.load(temp_bio, sr=None)
        audio_arrays.append(audio_array)

        # 지정된 무음 추가 (#숫자로 지정한 무음)
        if silence_duration > 0:
            silence = generate_silence_samples(silence_duration, sr)
            audio_arrays.append(silence)
            print(f"  세그먼트 {i+1}: {silence_duration}초 무음 추가 (정확히 {len(silence)} 샘플)")
        # 마지막이 아니면 기본 문장 간격 무음 추가
        elif i < len(audio_segments) - 1:
            silence = generate_silence_samples(SILENCE_DURATION, sr)
            audio_arrays.append(silence)

    # 오디오 배열들을 연결
    combined_audio = np.concatenate(audio_arrays)

    # WAV 형식으로 저장
    bio = io.BytesIO()
    soundfile.write(bio, combined_audio, sample_rate, format='WAV')
    bio.seek(0)

    return bio.getvalue()



# ===== 전역 객체 초기화 =====
model_manager_b = ModelManager("B모델 (G_550000)", 'G_550000_config.json', 'G_550000.pth')
speaker_info_manager = SpeakerInfoManager()

# 스피커 목록 초기화
speakers = list(model_manager_b.model.hps.data.spk2id.keys())
print(f"사용 가능한 화자: {speakers}")

# ===== 메인 합성 함수 =====
def synthesize(speaker, text, speed, model_manager):
    """텍스트를 음성으로 합성합니다. #숫자 패턴으로 무음 삽입 가능."""
    speaker_texts = parse_speaker_text(text, speaker)

    print(f"=== 음성 합성 시작 ({model_manager.model_name}) ===")
    print(f"문장 수: {len(speaker_texts)}")

    # (audio_data, silence_duration) 튜플 리스트
    audio_segments = []
    sample_rate = 44100  # 기본 샘플 레이트

    for i, (current_speaker, sentence) in enumerate(speaker_texts):
        if not sentence:
            continue

        # #숫자 패턴 파싱하여 텍스트와 무음 시간 분리
        text_silence_pairs = parse_text_with_silence(sentence)

        for text_part, silence_duration in text_silence_pairs:
            if not text_part:
                continue

            text_part = preprocess_text(text_part)

            print(f"\n문장 {i+1} 처리 중...")
            print(f"화자: {current_speaker}")
            print(f"텍스트: {text_part}")
            if silence_duration > 0:
                print(f"무음 삽입 예정: {silence_duration}초")

            audio_data = generate_audio_for_speaker(
                text_part, current_speaker, speed, model_manager
            )

            if audio_data:
                # 샘플 레이트 확인
                temp_bio = io.BytesIO(audio_data)
                _, sample_rate = librosa.load(temp_bio, sr=None)

                # (오디오, 무음시간) 튜플로 저장
                audio_segments.append((audio_data, silence_duration))

    if not audio_segments:
        print("경고: 생성된 오디오가 없습니다.")
        return None

    combined_audio = combine_audio_segments(audio_segments, sample_rate)

    print(f"\n=== 음성 합성 완료 ({model_manager.model_name}) ===")
    return combined_audio

def generate_audio_for_speaker(sentence, speaker, speed, model_manager):
    """특정 화자로 오디오를 생성합니다."""
    temp_bio = io.BytesIO()
    actual_speaker = speaker.split(' (')[0] if '(' in speaker else speaker

    try:
        speaker_ids = model_manager.get_speaker_ids()
        model = model_manager.get_model()

        if actual_speaker in speaker_ids:
            model.tts_to_file(
                sentence, speaker_ids[actual_speaker], temp_bio,
                speed=speed, pbar=None, format='wav', quiet=True
            )
        else:
            print(f"경고: 화자 '{actual_speaker}'를 찾을 수 없습니다. 기본 화자 사용.")
            first_speaker_id = list(speaker_ids.values())[0] if speaker_ids else 0
            model.tts_to_file(
                sentence, first_speaker_id, temp_bio,
                speed=speed, pbar=None, format='wav', quiet=True
            )

        temp_bio.seek(0)
        return temp_bio.read()

    except Exception as e:
        print(f"오디오 생성 중 오류 발생: {e}")
        return None

# ===== 텍스트 하이라이트를 위한 문장 분리 =====
def split_text_to_sentences(text):
    """텍스트를 문장 단위로 분리합니다. # 패턴도 분리자로 처리."""
    # #숫자 패턴 제거하고 문장 분리
    text_clean = re.sub(r'#\d*', '.', text)
    # 줄바꿈도 문장 구분자로 처리
    text_clean = text_clean.replace('\n', '.')
    # 여러 개의 점을 하나로
    text_clean = re.sub(r'\.+', '.', text_clean)
    # 문장 분리
    sentences = [s.strip() for s in text_clean.split('.') if s.strip()]
    return sentences


# ===== Gradio UI =====
def create_ui():
    """Gradio UI를 생성합니다."""

    # 자동 스크롤 JavaScript
    scroll_js = """
    function() {
        // MutationObserver로 highlight-container 변경 감지
        const observer = new MutationObserver(function(mutations) {
            setTimeout(function() {
                var el = document.getElementById('current-sentence');
                var container = document.getElementById('highlight-scroll-container');
                if (el && container) {
                    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            }, 100);
        });

        // highlight-container 감시 시작
        const targetNode = document.getElementById('highlight-container');
        if (targetNode) {
            observer.observe(targetNode, { childList: true, subtree: true, characterData: true });
        }

        // 페이지 로드 후 다시 시도
        setTimeout(function() {
            const targetNode2 = document.getElementById('highlight-container');
            if (targetNode2) {
                observer.observe(targetNode2, { childList: true, subtree: true, characterData: true });
            }
        }, 1000);
    }
    """

    with gr.Blocks(title="TTS 모델", js=scroll_js) as demo:
        # gr.Markdown('# TTS 학습 모델')
#         gr.Markdown('''
# **모델 학습 플로우**: 원본(4개) → 음성 복제 → 복제 음성으로 1,000개 이상 모델 학습용 음원 생성 → 모델 학습 및 튜닝 → 모델 생성
#         ''')

        # ===== 음성 합성 영역 =====
        # gr.Markdown('## 음성 합성')

        # 재생 중 텍스트 표시 영역 (상단)
        gr.Markdown('### 재생 중 텍스트')
        highlight_display = gr.HTML(
            value='<div style="padding: 16px; background-color: #f9fafb; border-radius: 8px; line-height: 1.8; font-size: 16px; min-height: 100px; max-height: 300px; overflow-y: auto;">음성 합성 버튼을 클릭하세요.</div>',
            elem_id="highlight-container"
        )

        # 오디오 플레이어
        aud_synth = gr.Audio(interactive=False, elem_id="synth-audio", autoplay=True)

        # 이전/다음 파일 버튼
        with gr.Row():
            btn_prev = gr.Button('◀ 이전 문장', scale=1, elem_id="btn-prev-sentence")
            btn_next = gr.Button('다음 문장 ▶', scale=1, elem_id="btn-next-sentence")

        result_md = gr.Markdown('처리단계 : -  총 걸린시간 -')

        # 화자 선택 및 텍스트 입력 (하단)
        gr.Markdown('---')
        with gr.Group():
            # 첫 번째 화자의 기본 텍스트 가져오기
            initial_speaker = speakers[0] if speakers else ''
            initial_text = SPEAKER_DEFAULT_TEXTS.get(initial_speaker, DEFAULT_TEXT)

            speaker_select = gr.Radio(
                choices=speakers,
                value=initial_speaker,
                label='화자 선택',
                interactive=True
            )

            text_input = gr.Textbox(
                value=initial_text,
                label='합성할 텍스트',
                placeholder="예: 안녕하세요#5다음 문장입니다. (#5 = 5초 무음)"
            )

            def update_text(speaker):
                """화자 변경 시 기본 텍스트 업데이트"""
                return SPEAKER_DEFAULT_TEXTS.get(speaker, DEFAULT_TEXT)

            speaker_select.change(
                fn=update_text,
                inputs=[speaker_select],
                outputs=[text_input]
            )

        btn_synth = gr.Button('음성합성', variant='primary')

        # 상태 관리
        sentences_state = gr.State([])
        current_idx_state = gr.State(0)
        audio_queue_state = gr.State([])  # [session_id, audio_path_0, audio_path_1, ...]
        synthesis_done_state = gr.State(False)  # 합성 완료 여부

        # 전역 합성 큐 (백그라운드 스레드와 공유)
        import threading
        import queue

        # 세션별 큐 저장소
        synthesis_queues = {}
        synthesis_threads = {}

        def make_highlight_html(sentences, current_index):
            """하이라이트 HTML 생성 (자동 스크롤 포함)"""
            if not sentences:
                return '<div style="padding: 16px; background-color: #f9fafb; border-radius: 8px; line-height: 1.8; font-size: 16px; min-height: 100px;">텍스트가 없습니다.</div>'

            html = '<div id="highlight-scroll-container" style="padding: 16px; background-color: #f9fafb; border-radius: 8px; line-height: 1.8; font-size: 16px; min-height: 100px; max-height: 300px; overflow-y: auto;">'
            for j, s in enumerate(sentences):
                if j == current_index:
                    # 현재 재생 중인 문장에 ID 부여
                    html += f'<span id="current-sentence" style="background-color: #fef08a; padding: 2px 4px; border-radius: 4px; margin-right: 4px;">{s}.</span> '
                elif j < current_index:
                    html += f'<span style="color: #9ca3af; margin-right: 4px;">{s}.</span> '
                else:
                    html += f'<span style="margin-right: 4px;">{s}.</span> '
            html += '</div>'
            return html

        def background_synthesize(session_id, sentences, speaker, start_idx=1):
            """백그라운드에서 나머지 문장들 합성"""
            q = synthesis_queues.get(session_id)
            if not q:
                return

            temp_dir = tempfile.gettempdir()
            for i in range(start_idx, len(sentences)):
                print(f"[{session_id}] 백그라운드 합성 {i+1}/{len(sentences)}: {sentences[i]}")
                audio_data = generate_audio_for_speaker(
                    preprocess_text(sentences[i]), speaker, DEFAULT_SPEED, model_manager_b
                )
                if audio_data:
                    audio_path = os.path.join(temp_dir, f"tts_{session_id}_{i}.wav")
                    with open(audio_path, 'wb') as f:
                        f.write(audio_data)
                    q.put(audio_path)
                    print(f"[{session_id}] 백그라운드 합성 완료 {i+1}, 큐에 추가")

            # 합성 완료 마커
            q.put(None)
            print(f"[{session_id}] 모든 합성 완료")

        def start_synthesis(speaker, text):
            """첫 문장 합성 + 백그라운드 합성 시작"""
            sentences = split_text_to_sentences(text)
            if not sentences:
                return (
                    None,
                    '<div style="padding: 16px; background-color: #f9fafb; border-radius: 8px;">텍스트가 없습니다.</div>',
                    '처리단계 : -',
                    [],
                    0,
                    [],
                    True
                )

            # 세션 ID 생성
            session_id = f"{int(time.time() * 1000)}"

            # 큐 생성
            synthesis_queues[session_id] = queue.Queue()

            # 첫 문장 합성 (메인 스레드에서)
            print(f"[{session_id}] 첫 문장 합성 시작: {sentences[0]}")
            first_audio = generate_audio_for_speaker(
                preprocess_text(sentences[0]), speaker, DEFAULT_SPEED, model_manager_b
            )

            if not first_audio:
                return (
                    None,
                    '<div style="padding: 16px; background-color: #f9fafb; border-radius: 8px;">합성 실패</div>',
                    '합성 실패',
                    [],
                    0,
                    [],
                    True
                )

            temp_dir = tempfile.gettempdir()
            first_path = os.path.join(temp_dir, f"tts_{session_id}_0.wav")
            with open(first_path, 'wb') as f:
                f.write(first_audio)
            print(f"[{session_id}] 첫 문장 합성 완료, 재생 시작")

            # 백그라운드 스레드에서 나머지 합성 시작
            if len(sentences) > 1:
                thread = threading.Thread(
                    target=background_synthesize,
                    args=(session_id, sentences, speaker, 1),
                    daemon=True
                )
                synthesis_threads[session_id] = thread
                thread.start()
                synthesis_done = False
            else:
                synthesis_done = True

            # 초기 큐 상태: [session_id, 첫번째 오디오 경로]
            initial_queue = [session_id, first_path]

            return (
                first_path,
                make_highlight_html(sentences, 0),
                f'재생중: 1/{len(sentences)}' + (' (합성중...)' if not synthesis_done else ''),
                sentences,
                0,
                initial_queue,
                synthesis_done
            )

        def play_next_from_queue(sentences, current_idx, audio_queue, synthesis_done):
            """큐에서 다음 오디오 가져와서 재생"""
            next_idx = current_idx + 1

            if next_idx >= len(sentences):
                # 모든 재생 완료
                final_html = '<div style="padding: 16px; background-color: #f9fafb; border-radius: 8px; line-height: 1.8; font-size: 16px; min-height: 100px; max-height: 300px; overflow-y: auto;">'
                for s in sentences:
                    final_html += f'<span style="color: #9ca3af; margin-right: 4px;">{s}.</span> '
                final_html += '</div>'
                return (
                    None,
                    final_html,
                    f'완료: {len(sentences)}문장',
                    sentences,
                    next_idx,
                    audio_queue,  # 큐 유지 (이전/다음 버튼용)
                    True
                )

            # session_id 가져오기
            session_id = audio_queue[0] if audio_queue else None
            if not session_id or session_id not in synthesis_queues:
                return (
                    None,
                    make_highlight_html(sentences, current_idx),
                    '오류: 세션 없음',
                    sentences,
                    current_idx,
                    audio_queue,
                    True
                )

            q = synthesis_queues[session_id]

            # 큐에서 다음 오디오 가져오기 (최대 30초 대기)
            try:
                next_audio = q.get(timeout=30)
            except queue.Empty:
                print(f"[{session_id}] 큐 대기 시간 초과")
                return (
                    None,
                    make_highlight_html(sentences, current_idx),
                    '합성 대기 시간 초과',
                    sentences,
                    current_idx,
                    audio_queue,
                    True
                )

            if next_audio is None:
                # 합성 완료 마커
                synthesis_done = True
                # 큐 정리
                if session_id in synthesis_queues:
                    del synthesis_queues[session_id]
                if session_id in synthesis_threads:
                    del synthesis_threads[session_id]

                final_html = '<div style="padding: 16px; background-color: #f9fafb; border-radius: 8px; line-height: 1.8; font-size: 16px; min-height: 100px; max-height: 300px; overflow-y: auto;">'
                for s in sentences:
                    final_html += f'<span style="color: #9ca3af; margin-right: 4px;">{s}.</span> '
                final_html += '</div>'
                return (
                    None,
                    final_html,
                    f'완료: {len(sentences)}문장',
                    sentences,
                    len(sentences),
                    audio_queue,  # 큐 유지
                    True
                )

            print(f"[{session_id}] 큐에서 가져옴: {next_audio}, 재생 시작")

            # 오디오 경로를 큐에 저장
            updated_queue = audio_queue + [next_audio]

            status_msg = f'재생중: {next_idx + 1}/{len(sentences)}'
            if not synthesis_done and next_idx + 1 < len(sentences):
                status_msg += ' (합성중...)'

            return (
                next_audio,
                make_highlight_html(sentences, next_idx),
                status_msg,
                sentences,
                next_idx,
                updated_queue,
                synthesis_done
            )

        btn_synth.click(
            fn=start_synthesis,
            inputs=[speaker_select, text_input],
            outputs=[aud_synth, highlight_display, result_md, sentences_state, current_idx_state, audio_queue_state, synthesis_done_state],
            show_progress="minimal"
        )

        # 오디오 재생 완료 시 큐에서 다음 오디오 가져와서 재생
        aud_synth.stop(
            fn=play_next_from_queue,
            inputs=[sentences_state, current_idx_state, audio_queue_state, synthesis_done_state],
            outputs=[aud_synth, highlight_display, result_md, sentences_state, current_idx_state, audio_queue_state, synthesis_done_state]
        )

        def go_to_prev_sentence(sentences, current_idx, audio_queue, synthesis_done):
            """이전 문장으로 이동"""
            if not sentences or not audio_queue or len(audio_queue) < 2:
                return (
                    None,
                    make_highlight_html(sentences, current_idx) if sentences else '<div style="padding: 16px; background-color: #f9fafb; border-radius: 8px;">텍스트가 없습니다.</div>',
                    '이전 문장 없음',
                    sentences,
                    current_idx,
                    audio_queue,
                    synthesis_done
                )

            # audio_queue 구조: [session_id, audio_path_0, audio_path_1, ...]
            # current_idx가 0이면 이전 없음
            if current_idx <= 0:
                return (
                    audio_queue[1] if len(audio_queue) > 1 else None,  # 첫번째 오디오 다시 재생
                    make_highlight_html(sentences, 0),
                    f'재생중: 1/{len(sentences)}',
                    sentences,
                    0,
                    audio_queue,
                    synthesis_done
                )

            prev_idx = current_idx - 1
            # audio_queue[prev_idx + 1]이 해당 인덱스의 오디오 경로
            if prev_idx + 1 < len(audio_queue):
                prev_audio = audio_queue[prev_idx + 1]
                return (
                    prev_audio,
                    make_highlight_html(sentences, prev_idx),
                    f'재생중: {prev_idx + 1}/{len(sentences)}',
                    sentences,
                    prev_idx,
                    audio_queue,
                    synthesis_done
                )
            else:
                return (
                    None,
                    make_highlight_html(sentences, current_idx),
                    '이전 오디오 없음',
                    sentences,
                    current_idx,
                    audio_queue,
                    synthesis_done
                )

        def go_to_next_sentence(sentences, current_idx, audio_queue, synthesis_done):
            """다음 문장으로 이동"""
            if not sentences or not audio_queue:
                return (
                    None,
                    make_highlight_html(sentences, current_idx) if sentences else '<div style="padding: 16px; background-color: #f9fafb; border-radius: 8px;">텍스트가 없습니다.</div>',
                    '다음 문장 없음',
                    sentences,
                    current_idx,
                    audio_queue,
                    synthesis_done
                )

            next_idx = current_idx + 1
            if next_idx >= len(sentences):
                # 마지막 문장 - 완료
                final_html = '<div style="padding: 16px; background-color: #f9fafb; border-radius: 8px; line-height: 1.8; font-size: 16px; min-height: 100px; max-height: 300px; overflow-y: auto;">'
                for s in sentences:
                    final_html += f'<span style="color: #9ca3af; margin-right: 4px;">{s}.</span> '
                final_html += '</div>'
                return (
                    None,
                    final_html,
                    f'완료: {len(sentences)}문장',
                    sentences,
                    len(sentences),
                    audio_queue,
                    synthesis_done
                )

            # audio_queue[next_idx + 1]이 해당 인덱스의 오디오 경로
            if next_idx + 1 < len(audio_queue):
                # 이미 합성된 오디오가 있으면 바로 재생
                next_audio = audio_queue[next_idx + 1]
                return (
                    next_audio,
                    make_highlight_html(sentences, next_idx),
                    f'재생중: {next_idx + 1}/{len(sentences)}',
                    sentences,
                    next_idx,
                    audio_queue,
                    synthesis_done
                )
            else:
                # 아직 합성 안됨 - 큐에서 가져오기
                return play_next_from_queue(sentences, current_idx, audio_queue, synthesis_done)

        btn_prev.click(
            fn=go_to_prev_sentence,
            inputs=[sentences_state, current_idx_state, audio_queue_state, synthesis_done_state],
            outputs=[aud_synth, highlight_display, result_md, sentences_state, current_idx_state, audio_queue_state, synthesis_done_state]
        )

        btn_next.click(
            fn=go_to_next_sentence,
            inputs=[sentences_state, current_idx_state, audio_queue_state, synthesis_done_state],
            outputs=[aud_synth, highlight_display, result_md, sentences_state, current_idx_state, audio_queue_state, synthesis_done_state]
        )

        # ===== 음원 파일 재생 영역 =====
        gr.Markdown('---')
        gr.Markdown('## 음원 파일 재생')

        with gr.Tabs():
            with gr.Tab("원본"), gr.Group():
                aaa_speaker_select = gr.Radio(
                    choices=speakers,
                    value=speakers[0] if speakers else '',
                    label='화자 선택'
                )

                # 초기 파일 목록 로드
                import os
                base_dir = os.path.dirname(os.path.abspath(__file__))
                initial_speaker = speakers[0] if speakers else ''
                initial_audio_dir = os.path.join(base_dir, 'mp3', 'aaa', initial_speaker)
                initial_files = []
                initial_audio_path = None
                if os.path.exists(initial_audio_dir):
                    initial_files = sorted([f for f in os.listdir(initial_audio_dir) if f.endswith('.wav')])
                    if initial_files:
                        initial_audio_path = os.path.join(initial_audio_dir, initial_files[0])

                aaa_file_dropdown = gr.Dropdown(
                    choices=initial_files,
                    value=initial_files[0] if initial_files else None,
                    label='오디오 파일 선택'
                )
                aaa_audio = gr.Audio(interactive=False, value=initial_audio_path)

                def load_aaa_files(speaker):
                    import os
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    audio_dir = os.path.join(base_dir, 'mp3', 'aaa', speaker)
                    if os.path.exists(audio_dir):
                        files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
                        return gr.update(choices=files, value=files[0] if files else None)
                    return gr.update(choices=[], value=None)

                def play_aaa_audio(speaker, filename):
                    if not filename:
                        return None
                    import os
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    audio_path = os.path.join(base_dir, 'mp3', 'aaa', speaker, filename)
                    if os.path.exists(audio_path):
                        return audio_path
                    return None

                aaa_speaker_select.change(
                    fn=load_aaa_files,
                    inputs=[aaa_speaker_select],
                    outputs=[aaa_file_dropdown]
                )
                aaa_file_dropdown.change(
                    fn=play_aaa_audio,
                    inputs=[aaa_speaker_select, aaa_file_dropdown],
                    outputs=[aaa_audio]
                )

            with gr.Tab("음성복제"), gr.Group():
                bbb_speaker_select = gr.Radio(
                    choices=speakers,
                    value=speakers[0] if speakers else '',
                    label='화자 선택'
                )

                # 초기 파일 목록 로드
                initial_speaker_bbb = speakers[0] if speakers else ''
                initial_audio_dir_bbb = os.path.join(base_dir, 'mp3', 'bbb', initial_speaker_bbb)
                initial_files_bbb = []
                initial_audio_path_bbb = None
                if os.path.exists(initial_audio_dir_bbb):
                    initial_files_bbb = sorted([f for f in os.listdir(initial_audio_dir_bbb) if f.endswith('.wav')])
                    if initial_files_bbb:
                        initial_audio_path_bbb = os.path.join(initial_audio_dir_bbb, initial_files_bbb[0])

                bbb_file_dropdown = gr.Dropdown(
                    choices=initial_files_bbb,
                    value=initial_files_bbb[0] if initial_files_bbb else None,
                    label='오디오 파일 선택'
                )
                bbb_audio = gr.Audio(interactive=False, value=initial_audio_path_bbb)

                def load_bbb_files(speaker):
                    import os
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    audio_dir = os.path.join(base_dir, 'mp3', 'bbb', speaker)
                    if os.path.exists(audio_dir):
                        files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
                        return gr.update(choices=files, value=files[0] if files else None)
                    return gr.update(choices=[], value=None)

                def play_bbb_audio(speaker, filename):
                    if not filename:
                        return None
                    import os
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    audio_path = os.path.join(base_dir, 'mp3', 'bbb', speaker, filename)
                    if os.path.exists(audio_path):
                        return audio_path
                    return None

                bbb_speaker_select.change(
                    fn=load_bbb_files,
                    inputs=[bbb_speaker_select],
                    outputs=[bbb_file_dropdown]
                )
                bbb_file_dropdown.change(
                    fn=play_bbb_audio,
                    inputs=[bbb_speaker_select, bbb_file_dropdown],
                    outputs=[bbb_audio]
                )

            with gr.Tab("모델A"), gr.Group():
                model_a_speaker_select = gr.Radio(
                    choices=speakers,
                    value=speakers[0] if speakers else '',
                    label='화자 선택'
                )

                # 초기 파일 목록 로드
                initial_speaker_model_a = speakers[0] if speakers else ''
                initial_audio_dir_model_a = os.path.join(base_dir, 'mp3', 'A', initial_speaker_model_a)
                initial_files_model_a = []
                initial_audio_path_model_a = None
                if os.path.exists(initial_audio_dir_model_a):
                    initial_files_model_a = sorted([f for f in os.listdir(initial_audio_dir_model_a) if f.endswith('.wav')])
                    if initial_files_model_a:
                        initial_audio_path_model_a = os.path.join(initial_audio_dir_model_a, initial_files_model_a[0])

                model_a_file_dropdown = gr.Dropdown(
                    choices=initial_files_model_a,
                    value=initial_files_model_a[0] if initial_files_model_a else None,
                    label='오디오 파일 선택'
                )
                model_a_audio = gr.Audio(interactive=False, value=initial_audio_path_model_a)

                def load_model_a_files(speaker):
                    import os
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    audio_dir = os.path.join(base_dir, 'mp3', 'A', speaker)
                    if os.path.exists(audio_dir):
                        files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
                        return gr.update(choices=files, value=files[0] if files else None)
                    return gr.update(choices=[], value=None)

                def play_model_a_audio(speaker, filename):
                    if not filename:
                        return None
                    import os
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    audio_path = os.path.join(base_dir, 'mp3', 'A', speaker, filename)
                    if os.path.exists(audio_path):
                        return audio_path
                    return None

                model_a_speaker_select.change(
                    fn=load_model_a_files,
                    inputs=[model_a_speaker_select],
                    outputs=[model_a_file_dropdown]
                )
                model_a_file_dropdown.change(
                    fn=play_model_a_audio,
                    inputs=[model_a_speaker_select, model_a_file_dropdown],
                    outputs=[model_a_audio]
                )

            with gr.Tab("모델B"), gr.Group():
                model_b_speaker_select = gr.Radio(
                    choices=speakers,
                    value=speakers[0] if speakers else '',
                    label='화자 선택'
                )

                # 초기 파일 목록 로드
                initial_speaker_model_b = speakers[0] if speakers else ''
                initial_audio_dir_model_b = os.path.join(base_dir, 'mp3', 'B', initial_speaker_model_b)
                initial_files_model_b = []
                initial_audio_path_model_b = None
                if os.path.exists(initial_audio_dir_model_b):
                    initial_files_model_b = sorted([f for f in os.listdir(initial_audio_dir_model_b) if f.endswith('.wav')])
                    if initial_files_model_b:
                        initial_audio_path_model_b = os.path.join(initial_audio_dir_model_b, initial_files_model_b[0])

                model_b_file_dropdown = gr.Dropdown(
                    choices=initial_files_model_b,
                    value=initial_files_model_b[0] if initial_files_model_b else None,
                    label='오디오 파일 선택'
                )
                model_b_audio = gr.Audio(interactive=False, value=initial_audio_path_model_b)

                def load_model_b_files(speaker):
                    import os
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    audio_dir = os.path.join(base_dir, 'mp3', 'B', speaker)
                    if os.path.exists(audio_dir):
                        files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
                        return gr.update(choices=files, value=files[0] if files else None)
                    return gr.update(choices=[], value=None)

                def play_model_b_audio(speaker, filename):
                    if not filename:
                        return None
                    import os
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    audio_path = os.path.join(base_dir, 'mp3', 'B', speaker, filename)
                    if os.path.exists(audio_path):
                        return audio_path
                    return None

                model_b_speaker_select.change(
                    fn=load_model_b_files,
                    inputs=[model_b_speaker_select],
                    outputs=[model_b_file_dropdown]
                )
                model_b_file_dropdown.change(
                    fn=play_model_b_audio,
                    inputs=[model_b_speaker_select, model_b_file_dropdown],
                    outputs=[model_b_audio]
                )

    return demo

# ===== 메인 함수 =====
@click.command()
@click.option('--share', '-s', is_flag=True, show_default=True, default=False,
              help="Expose a publicly-accessible shared Gradio link.")
@click.option('--host', '-h', default="0.0.0.0")
@click.option('--port', '-p', type=int, default=7860)
def main(share, host, port):
    """메인 실행 함수"""
    demo = create_ui()
    demo.queue().launch(
        share=share,
        server_name=host,
        server_port=port
    )

if __name__ == "__main__":
    main()
