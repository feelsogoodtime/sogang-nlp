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
import soundfile
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

# ===== 모델 관리 클래스 =====
class ModelManager:
    """TTS 모델을 관리하는 클래스"""
    
    def __init__(self):
        print("모델 로딩 중...")
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model = TTS(language='KR', device=DEVICE,
                        config_path=os.path.join(base_dir, 'model', 'G_55000_config.json'),
                        ckpt_path=os.path.join(base_dir, 'model', 'G_55000.pth'))
        print("모델 로딩 완료!")
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

# ===== 오디오 처리 함수 =====
def combine_audio_segments(audio_list, sample_rate):
    """여러 오디오 세그먼트를 하나로 합칩니다."""
    if len(audio_list) <= 1:
        return audio_list[0] if audio_list else None
    
    print("\n오디오 데이터 합치는 중...")
    audio_arrays = []
    
    for audio_data in audio_list:
        temp_bio = io.BytesIO(audio_data)
        audio_array, sr = librosa.load(temp_bio, sr=sample_rate)
        audio_arrays.append(audio_array)
        
        # 문장 사이에 무음 추가
        silence = np.zeros(int(sr * SILENCE_DURATION))
        audio_arrays.append(silence)
    
    # 마지막 무음 제거
    audio_arrays = audio_arrays[:-1]
    
    # 오디오 배열들을 연결
    combined_audio = np.concatenate(audio_arrays)
    
    # WAV 형식으로 저장
    bio = io.BytesIO()
    soundfile.write(bio, combined_audio, sample_rate, format='WAV')
    bio.seek(0)
    
    return bio.getvalue()



# ===== 전역 객체 초기화 =====
model_manager = ModelManager()
speaker_info_manager = SpeakerInfoManager()

# 스피커 목록 초기화
speakers = list(model_manager.model.hps.data.spk2id.keys())
print(f"사용 가능한 화자: {speakers}")

# ===== 메인 합성 함수 =====
def synthesize(speaker, text, speed, progress=gr.Progress()):
    """텍스트를 음성으로 합성합니다."""
    speaker_texts = parse_speaker_text(text, speaker)

    print(f"=== 음성 합성 시작 ===")
    print(f"문장 수: {len(speaker_texts)}")

    audio_list = []
    sample_rate = None

    for i, (current_speaker, sentence) in enumerate(speaker_texts):
        if not sentence:
            continue

        sentence = preprocess_text(sentence)

        print(f"\n문장 {i+1} 처리 중...")
        print(f"화자: {current_speaker}")
        print(f"텍스트: {sentence}")

        audio_data = generate_audio_for_speaker(
            sentence, current_speaker, speed, progress
        )

        if audio_data:
            audio_list.append(audio_data)

            if sample_rate is None:
                temp_bio = io.BytesIO(audio_data)
                _, sample_rate = librosa.load(temp_bio, sr=None)

    if not audio_list:
        print("경고: 생성된 오디오가 없습니다.")
        return None

    combined_audio = combine_audio_segments(audio_list, sample_rate)

    print("\n=== 음성 합성 완료 ===")
    return combined_audio

def generate_audio_for_speaker(sentence, speaker, speed, progress):
    """특정 화자로 오디오를 생성합니다."""
    temp_bio = io.BytesIO()
    actual_speaker = speaker.split(' (')[0] if '(' in speaker else speaker

    try:
        speaker_ids = model_manager.get_speaker_ids()
        model = model_manager.get_model()

        if actual_speaker in speaker_ids:
            model.tts_to_file(
                sentence, speaker_ids[actual_speaker], temp_bio,
                speed=speed, pbar=progress.tqdm, format='wav', quiet=True
            )
        else:
            print(f"경고: 화자 '{actual_speaker}'를 찾을 수 없습니다. 기본 화자 사용.")
            first_speaker_id = list(speaker_ids.values())[0] if speaker_ids else 0
            model.tts_to_file(
                sentence, first_speaker_id, temp_bio,
                speed=speed, pbar=progress.tqdm, format='wav', quiet=True
            )

        temp_bio.seek(0)
        return temp_bio.read()

    except Exception as e:
        print(f"오디오 생성 중 오류 발생: {e}")
        return None

# ===== Gradio UI =====
def create_ui():
    """Gradio UI를 생성합니다."""
    with gr.Blocks(title="TTS 모델") as demo:
        gr.Markdown('# TTS 학습 모델')
        gr.Markdown('## 음성 합성')
        gr.Markdown('''
**모델 학습 플로우**: 원본(4개) → 음성 복제 → 복제 음성으로 1,000개 이상 모델 학습용 음원 생성 → 모델 학습 및 튜닝 → 모델 생성

        ''')

        with gr.Group():
            speaker = gr.Radio(
                choices=speakers,
                value=speakers[0] if speakers else '',
                label='화자 선택',
                interactive=True
            )

            text = gr.Textbox(
                label="합성할 텍스트",
                value=DEFAULT_TEXT,
                placeholder="예: 안녕하세요. 텍스트 음성 변환입니다."
            )

        btn = gr.Button('음성합성', variant='primary')

        aud = gr.Audio(interactive=False)
        result_md = gr.Markdown('처리단계 : -  총 걸린시간 -')

        def process_synthesis(speaker, text, progress=gr.Progress()):
            start_time = time.time()
            audio_data = synthesize(speaker, text, DEFAULT_SPEED, progress)
            end_time = time.time()

            # 문장 수 계산
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            total_sentences = len(sentences)

            # 총 걸린 시간 계산
            elapsed_time = round(end_time - start_time, 3)

            if audio_data:
                return audio_data, f'처리단계 : {total_sentences}문장 | 총 걸린시간 {elapsed_time}초'
            return None, '처리단계 : -  총 걸린시간 -'

        btn.click(
            fn=process_synthesis,
            inputs=[speaker, text],
            outputs=[aud, result_md]
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

    return demo

# ===== 메인 함수 =====
@click.command()
@click.option('--share', '-s', is_flag=True, show_default=True, default=False, 
              help="Expose a publicly-accessible shared Gradio link.")
@click.option('--host', '-h', default=None)
@click.option('--port', '-p', type=int, default=None)
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
