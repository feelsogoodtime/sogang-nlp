import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile
import io
from typing import Optional, Tuple
import re

class MiniTTS(nn.Module):
    """mini.ckpt를 사용하는 TTS 클래스"""
    
    def __init__(self, ckpt_path: str, device: str = 'auto'):
        super().__init__()
        
        if device == 'auto':
            device = 'cpu'
            if torch.cuda.is_available(): 
                device = 'cuda'
            if torch.backends.mps.is_available(): 
                device = 'mps'
        
        self.device = device
        
        # 체크포인트 로드
        print("Mini TTS 모델 로딩 중...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        self.hparams = checkpoint.get('hyper_parameters', {})
        
        # 모델 구조 초기화 (간단한 구조)
        self._init_model()
        
        # 모델을 디바이스로 이동
        self.to(device)
        
        # 가중치 로드 (호환되는 부분만)
        self._load_compatible_weights(checkpoint['state_dict'])
        
        self.eval()
        print("Mini TTS 모델 로딩 완료!")
    
    def _init_model(self):
        """모델 구조 초기화"""
        # 간단한 인코더-디코더 구조
        embed_dim = self.hparams.get('embed_dim', 128)
        vocab_size = 100  # 기본 음소 사전 크기
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, embed_dim, batch_first=True)
        self.decoder = nn.LSTM(embed_dim, embed_dim, batch_first=True)
        self.mel_proj = nn.Linear(embed_dim, 80)  # mel spectrogram 차원
        self.vocoder = SimpleVocoder()
        
        # 스피커 임베딩
        self.speaker_embedding = nn.Embedding(1, embed_dim)  # 단일 스피커
    
    def _load_compatible_weights(self, state_dict):
        """호환되는 가중치만 로드"""
        model_dict = self.state_dict()
        
        # 호환되는 키만 필터링
        compatible_keys = []
        for key in state_dict.keys():
            if key in model_dict and state_dict[key].shape == model_dict[key].shape:
                compatible_keys.append(key)
        
        if compatible_keys:
            print(f"호환되는 가중치 {len(compatible_keys)}개 로드")
            # 호환되는 가중치만 로드
            for key in compatible_keys:
                model_dict[key] = state_dict[key]
            self.load_state_dict(model_dict, strict=False)
        else:
            print("호환되는 가중치가 없습니다. 랜덤 초기화를 사용합니다.")
    
    def text_to_sequence(self, text: str) -> torch.Tensor:
        """텍스트를 음소 시퀀스로 변환 (간단한 구현)"""
        # 간단한 음소 변환 (실제로는 더 복잡한 전처리가 필요)
        text = text.lower()
        # 기본적인 음소 매핑
        phoneme_map = {
            'a': 1, 'e': 2, 'i': 3, 'o': 4, 'u': 5,
            'b': 6, 'c': 7, 'd': 8, 'f': 9, 'g': 10,
            'h': 11, 'j': 12, 'k': 13, 'l': 14, 'm': 15,
            'n': 16, 'p': 17, 'q': 18, 'r': 19, 's': 20,
            't': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26
        }
        
        sequence = []
        for char in text:
            if char in phoneme_map:
                sequence.append(phoneme_map[char])
            else:
                sequence.append(0)  # unknown token
        
        tensor = torch.tensor(sequence, dtype=torch.long)
        return tensor.to(self.device)
    
    def forward(self, text: str, speaker_id: int = 0) -> torch.Tensor:
        """텍스트를 오디오로 변환"""
        # 텍스트를 시퀀스로 변환
        sequence = self.text_to_sequence(text)
        sequence = sequence.unsqueeze(0)  # batch dimension 추가
        
        # 임베딩
        embedded = self.embedding(sequence)
        
        # 스피커 임베딩 추가
        speaker_tensor = torch.tensor([speaker_id], device=self.device)
        speaker_emb = self.speaker_embedding(speaker_tensor)
        speaker_emb = speaker_emb.unsqueeze(1).expand(-1, embedded.size(1), -1)
        embedded = embedded + speaker_emb
        
        # 인코더
        encoder_output, _ = self.encoder(embedded)
        
        # 디코더
        decoder_output, _ = self.decoder(encoder_output)
        
        # mel spectrogram 생성
        mel_output = self.mel_proj(decoder_output)
        
        # vocoder를 통한 오디오 생성
        audio = self.vocoder(mel_output.squeeze(0))
        
        return audio
    
    def tts_to_file(self, text: str, speaker_id: int, output_path, 
                   speed: float = 1.0, **kwargs) -> None:
        """텍스트를 파일로 저장"""
        with torch.no_grad():
            audio = self.forward(text, speaker_id)
            
            # 속도 조절
            if speed != 1.0:
                audio = self._adjust_speed(audio, speed)
            
            # 오디오를 파일로 저장
            if hasattr(output_path, 'write'):
                # BytesIO 객체인 경우
                audio_np = audio.cpu().numpy()
                soundfile.write(output_path, audio_np, 22050, format='WAV')
            else:
                # 파일 경로인 경우
                audio_np = audio.cpu().numpy()
                soundfile.write(output_path, audio_np, 22050)
    
    def _adjust_speed(self, audio: torch.Tensor, speed: float) -> torch.Tensor:
        """오디오 속도 조절"""
        if speed == 1.0:
            return audio
        
        # 간단한 리샘플링으로 속도 조절
        original_length = audio.size(-1)
        new_length = int(original_length / speed)
        
        # 리샘플링
        audio_resampled = F.interpolate(
            audio.unsqueeze(0).unsqueeze(0), 
            size=new_length, 
            mode='linear', 
            align_corners=False
        )
        
        return audio_resampled.squeeze(0).squeeze(0)


class SimpleVocoder(nn.Module):
    """간단한 vocoder"""
    
    def __init__(self, mel_channels: int = 80, hidden_dim: int = 512):
        super().__init__()
        
        self.mel_channels = mel_channels
        self.hidden_dim = hidden_dim
        
        # 간단한 upsampling 구조
        self.upsample = nn.Sequential(
            nn.Linear(mel_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel spectrogram을 오디오로 변환"""
        # mel: (time, mel_channels)
        batch_size = mel.size(0)
        
        # 각 시간 스텝을 오디오 샘플로 변환
        audio_samples = []
        for i in range(batch_size):
            sample = self.upsample(mel[i])  # (1,)
            audio_samples.append(sample)
        
        audio = torch.cat(audio_samples, dim=0)  # (time,)
        
        # 간단한 후처리
        audio = torch.tanh(audio) * 0.95  # 클리핑 방지
        
        return audio


def create_mini_tts(ckpt_path: str, device: str = 'auto') -> MiniTTS:
    """Mini TTS 인스턴스 생성"""
    return MiniTTS(ckpt_path, device) 