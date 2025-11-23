# Convert Japanese text to phonemes which is
# compatible with Julius https://github.com/julius-speech/segmentation-kit
import re
import unicodedata

from transformers import AutoTokenizer

from . import punctuation, symbols


from num2words import num2words
from melo.text.ko_dictionary import english_dictionary, etc_dictionary
from anyascii import anyascii
from jamo import hangul_to_jamo

def normalize(text):
    text = text.strip()
    
    # 인터넷 슬랭/줄임말 처리 (자음 처리보다 먼저!)
    slang_map = {
        'ㅇㅈ': '인정',
        'ㄹㅇ': '레알',
        # 'ㅇㅇ': '응응',
        'ㄴㄴ': '노노',
        'ㅂㅂ': '바이바이',
        'ㄱㅅ': '감사',
        'ㄱㅅㅇ': '감사요',
        # 'ㄱㄱ': '고고',
        'ㅈㅅ': '죄송',
        'ㅅㄱ': '수고',
        'ㅊㅋ': '축하',
        'ㅎㅇ': '하이',
        'ㅂㅇ': '바이',
        'ㄷㄷ': '덜덜',
        'ㅎㄷㄷ': '후덜덜',
        'ㅆㅇㅈ': '쌉인정',
        'ㄱㅊ': '괜찮',
        'ㅇㅋ': '오케이',
        'ㄱㄷ': '기달',
        'ㅈㄱㅊㅇ': '정글차이',
        'ㅈㄱㄴ': '제곧내',
        'ㅇㄷ': '어디',
        'ㅁㅊ': '미친',
        'ㅅㅂ': '시발',
        'ㅈㄴ': '존나',
        'ㅆㅂ': '씨발',
        'ㄲㅂ': '까비',
        'ㅄ': '병신',
        'ㅂㅅ': '병신',
        'ㅅㅌㅊ': '상타치',
        'ㅎㅌㅊ': '하타치',
        'ㄴㅇㅅ': '노양심',
        'ㅇㄱㄹㅇ': '이거레알',
        'ㅇㅉ': '어쩔',
        'ㅈㅇ': '존예',
        'ㅈㅈ': '지지',
        'ㅉㅉ': '쯧쯧',
        'ㄱㅇㄷ': '개이득',
        'ㅇㅅㅇ': '응슷응',
        # 'ㅎㅎ': '하하',
        # 'ㅋㅋ': '크크',
        # 'ㅠㅠ': '유유',
        # 'ㅜㅜ': '우우'
    }
    
    # 슬랭 변환 (긴 것부터 먼저 처리)
    for slang in sorted(slang_map.keys(), key=len, reverse=True):
        text = text.replace(slang, slang_map[slang])
    
    # 한글 자음 단독 사용 처리 (슬랭 처리 후!)
    consonant_map = {
        'ㄱ': '기역', 'ㄴ': '니은', 'ㄷ': '디귿', 'ㄹ': '리을',
        'ㅁ': '미음', 'ㅂ': '비읍', 'ㅅ': '시옷', 'ㅇ': '이응',
        'ㅈ': '지읒', 'ㅊ': '치읓', 'ㅋ': '키읔', 'ㅌ': '티읕',
        'ㅍ': '피읖', 'ㅎ': '히읗',
        'ㄲ': '쌍기역', 'ㄸ': '쌍디귿', 'ㅃ': '쌍비읍', 
        'ㅆ': '쌍시옷', 'ㅉ': '쌍지읒'
    }
    
    # 웃음 표현 특별 처리
    # ㅋㅋㅋ -> 크크크, ㅎㅎㅎ -> 흐흐흐
    text = re.sub(r'ㅋ+', lambda m: '크' * len(m.group(0)), text)
    text = re.sub(r'ㅎ+', lambda m: '하' * len(m.group(0)), text)
    text = re.sub(r'ㅠ+', lambda m: '유' * len(m.group(0)), text)
    text = re.sub(r'ㅜ+', lambda m: '우' * len(m.group(0)), text)
    text = re.sub(r'ㅇ+', lambda m: '응' * len(m.group(0)), text)
    text = re.sub(r'ㄱ+', lambda m: '고' * len(m.group(0)), text)


    # 나머지 단독 자음 처리
    for consonant, reading in consonant_map.items():
        if consonant not in ['ㅋ', 'ㅎ','ㅇ','ㄱ']:  # 이미 처리한 것 제외
            text = text.replace(consonant, reading)
    
    # 한글 모음 단독 사용 처리
    vowel_map = {
        'ㅏ': '아', 'ㅑ': '야', 'ㅓ': '어', 'ㅕ': '여',
        'ㅗ': '오', 'ㅛ': '요', 'ㅜ': '우', 'ㅠ': '유',
        'ㅡ': '으', 'ㅣ': '이', 'ㅐ': '애', 'ㅒ': '얘',
        'ㅔ': '에', 'ㅖ': '예', 'ㅘ': '와', 'ㅙ': '왜',
        'ㅚ': '외', 'ㅝ': '워', 'ㅞ': '웨', 'ㅟ': '위',
        'ㅢ': '의'
    }
    
    for vowel, reading in vowel_map.items():
        if vowel not in ['ㅠ', 'ㅜ']:  # 이미 처리한 것 제외
            text = text.replace(vowel, reading)
    
    # 연속된 느낌표/물음표 처리
    text = re.sub(r'!+', '!', text)  # 여러 개의 !를 하나로
    text = re.sub(r'\?+', '?', text)  # 여러 개의 ?를 하나로
    text = re.sub(r'\.+', '.', text)  # 여러 개의 .를 하나로
    
    # "점"을 "쩜"으로 변환 (소수점 표현)
    text = text.replace('점', '쩜')
    
    # 소수점 숫자를 한글로 먼저 변환하여 분리 방지
    def convert_decimal_early(match):
        number = match.group(0)
        parts = number.split('.')
        
        # 숫자를 한글로 변환하는 간단한 함수
        def simple_num_to_korean(num):
            if num == 0:
                return '영'
            digits = ['', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구']
            positions = ['', '십', '백', '천']
            
            num_str = str(num)
            result = ''
            
            for i, digit in enumerate(num_str):
                d = int(digit)
                if d != 0:
                    pos = len(num_str) - i - 1
                    if pos < 4:
                        if d == 1 and pos > 0:
                            result += positions[pos]
                        else:
                            result += digits[d] + positions[pos]
                    else:
                        result += digits[d]
            
            return result
        
        # 정수 부분
        integer_part = int(parts[0])
        result = simple_num_to_korean(integer_part) if integer_part > 0 else '영'
        
        # 소수 부분
        if len(parts) > 1 and parts[1]:
            result += '쩜'
            decimal_digits = ['영', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구']
            for digit in parts[1]:
                result += decimal_digits[int(digit)]
        
        return result
    
    # 소수점 숫자를 먼저 한글로 변환
    text = re.sub(r'\d+\.\d+', convert_decimal_early, text)
    
    # 줄바꿈이나 공백으로 분리된 숫자 처리
    # 예: "1,299,\n000" -> "1,299,000" 또는 "1,299, 000" -> "1,299,000"
    text = re.sub(r'(\d+,)\s+(\d+)', r'\1\2', text)
    text = re.sub(r'(\d+)\s+,\s*(\d+)', r'\1,\2', text)
    
    # 끝에 쉼표가 있는 숫자 처리 (예: "1299," -> "1299")
    text = re.sub(r'(\d+),$', r'\1', text)
    
    # 숫자 내의 모든 쉼표 제거 (예: "1,299,000" -> "1299000")
    def remove_number_commas(match):
        return match.group(0).replace(',', '')
    text = re.sub(r'\d{1,3}(?:,\d{3})+', remove_number_commas, text)
    
    # 특수문자 처리 (지원되지 않는 문자 제거 또는 변환)
    text = text.replace('&', ' 그리고 ')  # & -> and로 변환
    text = text.replace('@', ' ')   # @ -> at으로 변환
    text = text.replace('#', ' ')      # # 제거
    text = text.replace('$', ' ')      # $ 제거
    text = text.replace('%', ' 퍼센트 ')  # % -> percent로 변환
    text = text.replace('^', ' ')      # ^ 제거
    text = text.replace('*', ' ')      # * 제거
    text = text.replace('_', ' ')      # _ 제거
    text = text.replace('=', ' ')      # = 제거
    text = text.replace('+', ' 더하기 ') # + -> plus로 변환
    text = text.replace('|', ' ')      # | 제거
    text = text.replace('\\', ' ')     # \ 제거
    text = text.replace('/', ' ')      # / 제거
    text = text.replace('<', ' ')      # < 제거
    text = text.replace('>', ' ')      # > 제거
    text = text.replace('[', ' ')      # [ 제거
    text = text.replace(']', ' ')      # ] 제거
    text = text.replace('{', ' ')      # { 제거
    text = text.replace('}', ' ')      # } 제거
    text = text.replace('~', ' ')      # ~ 제거
    text = text.replace('`', ' ')      # ` 제거
    text = text.replace(';', ' ')      # ; 제거
    
    # 괄호 안의 영어 대문자를 읽을 수 있도록 변환
    def process_parentheses(match):
        content = match.group(1)
        # 영어 대문자를 한 글자씩 분리
        if content.isupper():
            return ' '.join(content) + ' '
        return content + ' '
    
    # 괄호 안의 내용을 처리 (제거하지 않고 변환)
    text = re.sub(r'\(([^)]+)\)', process_parentheses, text)
    
    # 따옴표 제거
    text = text.replace('"', '').replace("'", '')
    
    # 영어 대문자를 한 글자씩 읽도록 변환
    def replace_uppercase(match):
        letters = match.group(0)
        # 연속된 대문자를 공백으로 분리
        return ' '.join(letters)
    
    # 연속된 영어 대문자를 찾아서 공백으로 분리
    text = re.sub(r'[A-Z]{2,}', replace_uppercase, text)
    
    # 쉼표가 포함된 숫자를 임시로 다른 문자로 대체 (이미 위에서 처리했으므로 제거)
    # text = re.sub(r'(\d+),(\d+)', r'\1\2', text)
    text = re.sub("[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]", "", text)
    text = normalize_with_dictionary(text, etc_dictionary)
    text = normalize_english(text)
    
    text = convert_number_to_korean(text)
    text = text.lower()
    return text


def normalize_with_dictionary(text, dic):
    if any(key in text for key in dic.keys()):
        pattern = re.compile("|".join(re.escape(key) for key in dic.keys()))
        return pattern.sub(lambda x: dic[x.group()], text)
    return text


def normalize_english(text):
    def fn(m):
        word = m.group()
        if word in english_dictionary:
            return english_dictionary.get(word)
        return word

    text = re.sub("([A-Za-z]+)", fn, text)
    return text


def convert_number_to_korean(text):
    # 먼저 number_to_korean_simple 함수 정의
    def number_to_korean_simple(num):
        """간단한 숫자를 한글로 변환"""
        units = ['', '만', '억', '조', '경']
        digits = ['', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구']
        positions = ['', '십', '백', '천']
        
        if num == 0:
            return '영'
            
        result = []
        unit_index = 0
        
        while num > 0:
            chunk = num % 10000
            if chunk > 0:
                chunk_str = ''
                chunk_digits = str(chunk).zfill(4)  # 4자리로 맞춤
                
                for i in range(4):
                    digit = chunk_digits[i]
                    if digit != '0':
                        pos_index = 3 - i
                        # 천의 자리가 1일 때는 '일천'이 아닌 '천'으로
                        if digit == '1' and pos_index == 3:
                            chunk_str += '천'
                        # 십의 자리가 1일 때는 '일십'이 아닌 '십'으로
                        elif digit == '1' and pos_index == 1:
                            chunk_str += '십'
                        # 백의 자리가 1일 때는 '일백'이 아닌 '백'으로
                        elif digit == '1' and pos_index == 2:
                            chunk_str += '백'
                        else:
                            chunk_str += digits[int(digit)] + positions[pos_index]
                
                if unit_index > 0:
                    chunk_str += units[unit_index]
                result.append(chunk_str)
            num //= 10000
            unit_index += 1
            
        return ''.join(reversed(result))
    
    # 소수점 숫자 처리 (예: 3.5 -> 삼쩜오)
    def convert_decimal(match):
        number = match.group(0)
        parts = number.split('.')
        
        # 정수 부분
        integer_part = int(parts[0])
        result = number_to_korean_simple(integer_part) if integer_part > 0 else '영'
        
        # 소수 부분
        if len(parts) > 1 and parts[1]:
            result += '쩜'
            decimal_digits = ['영', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구']
            for digit in parts[1]:
                result += decimal_digits[int(digit)]
        
        return result
    
    # 소수점 숫자 먼저 처리
    text = re.sub(r'\d+\.\d+', convert_decimal, text)
    
    # 시간 표현 처리 (예: 3시 -> 세시)
    def convert_time(match):
        hour = int(match.group(1))
        time_words = {
            1: '한', 2: '두', 3: '세', 4: '네', 5: '다섯',
            6: '여섯', 7: '일곱', 8: '여덟', 9: '아홉', 10: '열',
            11: '열한', 12: '열두'
        }
        if hour in time_words:
            return time_words[hour] + '시'
        elif hour <= 24:
            return number_to_korean_simple(hour) + '시'
        return match.group(0)
    
    # 분 표현 처리 (예: 30분 -> 삼십분)
    def convert_minute(match):
        minute = int(match.group(1))
        return number_to_korean_simple(minute) + '분'
    
    # 시간 패턴 먼저 처리
    text = re.sub(r'(\d+)시', convert_time, text)
    text = re.sub(r'(\d+)분', convert_minute, text)
    
    # 숫자 패턴 찾기 (쉼표가 포함된 숫자도 포함)
    number_pattern = re.compile(r'(\d+(?:,\d+)*)')
    
    def number_to_korean(match):
        # 쉼표 제거 후 숫자로 변환
        num = int(match.group(1).replace(',', ''))
        return number_to_korean_simple(num)
    
    # 숫자를 한글로 변환
    text = number_pattern.sub(number_to_korean, text)
    return text


g2p_kr = None
def korean_text_to_phonemes(text, character: str = "hangeul") -> str:
    """
    The input and output values look the same, but they are different in Unicode.
    example :
        input = '하늘' (Unicode : \ud558\ub298), (하 + 늘)
        output = '하늘' (Unicode :\u1112\u1161\u1102\u1173\u11af), (ᄒ + ᅡ + ᄂ + ᅳ + ᆯ)
    """
    global g2p_kr  # pylint: disable=global-statement
    if g2p_kr is None:
        from g2pkk import G2p
        g2p_kr = G2p()

    # 특수문자 처리
    text = re.sub(r'[<>]', '', text)  # < > 특수문자 제거

    if character == "english":
        from anyascii import anyascii
        text = normalize(text)
        text = g2p_kr(text)
        text = anyascii(text)
        return text

    text = normalize(text)
    text = g2p_kr(text)
    text = list(hangul_to_jamo(text))  # '하늘' --> ['ᄒ', 'ᅡ', 'ᄂ', 'ᅳ', 'ᆯ']
    return "".join(text)

def text_normalize(text):
    # res = unicodedata.normalize("NFKC", text)
    # res = japanese_convert_numbers_to_words(res)
    # # res = "".join([i for i in res if is_japanese_character(i)])
    # res = replace_punctuation(res)
    text = normalize(text)
    return text


def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word



# tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v3')

model_id = 'kykim/bert-kor-base'
tokenizer = AutoTokenizer.from_pretrained(model_id)

def g2p(norm_text):
    tokenized = tokenizer.tokenize(norm_text)
    phs = []
    ph_groups = []
    for t in tokenized:
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            ph_groups[-1].append(t.replace("#", ""))
    word2ph = []
    for group in ph_groups:
        text = ""
        for ch in group:
            text += ch
        if text == '[UNK]':
            phs += ['_']
            word2ph += [1]
            continue
        elif text in punctuation:
            phs += [text]
            word2ph += [1]
            continue
        # import pdb; pdb.set_trace()
        # phonemes = japanese_text_to_phonemes(text)
        # text = g2p_kr(text)
        phonemes = korean_text_to_phonemes(text)
        # import pdb; pdb.set_trace()
        # # phonemes = [i for i in phonemes if i in symbols]
        # for i in phonemes:
        #     assert i in symbols, (group, norm_text, tokenized, i)
        phone_len = len(phonemes)
        word_len = len(group)

        aaa = distribute_phone(phone_len, word_len)
        assert len(aaa) == word_len
        word2ph += aaa

        phs += phonemes
    phones = ["_"] + phs + ["_"]
    tones = [0 for i in phones]
    word2ph =  [1] + word2ph + [1]
    assert len(word2ph) == len(tokenized) + 2
    return phones, tones, word2ph

def get_bert_feature(text, word2ph, device='cuda'):
    from . import japanese_bert
    return japanese_bert.get_bert_feature(text, word2ph, device=device, model_id=model_id)


if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")
    from text.symbols import symbols
    text = "전 제 일의 가치와 폰타인 대중들이 한 일의 의미를 잘 압니다. 앞으로도 전 제 일에 자부심을 갖고 살아갈 겁니다"
    import json

    # genshin_data = json.load(open('/data/zwl/workspace/StarRail_Datasets/Index & Scripts/Index/1.3/Korean.json'))
    genshin_data = json.load(open('/data/zwl/workspace/Genshin_Datasets/Index & Script/AI Hobbyist Version/Index/4.1/KR_output.json'))
    from tqdm import tqdm
    new_symbols = []
    for key, item in tqdm(genshin_data.items()):
        texts = item.get('voiceContent', '')
        if isinstance(texts, list):
            texts = ','.join(texts)
        if texts is None:
            continue
        if len(texts) == 0:
            continue

        text = text_normalize(text)
        phones, tones, word2ph = g2p(text)
        bert = get_bert_feature(text, word2ph)
        import  pdb; pdb.set_trace()
        for ph in phones:
            if ph not in symbols and ph not in new_symbols:
                new_symbols.append(ph)
                print('update!, now symbols:')
                print(new_symbols)
                with open('korean_symbol.txt', 'w') as f:
                    f.write(f'{new_symbols}')

        

# if __name__ == '__main__':
#     from pykakasi import kakasi
#     # Initialize kakasi object
#     kakasi = kakasi()

#     # Set options for converting Chinese characters to Katakana
#     kakasi.setMode("J", "H")  # Chinese to Katakana
#     kakasi.setMode("K", "H")  # Hiragana to Katakana

#     # Convert Chinese characters to Katakana
#     conv = kakasi.getConverter()
#     katakana_text = conv.do('ええ、僕はおきなと申します。こちらの小さいわらべは杏子。ご挨拶が遅れてしまいすみません。あなたの名は?')  # Replace with your Chinese text

#     print(katakana_text)  # Output: ニーハオセカイ