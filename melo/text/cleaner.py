from . import chinese, english, chinese_mix, korean, french, spanish

# Conditional import for Japanese to avoid dependency issues
try:
    from . import japanese
    JAPANESE_AVAILABLE = True
except ImportError:
    print("Warning: Japanese text processing not available (mecab-python3 and unidic-lite required)")
    JAPANESE_AVAILABLE = False
    japanese = None

from . import cleaned_text_to_sequence
import copy

language_module_map = {"ZH": chinese, "EN": english, 'ZH_MIX_EN': chinese_mix, 'KR': korean,
                    'FR': french, 'SP': spanish, 'ES': spanish}

# Add Japanese to the map only if available
if JAPANESE_AVAILABLE:
    language_module_map["JP"] = japanese


def clean_text(text, language):
    if language == "JP" and not JAPANESE_AVAILABLE:
        raise ImportError("Japanese text processing is not available. Please install mecab-python3 and unidic-lite.")
    
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    # Pass tokenized text to avoid double normalization in English
    if language == 'EN':
        tokenized = language_module.tokenizer.tokenize(norm_text)
        phones, tones, word2ph = language_module.g2p(text=None, tokenized=tokenized)
    else:
        phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph


def clean_text_bert(text, language, device=None):
    if language == "JP" and not JAPANESE_AVAILABLE:
        raise ImportError("Japanese text processing is not available. Please install mecab-python3 and unidic-lite.")
    
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    # Pass tokenized text to avoid double normalization in English
    if language == 'EN':
        tokenized = language_module.tokenizer.tokenize(norm_text)
        phones, tones, word2ph = language_module.g2p(text=None, tokenized=tokenized)
    else:
        phones, tones, word2ph = language_module.g2p(norm_text)
    
    word2ph_bak = copy.deepcopy(word2ph)
    for i in range(len(word2ph)):
        word2ph[i] = word2ph[i] * 2
    word2ph[0] += 1
    bert = language_module.get_bert_feature(norm_text, word2ph, device=device)
    
    return norm_text, phones, tones, word2ph_bak, bert


def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)


if __name__ == "__main__":
    pass