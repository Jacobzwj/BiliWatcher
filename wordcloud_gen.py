import os
from collections import Counter
from typing import List, Tuple
import re

import pandas as pd
import jieba
from importlib import resources
from matplotlib import font_manager as fm
from wordcloud import WordCloud


def _load_lines_from_resource(pkg: str, name: str) -> List[str]:
    try:
        with resources.files(pkg).joinpath(name).open('r', encoding='utf-8') as f:
            return [ln.strip() for ln in f if ln.strip()]
    except Exception:
        return []


def _load_default_stopwords() -> set[str]:
    zh = _load_lines_from_resource('resources', 'stopwords_zh.txt')
    en = _load_lines_from_resource('resources', 'stopwords_en.txt')
    if not zh:
        zh = _load_lines_from_path('clearwords.txt')
    if not en:
        en = _load_lines_from_path('english_stopwords.txt')
    return set(w.lower() for w in (zh + en))


def _load_lines_from_path(path: str) -> List[str]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return [ln.strip() for ln in f if ln.strip()]
    except Exception:
        return []


def _choose_font_path() -> str | None:
    candidates = [
        'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Source Han Sans CN',
        'PingFang SC', 'Hiragino Sans GB', 'WenQuanYi Micro Hei', 'DengXian'
    ]
    for name in candidates:
        try:
            prop = fm.FontProperties(family=name)
            path = fm.findfont(prop, fallback_to_default=False)
            if path and os.path.exists(path):
                return path
        except Exception:
            continue
    return None


def build_word_frequencies(
    input_csv: str,
    chinese_stop_path: str | None = None,
    english_stop_path: str | None = None,
    stop_path: str | None = None,
    append_stopwords: List[str] | None = None,
    extra_custom_words: List[str] | None = None,
    default_remove: List[str] | None = None,
    top_n: int = 200,
) -> List[Tuple[str, int]]:
    df = pd.read_csv(input_csv)
    df['评论内容'] = df['评论内容'].astype(str)
    df['评论内容'] = df['评论内容'].str.replace(r'\[.*?\]', '', regex=True)
    df['评论内容'] = df['评论内容'].str.replace(r'^回复 *@[^:：]+[:：]\s*', '', regex=True)

    stopwords = _load_default_stopwords()
    for p in [chinese_stop_path, english_stop_path, stop_path]:
        if p:
            stopwords |= set(w.lower() for w in _load_lines_from_path(p))
    if append_stopwords:
        stopwords |= set(w.lower() for w in append_stopwords if w)

    for w in extra_custom_words or []:
        try:
            jieba.add_word(w)
        except Exception:
            pass

    counter = Counter()
    for text in df['评论内容']:
        for w in jieba.cut(str(text)):
            wl = str(w).strip().lower()
            if not wl:
                continue
            if wl.isnumeric():
                continue
            if wl in stopwords:
                continue
            # 过滤纯标点/符号：必须包含至少一个中英文字符或数字
            if not re.search(r'[\u4e00-\u9fa5A-Za-z0-9]', wl):
                continue
            counter[wl] += 1

    words = counter.most_common()
    if default_remove:
        words = [(w, c) for (w, c) in words if w not in set(default_remove)]
    if top_n and top_n > 0:
        words = words[:top_n]
    return words


def generate_wordcloud(words: List[Tuple[str, int]], out_png: str, bg_color: str = 'white') -> str:
    font_path = _choose_font_path()
    wc = WordCloud(
        width=1200,
        height=800,
        background_color=bg_color,
        font_path=font_path,
        prefer_horizontal=0.9,
        max_words=500,
        colormap='tab20'
    )
    wc.generate_from_frequencies(dict(words))
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    wc.to_file(out_png)
    return out_png


