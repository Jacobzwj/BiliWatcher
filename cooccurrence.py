import os
import re
import pandas as pd
import jieba
from collections import Counter
from itertools import combinations
from importlib import resources


def load_stopwords(file_path: str):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return []


def _load_default_stopwords() -> set[str]:
    """加载内置资源中的默认停用词（资源优先，找不到则回退到根目录文件）。"""
    zh_lines: list[str] = []
    en_lines: list[str] = []
    # 1) 资源包优先
    try:
        with resources.files('resources').joinpath('stopwords_zh.txt').open('r', encoding='utf-8') as f:
            zh_lines = [ln.strip() for ln in f if ln.strip()]
    except Exception:
        pass
    try:
        with resources.files('resources').joinpath('stopwords_en.txt').open('r', encoding='utf-8') as f:
            en_lines = [ln.strip() for ln in f if ln.strip()]
    except Exception:
        pass
    # 2) 回退到根目录文件
    if not zh_lines:
        zh_lines = load_stopwords('clearwords.txt')
    if not en_lines:
        en_lines = load_stopwords('english_stopwords.txt')
    return set(w.lower() for w in (zh_lines + en_lines))


def build_cooccurrence_csv(
    input_csv: str,
    output_csv: str,
    chinese_stop_path: str = None,
    english_stop_path: str = None,
    stop_path: str = None,
    append_stopwords: list[str] = None,
    extra_custom_words: list[str] = None,
    words_to_remove: list[str] = None,
    top_n: int = 100,
) -> str:
    df = pd.read_csv(input_csv)
    df['评论内容'] = df['评论内容'].astype(str)
    df['评论内容'] = df['评论内容'].str.replace(r'\[.*?\]', '', regex=True)
    df['评论内容'] = df['评论内容'].str.replace(r'^回复 *@[^:：]+[:：]\s*', '', regex=True)

    # 默认内置集合
    stopwords = _load_default_stopwords()
    # 可选覆盖：外部中文/英文路径 + 单一上传文件
    for p in [chinese_stop_path, english_stop_path, stop_path]:
        if p:
            stopwords |= set(w.lower() for w in load_stopwords(p))
    if append_stopwords:
        stopwords |= set(w.lower() for w in append_stopwords if w)

    for w in extra_custom_words or []:
        try:
            jieba.add_word(w)
        except Exception:
            pass

    def tokenize_and_filter(text: str) -> list[str]:
        words = list(jieba.cut(text))
        out = []
        for w in words:
            wl = str(w).strip().lower()
            if not wl or wl == '\n' or wl.isnumeric() or wl in stopwords:
                continue
            # 过滤纯符号，至少包含一个中英文字符或数字
            if not re.search(r'[\u4e00-\u9fa5A-Za-z0-9]', wl):
                continue
            out.append(wl)
        return out

    df['text_cleaned'] = df['评论内容'].apply(tokenize_and_filter)
    df['text_string'] = df['text_cleaned'].apply(lambda x: ' '.join(x))

    counter = Counter()
    for content in df['text_string']:
        words_set = set(content.split()) if isinstance(content, str) else set()
        for a, b in combinations(sorted(words_set), 2):
            counter[(a, b)] += 1

    co_df = pd.DataFrame([(s, t, w) for (s, t), w in counter.items()], columns=['source', 'target', 'weight'])
    if co_df.empty:
        co_df = pd.DataFrame(columns=['source', 'target', 'weight'])
    co_df = co_df.sort_values('weight', ascending=False)

    default_remove = ['回复', '没', '_', 'doge', '说', '还', '只能', '会', '亱', '才能', '次', '没有', '现在', 'libo', '里', '觉得', '这种', '已经', '不会', '出来', '应该', '直接']
    words_to_remove = words_to_remove or default_remove
    if not co_df.empty:
        mask = ~(co_df['source'].isin(words_to_remove) | co_df['target'].isin(words_to_remove))
        co_df = co_df[mask]

    if top_n and top_n > 0:
        co_df = co_df.head(top_n)

    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    co_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    return output_csv


