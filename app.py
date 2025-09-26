import argparse
import os
import re
import time
import io
import base64
import zipfile
from typing import List, Tuple
import importlib.util

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib import font_manager as fm
import colorsys
import matplotlib.patches as mpatches
import streamlit as st
import importlib.util as _il_util
from crawler import get_video_id, fetch_comments, save_comments_to_csv, headers as crawler_headers, sanitize_filename as crawler_sanitize
try:
    from crawler import get_last_fetch_info as crawler_last_info
except Exception:
    crawler_last_info = None
from cooccurrence import build_cooccurrence_csv
from network import draw_network
from wordcloud_gen import build_word_frequencies, generate_wordcloud

# 可选依赖
try:
    import community as community_louvain  # python-louvain
except Exception:  # pragma: no cover
    community_louvain = None
try:
    from adjustText import adjust_text as _adjust_text
except Exception:  # pragma: no cover
    _adjust_text = None
try:
    import distinctipy as _distinctipy
except Exception:  # pragma: no cover
    _distinctipy = None
try:
    import hsluv as _hsluv
except Exception:  # pragma: no cover
    _hsluv = None

def _load_crawler_module():
    # 向后兼容：已改为直接 import crawler
    return __import__('crawler')


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# --- 字体选择（中英文友好） ---
def choose_font_family(candidates: List[str] = None) -> str:
    # 1) 优先注册并使用 resources 下的中文字体（云端常无中文系统字体）
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        for rel in [
            os.path.join(here, 'resources', 'NotoSansSC-Regular.otf'),
            os.path.join(here, 'resources', 'NotoSansSC-Regular.ttf'),
            os.path.join(here, 'resources', 'SourceHanSansCN-Regular.otf'),
            os.path.join(here, 'resources', 'SourceHanSansCN-Regular.ttf'),
        ]:
            if os.path.exists(rel):
                fm.fontManager.addfont(rel)
                return fm.FontProperties(fname=rel).get_name()
    except Exception:
        pass

    # 2) 退回系统已安装字体
    if candidates is None:
        candidates = [
            'Microsoft YaHei', 'DengXian', 'Noto Sans CJK SC', 'Noto Sans SC',
            'Source Han Sans CN', 'PingFang SC', 'Hiragino Sans GB', 'WenQuanYi Micro Hei',
            'Microsoft YaHei UI', 'SimHei', 'KaiTi', 'Kaiti SC', 'STKaiti',
            'Arial Unicode MS', 'Segoe UI', 'DejaVu Sans'
        ]
    for name in candidates:
        try:
            _ = fm.findfont(mpl.font_manager.FontProperties(family=name), fallback_to_default=False)
            return name
        except Exception:
            continue
    return 'DejaVu Sans'


FONT_FAMILY = choose_font_family()
mpl.rcParams['font.family'] = FONT_FAMILY
mpl.rcParams['axes.unicode_minus'] = False

# --- Cookie 持久化 & 规范化 ---
def _cookie_store_path() -> str:
    try:
        home = os.path.expanduser('~')
        return os.path.join(home, '.bilicomment2network_cookie.txt')
    except Exception:
        return os.path.join(os.getcwd(), '.bilicomment2network_cookie.txt')

def _sanitize_cookie_str(raw: str) -> str:
    if not raw:
        return ''
    s = str(raw).strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")) or (s.startswith('“') and s.endswith('”')):
        s = s[1:-1].strip()
    s = s.replace('\r', ' ').replace('\n', ' ').strip()
    return s

def _read_saved_cookie() -> str:
    try:
        path = _cookie_store_path()
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return _sanitize_cookie_str(f.read())
    except Exception:
        pass
    return ''

def _write_saved_cookie(cookie: str) -> None:
    try:
        path = _cookie_store_path()
        with open(path, 'w', encoding='utf-8') as f:
            f.write(_sanitize_cookie_str(cookie))
    except Exception:
        pass

# --- DeepSeek API Key 持久化 ---
def _api_store_path() -> str:
    try:
        home = os.path.expanduser('~')
        return os.path.join(home, '.bilicomment2network_deepseek_api.txt')
    except Exception:
        return os.path.join(os.getcwd(), '.bilicomment2network_deepseek_api.txt')

def _sanitize_api_key(raw: str) -> str:
    return (str(raw or '')).strip()

def _read_saved_api() -> str:
    try:
        path = _api_store_path()
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return _sanitize_api_key(f.read())
    except Exception:
        pass
    return ''

def _write_saved_api(key: str) -> None:
    try:
        path = _api_store_path()
        with open(path, 'w', encoding='utf-8') as f:
            f.write(_sanitize_api_key(key))
    except Exception:
        pass
def _load_deepseek_summary():
    # 更新为直接导入新模块名
    return __import__('ai_summary')

def _load_comment_summary():
    # 新增：评论文本主题总结模块
    return __import__('comment_summary')


def _get_effective_api_key() -> str:
    """获取可用的 DeepSeek API Key：优先用户输入，其次 secrets 默认。"""
    try:
        default_api = st.secrets.get('DEEPSEEK_API_KEY', '')
    except Exception:
        default_api = ''
    user = (st.session_state.get('user_api_key') or '').strip()
    return user or default_api


def _format_end_info_cn(info: dict, fallback_total: int | None = None) -> str:
    try:
        reason_map = {
            'is_end': '平台返回：已到末尾',
            'max_offset': '已到平台翻页上限',
            'hard_limit': '达到本地安全上限',
            'no_progress': '无新增数据（可能已抓完）',
            'http_error': '网络/HTTP 错误',
            'exception': '请求异常中断',
            'exhausted': '达到最大页数限制',
        }
        method_map = {'wbi': '光标接口（wbi）', 'legacy': '分页接口（旧版）'}
        r = reason_map.get((info or {}).get('end_reason'), '未知')
        m = method_map.get((info or {}).get('method'), (info or {}).get('method', '?'))
        p = (info or {}).get('pages', '?')
        t = (info or {}).get('total_count', fallback_total if fallback_total is not None else '?')
        return f"抓取结束：{r}；接口：{m}；页数：{p}；总计：{t}"
    except Exception:
        return ''


def load_stopwords(file_path: str) -> List[str]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return []


def step1_fetch_to_csv(bv: str, title: str, result_dir: str, on_progress=None, cookie: str = None) -> str:
    """抓取评论，保存到 CSV，返回 CSV 路径。"""
    ensure_dir(result_dir)
    crawler = _load_crawler_module()
    # 覆盖 Cookie（如提供）
    try:
        cookie = _sanitize_cookie_str(cookie)
        if cookie and isinstance(crawler_headers, dict):
            crawler_headers['Cookie'] = cookie
    except Exception:
        pass
    safe_name = crawler_sanitize(title or bv)
    outfile = os.path.join(result_dir, f"{safe_name}.csv")
    start_time = time.time()
    aid = get_video_id(bv)
    comments = fetch_comments(aid, on_progress=on_progress)
    save_comments_to_csv(comments, safe_name)
    # 若用户指定了自定义目录，移动文件
    src = os.path.join('result', f"{safe_name}.csv")
    if os.path.abspath(result_dir) != os.path.abspath('result'):
        if os.path.exists(src):
            try:
                os.replace(src, outfile)
            except Exception:
                # 回退到拷贝
                import shutil
                shutil.copy2(src, outfile)
        else:
            # 如果源文件不存在，仍然以 outfile 为报告路径
            pass
    elapsed = time.time() - start_time
    print(f"[Step1] 抓取完成: {len(comments)} 条，用时 {elapsed:.2f}s，输出: {outfile}")
    return outfile


def step2_comments_to_cooccurrence(
    input_csv: str,
    output_csv: str,
    chinese_stop_path: str = 'clearwords.txt',
    english_stop_path: str = 'english_stopwords.txt',
    extra_custom_words: List[str] = None,
    words_to_remove: List[str] = None,
    top_n: int = 100,
    stop_path: str = None,
    append_stopwords: List[str] = None
) -> str:
    """将评论 CSV 转为共现边列表 CSV，返回输出路径。"""
    import jieba
    from collections import Counter
    from itertools import combinations

    df = pd.read_csv(input_csv)
    # 清洗表情/引用回复
    df['评论内容'] = df['评论内容'].astype(str)
    df['评论内容'] = df['评论内容'].str.replace(r'\[.*?\]', '', regex=True)
    df['评论内容'] = df['评论内容'].str.replace(r'^回复 *@[^:：]+[:：]\s*', '', regex=True)

    # 停用词：默认中文+英文；可选单一文件覆盖；追加手输
    stopwords = set()
    if chinese_stop_path:
        stopwords |= set(w.lower() for w in load_stopwords(chinese_stop_path))
    if english_stop_path:
        stopwords |= set(w.lower() for w in load_stopwords(english_stop_path))
    if stop_path:
        stopwords |= set(w.lower() for w in load_stopwords(stop_path))
    if append_stopwords:
        stopwords |= set(w.lower() for w in append_stopwords if w)

    # 自定义词
    extra_custom_words = extra_custom_words or []
    for w in extra_custom_words:
        try:
            jieba.add_word(w)
        except Exception:
            pass

    def tokenize_and_filter(text: str) -> List[str]:
        words = list(jieba.cut(text))
        return [w for w in words if w and w != '\n' and not w.isnumeric() and w.lower() not in stopwords]

    df['text_cleaned'] = df['评论内容'].apply(tokenize_and_filter)
    df['text_string'] = df['text_cleaned'].apply(lambda x: ' '.join(x))

    # 共现统计（每条评论按集合去重）
    counter = Counter()
    for content in df['text_string']:
        words_set = set(content.split()) if isinstance(content, str) else set()
        for a, b in combinations(sorted(words_set), 2):
            counter[(a, b)] += 1

    co_df = pd.DataFrame([(s, t, w) for (s, t), w in counter.items()], columns=['source', 'target', 'weight'])
    if co_df.empty:
        co_df = pd.DataFrame(columns=['source', 'target', 'weight'])

    co_df = co_df.sort_values('weight', ascending=False)

    # 过滤不需要的词
    default_remove = ['回复', '没', '_', 'doge', '说', '还', '只能', '会', '亱', '才能', '次',
                      '没有', '现在', 'libo', '里', '觉得', '这种', '已经', '不会', '出来', '应该', '直接']
    words_to_remove = words_to_remove or default_remove
    if not co_df.empty:
        mask = ~(co_df['source'].isin(words_to_remove) | co_df['target'].isin(words_to_remove))
        co_df = co_df[mask]

    # Top N
    if top_n and top_n > 0:
        co_df = co_df.head(top_n)

    ensure_dir(os.path.dirname(output_csv) or '.')
    # 使用 UTF-8 带 BOM，避免 Excel 打开乱码
    co_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"[Step2] 共现构建完成: {len(co_df)} 条边，输出: {output_csv}")
    return output_csv


# --- 配色工具 ---
def _desaturate_hex(hex_color: str, target_s: float = 0.35, lightness_shift: float = 0.08, target_l: float = None) -> str:
    r, g, b = mcolors.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s = max(0.0, min(1.0, target_s if target_s is not None else s))
    if target_l is not None:
        l = target_l
    l = max(0.0, min(1.0, l + lightness_shift))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return mcolors.to_hex((r2, g2, b2))


OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']
TOL_12 = ['#4477AA', '#66CCEE', '#228833', '#CCBB44', '#EE6677', '#AA3377', '#BBBBBB', '#2288AA', '#66A61E', '#FFD92F', '#E78AC3', '#A6761D']
TOL_20 = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#661100', '#CC6677', '#882255', '#AA4499', '#DDDDDD', '#1F78B4', '#33A02C', '#FB9A99', '#E31A1C', '#FDBF6F', '#FF7F00', '#CAB2D6', '#6A3D9A', '#B15928']


def _hsluv_palette(n: int, s: float = 60.0, l: float = 62.0) -> List[str]:
    if _hsluv is None:
        colors = []
        for i in range(max(1, n)):
            h = i / max(1, n)
            r, g, b = colorsys.hls_to_rgb(h, l/100.0, s/100.0)
            colors.append(mcolors.to_hex((r, g, b)))
        return colors
    return [_hsluv.hsluv_to_hex([i*360.0/max(1, n), s, l]) for i in range(n)]


def get_community_colors(n: int, theme: str = 'distinct', pastelize: float = 0.2, target_s: float = None, target_l: float = None) -> List[str]:
    theme = (theme or 'auto').lower()
    if theme == 'okabe':
        return OKABE_ITO[:n] if n <= len(OKABE_ITO) else _hsluv_palette(n)
    if theme == 'tol':
        if n <= len(TOL_12):
            return TOL_12[:n]
        if n <= len(TOL_20):
            return TOL_20[:n]
        return _hsluv_palette(n)
    if theme == 'hsluv':
        return _hsluv_palette(n)
    if theme == 'distinct':
        if _distinctipy is not None:
            cols = _distinctipy.get_colors(n, pastelize=pastelize)
            return [_distinctipy.get_hex(c) for c in cols]
        return _hsluv_palette(n)

    # auto
    if n <= 8:
        base = OKABE_ITO[:n]
    elif n <= 12:
        base = TOL_12[:n]
    elif n <= 20:
        base = TOL_20[:n]
    else:
        base = _hsluv_palette(n)

    if target_s is None and target_l is None:
        return base
    return [_desaturate_hex(c, target_s=target_s, lightness_shift=0.0, target_l=target_l) for c in base]


def step3_draw_network(
    edge_csv: str,
    out_png: str,
    layout: str = 'auto',
    min_weight: int = 0,
    seed: int = 42
) -> Tuple[str, dict]:
    if community_louvain is None:
        raise RuntimeError("缺少依赖 python-louvain，请先安装：pip install python-louvain")

    df = pd.read_csv(edge_csv)
    if df.empty:
        raise ValueError('输入的共现数据为空，无法绘图。')

    # 规范化无向边并聚合
    df = df.copy()
    df['a'] = df[['source', 'target']].min(axis=1)
    df['b'] = df[['source', 'target']].max(axis=1)
    df_agg = (
        df.groupby(['a', 'b'], as_index=False)['weight']
          .sum()
          .rename(columns={'a': 'source', 'b': 'target'})
    )
    if min_weight and min_weight > 0:
        df_agg = df_agg[df_agg['weight'] >= min_weight]

    G = nx.from_pandas_edgelist(df_agg, 'source', 'target', edge_attr='weight', create_using=nx.Graph())
    if G.number_of_nodes() == 0:
        raise ValueError('图为空：请检查输入数据是否包含有效的 source/target/weight 列。')

    partition = community_louvain.best_partition(G, weight='weight')

    # 社区感知布局参数
    if layout == 'none':
        intra_mul, inter_mul, k_val, iter_val = 1.0, 1.0, 2.0, 120
    elif layout == 'mild':
        intra_mul, inter_mul, k_val, iter_val = 2.0, 0.5, 2.0, 120
    elif layout == 'strong':
        intra_mul, inter_mul, k_val, iter_val = 4.5, 0.25, 1.5, 200
    else:
        intra_sum = sum(d.get('weight', 1) for _, _, d in G.edges(data=True))
        inter_sum = sum(d.get('weight', 1) for u, v, d in G.edges(data=True) if partition[u] != partition[v])
        ratio = inter_sum / (intra_sum + 1e-9)
        try:
            mod_score = community_louvain.modularity(partition, G, weight='weight')
        except Exception:
            mod_score = 0.0
        strength = min(1.0, max(0.0, 0.6*ratio + 0.4*max(0.0, (0.3 - mod_score)/0.3)))
        intra_mul = 2.0 + strength * (4.0 - 2.0)
        inter_mul = 0.5 - strength * (0.5 - 0.25)
        k_val = 2.0 - strength * (2.0 - 1.5)
        iter_val = int(120 + strength * (220 - 120))

    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1)
        data['layout_weight'] = w * (intra_mul if partition[u] == partition[v] else inter_mul)

    num_communities = max(partition.values()) + 1
    colors = get_community_colors(num_communities, theme='distinct', pastelize=0.2, target_s=None, target_l=None)
    color_map = {node: colors[partition[node]] for node in G.nodes()}
    node_colors = [color_map[n] for n in G.nodes()]

    degrees_unweighted = dict(G.degree())
    size_min, size_max = 200, 3000
    gamma = 1.6
    min_d = min(degrees_unweighted.values()) if degrees_unweighted else 0
    max_d = max(degrees_unweighted.values()) if degrees_unweighted else 1
    if max_d == min_d:
        node_sizes = [(size_min + size_max) / 2 for _ in G.nodes()]
    else:
        node_sizes = []
        for n in G.nodes():
            norm = (degrees_unweighted[n] - min_d) / (max_d - min_d)
            boosted = norm ** gamma
            node_sizes.append(size_min + (size_max - size_min) * boosted)

    pos = nx.spring_layout(G, k=k_val, iterations=iter_val, weight='layout_weight', seed=seed)

    plt.figure(figsize=(16, 10))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.95, linewidths=1.0, edgecolors='white')
    max_weight = df_agg['weight'].max() if not df_agg.empty else 1
    edge_widths = [0.5 + 2 * G[u][v]['weight'] / max_weight for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.35, edge_color='#9aa1a6')
    label_artists = nx.draw_networkx_labels(G, pos, font_size=14, font_family=FONT_FAMILY)
    if _adjust_text is not None:
        try:
            _adjust_text(list(label_artists.values()), expand_text=(1.1, 1.1), expand_points=(1.1, 1.1), force_text=(0.5, 0.5), force_points=0.2, lim=100)
        except Exception:
            pass
    ax = plt.gca()
    legend_patches = [mpatches.Patch(color=colors[i], label=f'聚类 {i+1}') for i in range(num_communities)]
    ax.legend(handles=legend_patches, loc='lower right', frameon=False, fontsize=10, ncol=1)
    plt.title('语义网络图（Louvain社区检测）', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png) or '.')
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()

    # 控制台输出 + 汇总为信息返回
    print(f"[Step3] 发现社区数: {num_communities}")
    communities_detail = []
    for i, color in enumerate(colors):
        community_nodes = [node for node in G.nodes() if partition[node] == i]
        print(f"  社区 {i+1} (颜色: {color}): {', '.join(community_nodes)}")
        communities_detail.append({'index': i+1, 'color': color, 'nodes': community_nodes})
    try:
        modularity_score = community_louvain.modularity(partition, G, weight='weight')
        print(f"  模块度: {modularity_score:.3f}")
    except Exception:
        modularity_score = None

    print(f"[Step3] 绘图完成: {out_png}")
    info = {
        'num_communities': num_communities,
        'communities': communities_detail,
        'modularity': modularity_score
    }
    return out_png, info


def run_pipeline(bv: str, title: str, result_dir: str = 'result', layout: str = 'auto', min_weight: int = 0, top_n: int = 100,
                 extra_words: List[str] = None, chinese_stop: str = 'clearwords.txt', english_stop: str = 'english_stopwords.txt',
                 on_progress=None, cookie: str = None) -> Tuple[str, str, str, dict]:
    crawler = _load_crawler_module()
    safe_name = crawler_sanitize(title or bv)
    raw_csv = step1_fetch_to_csv(bv, safe_name, result_dir, on_progress=on_progress, cookie=cookie)
    processed_csv = os.path.join(result_dir, f"{safe_name}_processed.csv")
    build_cooccurrence_csv(
        input_csv=raw_csv,
        output_csv=processed_csv,
        chinese_stop_path=chinese_stop,
        english_stop_path=english_stop,
        stop_path=None,
        append_stopwords=None,
        extra_custom_words=extra_words or [],
        words_to_remove=None,
        top_n=top_n,
    )
    out_png = os.path.join(result_dir, f"{safe_name}_semantic_network.png")
    out_png, info = draw_network(processed_csv, out_png, layout=layout, min_weight=min_weight, seed=42)
    return raw_csv, processed_csv, out_png, info


def _save_uploaded_to(path: str, file_obj) -> str:
    if file_obj is None:
        return path
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'wb') as f:
        f.write(file_obj.read())
    return path


def render_streamlit_app():
    st.set_page_config(page_title='B站评论语义网络生成器', layout='wide')
    st.title('B站评论语义网络生成器')
    st.caption('输入 BV 号与自定义名称，分步或一键生成语义网络图。')

    # Sidebar 参数
    with st.sidebar:
        st.header('参数设置')
        # DeepSeek API Key：云端默认来自 secrets，用户可覆盖；不直接显示默认值
        try:
            default_api = st.secrets.get('DEEPSEEK_API_KEY', '')
        except Exception:
            default_api = ''
        # 用单独的 session_state 键存放“用户输入的 key”，默认空字符串
        if 'user_api_key' not in st.session_state:
            st.session_state['user_api_key'] = ''
        col_api1, col_api2 = st.columns([1,1])
        with col_api1:
            if st.button('恢复默认 Key', width='stretch'):
                # 恢复为默认：将用户输入清空，实际取用时回落到 secrets
                st.session_state['user_api_key'] = ''
        with col_api2:
            if st.button('清空 Key', width='stretch'):
                st.session_state['user_api_key'] = ''
        # 仅显示“用户输入”的 Key，不显示（也不回显）默认 Key
        st.text_input('DeepSeek API Key（可选，用于AI总结）', type='password', key='user_api_key')
        st.caption('说明：部署者配置的默认 Key 不会在此处回显；若留空则使用默认 Key。不会保存到服务器。')
        # Cookies：不保存到服务器，仅本次会话
        if 'bili_cookie' not in st.session_state:
            st.session_state['bili_cookie'] = ''
        cookie_input = st.text_area('B站 Cookies（每次访问需手动粘贴）', height=90, key='bili_cookie')
        col_ck1, col_ck2 = st.columns([1,1])
        with col_ck1:
            if st.button('清空 Cookies', width='stretch'):
                st.session_state.bili_cookie = ''
                st.info('已清空（仅本次会话，不会保存到服务器）')
        with col_ck2:
            st.write('')
        bv = st.text_input('视频 BV 号', value='', placeholder='例如：BV1xxxxxxx')
        title = st.text_input('自定义名称（文件名）', value='')
        result_dir = st.text_input('输出目录', value='result')
        layout = st.selectbox('布局强度', options=['auto', 'mild', 'strong', 'none'], index=0)
        min_weight = st.number_input('最小边权阈值（绘图前过滤）', min_value=0, value=0, step=1)
        top_n = st.number_input('共现 Top N', min_value=10, max_value=2000, value=100, step=10)
        cmt_topk = st.number_input('评论AI总结 TopK（按点赞）', min_value=1, value=100, step=10)
        st.caption('提示：不设上限。若超过总评论数，将自动按总数截断。')
        extra_words_raw = st.text_area('自定义分词（逗号/空格/换行分隔）', '')
        up_words = st.file_uploader('自定义分词文件（TXT）', type=['txt'], key='extra_words')
        st.markdown('—— 停用词（可输入或上传；若都为空将使用默认文件） ——')
        stopwords_text = st.text_area('停用词（逗号/空格/换行分隔）', '')
        up_stop = st.file_uploader('停用词文件（TXT）', type=['txt'], key='stop_all')

    if 'raw_csv' not in st.session_state:
        st.session_state.raw_csv = ''
    if 'processed_csv' not in st.session_state:
        st.session_state.processed_csv = ''
    if 'out_png' not in st.session_state:
        st.session_state.out_png = ''
    # 确保 AI 文本在重绘后仍可展示
    if 'ai_summary_text' not in st.session_state:
        st.session_state.ai_summary_text = ''
    if 'ai_trigger' not in st.session_state:
        st.session_state.ai_trigger = False
    if 'cmt_ai_text' not in st.session_state:
        st.session_state.cmt_ai_text = ''
    if 'cmt_ai_trigger' not in st.session_state:
        st.session_state.cmt_ai_trigger = False

    col1, col2, col3, col4, col5, col6 = st.columns([1,1,1,1,1,1])

    def parse_extra_words(text: str) -> List[str]:
        if not text:
            return []
        parts = re.split(r'[\s,;，；]+', text)
        return [p.strip() for p in parts if p.strip()]

    with col1:
        if st.button('Step 1 · 抓取评论', width='stretch'):
            if not bv:
                st.warning('请先输入 BV 号')
            else:
                # 展示运行时间与累计数量
                counter = {'n': 0}
                info_box = st.empty()

                def _on_progress(delta: int):
                    counter['n'] += delta
                    elapsed_now = time.time() - start_time
                    info_box.markdown(f"**已爬取评论数: {counter['n']} ｜ 已运行时间: {elapsed_now:.1f}s**")

                with st.spinner('正在抓取评论...'):
                    try:
                        # 包装抓取：在本进程暂存一个替代函数
                        crawler = _load_crawler_module()
                        safe_name = crawler.sanitize_filename(title or bv)
                        ensure_dir(result_dir)
                        outfile = os.path.join(result_dir, f"{safe_name}.csv")

                        start_time = time.time()
                        aid = crawler.get_video_id(bv)
                        # 覆盖 headers 中的 Cookie（若提供）
                        try:
                            user_cookie = _sanitize_cookie_str(st.session_state.get('bili_cookie', ''))
                            if user_cookie and hasattr(crawler, 'headers') and isinstance(crawler.headers, dict):
                                crawler.headers = dict(crawler.headers)
                                crawler.headers['Cookie'] = user_cookie
                        except Exception:
                            pass
                        comments = crawler.fetch_comments(aid, on_progress=_on_progress)
                        crawler.save_comments_to_csv(comments, safe_name)
                        src = os.path.join('result', f"{safe_name}.csv")
                        if os.path.abspath(result_dir) != os.path.abspath('result') and os.path.exists(src):
                            try:
                                os.replace(src, outfile)
                            except Exception:
                                import shutil
                                shutil.copy2(src, outfile)

                        elapsed = time.time() - start_time
                        raw_csv = outfile
                        st.session_state.raw_csv = raw_csv
                        st.success(f'已完成抓取：{raw_csv}（总数 {len(comments)}，耗时 {elapsed:.2f}s）')
                        # 展示抓取结束原因（成功或截断）
                        try:
                            if hasattr(crawler, 'get_last_fetch_info'):
                                _info = crawler.get_last_fetch_info()
                            elif callable(crawler_last_info):
                                _info = crawler_last_info()
                            else:
                                _info = {}
                            if _info:
                                _reason_map = {
                                    'is_end': '平台返回：已到末尾',
                                    'max_offset': '已到平台翻页上限',
                                    'hard_limit': '达到本地安全上限',
                                    'no_progress': '无新增数据（可能已抓完）',
                                    'http_error': '网络/HTTP 错误',
                                    'exception': '请求异常中断',
                                    'exhausted': '达到最大页数限制',
                                }
                                _method_map = {'wbi': '光标接口（wbi）', 'legacy': '分页接口（旧版）'}
                                st.caption(f"抓取结束：{_reason_map.get(_info.get('end_reason'),'未知')}；接口：{_method_map.get(_info.get('method'), _info.get('method','?'))}；页数：{_info.get('pages','?')}；总计：{_info.get('total_count', len(comments))}")
                        except Exception:
                            pass
                    except Exception as e:
                        st.error(f'抓取失败：{e}')

    with col2:
        if st.button('Step 2 · 构建共现', width='stretch'):
            if not st.session_state.raw_csv:
                st.warning('请先完成 Step 1 或提供已有评论CSV')
            else:
                with st.spinner('正在构建词共现...'):
                    try:
                        # 停用词：合并输入与上传
                        cn_path = 'clearwords.txt'
                        en_path = 'english_stopwords.txt'
                        stop_path = None
                        if up_stop is not None:
                            stop_path = _save_uploaded_to(os.path.join(result_dir, 'uploaded_stopwords.txt'), up_stop)

                        crawler = _load_crawler_module()
                        safe_name = crawler.sanitize_filename(title or bv)
                        processed_csv = os.path.join(result_dir, f"{safe_name}_processed.csv")
                        # 组合自定义分词：文本输入 + 上传文件
                        extra_words_list = parse_extra_words(extra_words_raw)
                        try:
                            if up_words is not None:
                                words_path = _save_uploaded_to(os.path.join(result_dir, 'uploaded_extra_words.txt'), up_words)
                                extra_words_list += load_stopwords(words_path)
                        except Exception:
                            pass
                        step2_comments_to_cooccurrence(
                            input_csv=st.session_state.raw_csv,
                            output_csv=processed_csv,
                            chinese_stop_path=cn_path,
                            english_stop_path=en_path,
                            stop_path=stop_path,
                            append_stopwords=parse_extra_words(stopwords_text),
                            extra_custom_words=extra_words_list,
                            words_to_remove=None,
                            top_n=int(top_n)
                        )
                        st.session_state.processed_csv = processed_csv
                        # 读取边数并展示
                        try:
                            _df = pd.read_csv(processed_csv)
                            st.info(f"共现边数：{len(_df)}")
                        except Exception:
                            pass
                        st.success(f'已生成共现：{processed_csv}')
                    except Exception as e:
                        st.error(f'共现构建失败：{e}')

    with col3:
        if st.button('Step 3 · 词云', width='stretch'):
            if not st.session_state.raw_csv:
                st.warning('请先完成 Step 1 或提供已有评论CSV')
            else:
                with st.spinner('正在生成词云...'):
                    try:
                        default_remove = ['回复', '没', '_', 'doge', '说', '还', '只能', '会', '亱', '才能', '次', '没有', '现在', 'libo', '里', '觉得', '这种', '已经', '不会', '出来', '应该', '直接']
                        extra_words_list = parse_extra_words(extra_words_raw)
                        try:
                            if up_words is not None:
                                words_path = _save_uploaded_to(os.path.join(result_dir, 'uploaded_extra_words.txt'), up_words)
                                extra_words_list += load_stopwords(words_path)
                        except Exception:
                            pass
                        cn_path = 'clearwords.txt'
                        en_path = 'english_stopwords.txt'
                        stop_path = None
                        if up_stop is not None:
                            stop_path = _save_uploaded_to(os.path.join(result_dir, 'uploaded_stopwords.txt'), up_stop)
                        freqs = build_word_frequencies(
                            input_csv=st.session_state.raw_csv,
                            chinese_stop_path=cn_path,
                            english_stop_path=en_path,
                            stop_path=stop_path,
                            append_stopwords=parse_extra_words(stopwords_text),
                            extra_custom_words=extra_words_list,
                            default_remove=default_remove,
                            top_n=300,
                        )
                        crawler = _load_crawler_module()
                        safe_name = crawler.sanitize_filename(title or bv)
                        wc_png = os.path.join(result_dir, f"{safe_name}_wordcloud.png")
                        generate_wordcloud(freqs, wc_png)
                        st.session_state['word_freqs'] = freqs
                        st.session_state['wordcloud_png'] = wc_png
                        # 保存词频为 CSV（完整列表）
                        try:
                            import pandas as _pd
                            df_all = _pd.DataFrame(freqs, columns=['词语', '词频'])
                            df_all.index = _pd.RangeIndex(start=1, stop=len(df_all)+1, step=1)
                            df_all.index.name = 'Top'
                            wc_csv = os.path.join(result_dir, f"{safe_name}_wordfreq.csv")
                            df_all.to_csv(wc_csv, index=False, encoding='utf-8-sig')
                            st.session_state['wordfreq_csv'] = wc_csv
                        except Exception:
                            pass
                        st.image(wc_png, caption='词云', width='stretch')
                        st.image(wc_png, caption='词云', width='stretch')
                    except Exception as e:
                        st.error(f'词云生成失败：{e}')

    with col4:
        if st.button('Step 4 · 绘制网络', width='stretch'):
            if not st.session_state.processed_csv:
                st.warning('请先完成 Step 2 或提供已有共现CSV')
            else:
                with st.spinner('正在绘制语义网络图...'):
                    try:
                        crawler = _load_crawler_module()
                        safe_name = crawler.sanitize_filename(title or bv)
                        out_png = os.path.join(result_dir, f"{safe_name}_semantic_network.png")
                        out_png, info = step3_draw_network(
                            edge_csv=st.session_state.processed_csv,
                            out_png=out_png,
                            layout=layout,
                            min_weight=int(min_weight),
                            seed=42
                        )
                        st.session_state.out_png = out_png
                        st.session_state.network_info = info
                        st.success(f'已生成语义网络图：{out_png}')
                        if info:
                            st.info(f"社区数：{info.get('num_communities')}；模块度：{(info.get('modularity') or 0):.3f}")
                            with st.expander('查看各社区节点列表'):
                                for c in info.get('communities', []):
                                    st.markdown(f"- 社区 {c['index']}（颜色 {c['color']}）：{', '.join(c['nodes'])}")
                    except Exception as e:
                        st.error(f'绘图失败：{e}')

    with col5:
        if st.button('Step 5 · AI 总结语义网络', width='stretch'):
            if not st.session_state.get('network_info'):
                st.warning('请先完成 Step 4（绘制网络）')
            elif not _get_effective_api_key():
                st.warning('请在侧边栏填写 DeepSeek API Key')
            else:
                st.session_state['ai_summary_text'] = ''
                st.session_state['ai_trigger'] = True

    with col6:
        if st.button('Step 6 · AI总结评论主题与热门议题', width='stretch'):
            if not st.session_state.get('raw_csv'):
                st.warning('请先完成 Step 1（抓取评论），或提供评论CSV')
            elif not _get_effective_api_key():
                st.warning('请在侧边栏填写 DeepSeek API Key')
            else:
                st.session_state['cmt_ai_text'] = ''
                st.session_state['cmt_ai_trigger'] = True
                st.session_state['cmt_ai_topk'] = int(cmt_topk)

    if st.button('一键执行（抓取→共现→词云→绘图→总结）', type='primary', width='stretch'):
            if not bv:
                st.warning('请先输入 BV 号')
            else:
                # 运行时间提示（用于 Step1 抓取阶段）
                counter = {'n': 0}
                info_box = st.empty()
                start_time = time.time()

                def _on_progress(delta: int):
                    counter['n'] += delta
                    elapsed_now = time.time() - start_time
                    info_box.markdown(f"**已爬取评论数: {counter['n']} ｜ 已运行时间: {elapsed_now:.1f}s**")

                with st.spinner('正在执行流水线...'):
                    try:
                        cn_path = 'clearwords.txt'
                        en_path = 'english_stopwords.txt'
                        stop_path = None
                        if up_stop is not None:
                            stop_path = _save_uploaded_to(os.path.join(result_dir, 'uploaded_stopwords.txt'), up_stop)
                        # 组合自定义分词：文本输入 + 上传文件
                        extra_words_list = parse_extra_words(extra_words_raw)
                        try:
                            if up_words is not None:
                                words_path = _save_uploaded_to(os.path.join(result_dir, 'uploaded_extra_words.txt'), up_words)
                                extra_words_list += load_stopwords(words_path)
                        except Exception:
                            pass
                        raw_csv, processed_csv, out_png, info = run_pipeline(
                            bv=bv,
                            title=(title or bv),
                            result_dir=result_dir,
                            layout=layout,
                            min_weight=int(min_weight),
                            top_n=int(top_n),
                            extra_words=extra_words_list,
                            chinese_stop=cn_path,
                            english_stop=en_path,
                            on_progress=_on_progress,
                            cookie=_sanitize_cookie_str(st.session_state.get('bili_cookie', '')),
                        )
                        st.session_state.raw_csv = raw_csv
                        st.session_state.processed_csv = processed_csv
                        st.session_state.out_png = out_png
                        st.session_state.network_info = info
                        # 词云（Step3）
                        default_remove = ['回复', '没', '_', 'doge', '说', '还', '只能', '会', '亱', '才能', '次', '没有', '现在', 'libo', '里', '觉得', '这种', '已经', '不会', '出来', '应该', '直接']
                        freqs = build_word_frequencies(
                            input_csv=raw_csv,
                            chinese_stop_path=cn_path,
                            english_stop_path=en_path,
                            stop_path=stop_path,
                            append_stopwords=parse_extra_words(stopwords_text),
                            extra_custom_words=extra_words_list,
                            default_remove=default_remove,
                            top_n=300,
                        )
                        crawler2 = _load_crawler_module()
                        safe_name2 = crawler2.sanitize_filename(title or bv)
                        wc_png = os.path.join(result_dir, f"{safe_name2}_wordcloud.png")
                        generate_wordcloud(freqs, wc_png)
                        st.session_state['word_freqs'] = freqs
                        st.session_state['wordcloud_png'] = wc_png
                        # 保存词频为 CSV（完整列表）
                        try:
                            import pandas as _pd
                            df_all = _pd.DataFrame(freqs, columns=['词语', '词频'])
                            df_all.index = _pd.RangeIndex(start=1, stop=len(df_all)+1, step=1)
                            df_all.index.name = 'Top'
                            wc_csv = os.path.join(result_dir, f"{safe_name2}_wordfreq.csv")
                            df_all.to_csv(wc_csv, index=False, encoding='utf-8-sig')
                            st.session_state['wordfreq_csv'] = wc_csv
                        except Exception:
                            pass
                        st.success('流水线完成')
                        # 展示抓取结束原因（成功或截断）
                        try:
                            crawler_info = _load_crawler_module()
                            if hasattr(crawler_info, 'get_last_fetch_info'):
                                _info2 = crawler_info.get_last_fetch_info()
                            elif callable(crawler_last_info):
                                _info2 = crawler_last_info()
                            else:
                                _info2 = {}
                            if _info2:
                                _reason_map = {
                                    'is_end': '平台返回：已到末尾',
                                    'max_offset': '已到平台翻页上限',
                                    'hard_limit': '达到本地安全上限',
                                    'no_progress': '无新增数据（可能已抓完）',
                                    'http_error': '网络/HTTP 错误',
                                    'exception': '请求异常中断',
                                    'exhausted': '达到最大页数限制',
                                }
                                _method_map = {'wbi': '光标接口（wbi）', 'legacy': '分页接口（旧版）'}
                                st.caption(f"抓取结束：{_reason_map.get(_info2.get('end_reason'),'未知')}；接口：{_method_map.get(_info2.get('method'), _info2.get('method','?'))}；页数：{_info2.get('pages','?')}；总计：{_info2.get('total_count','?')}")
                        except Exception:
                            pass
                        if info:
                            st.info(f"社区数：{info.get('num_communities')}；模块度：{(info.get('modularity') or 0):.3f}")
                            with st.expander('查看各社区节点列表'):
                                for c in info.get('communities', []):
                                    st.markdown(f"- 社区 {c['index']}（颜色 {c['color']}）：{', '.join(c['nodes'])}")
                        # DeepSeek 总结
                        if _get_effective_api_key():
                            # 触发在“结果与下载”区域的流式渲染
                            st.session_state['ai_summary_text'] = ''
                            st.session_state['ai_trigger'] = True
                            # 额外：触发“评论主题与热门议题”总结
                            st.session_state['cmt_ai_text'] = ''
                            st.session_state['cmt_ai_trigger'] = True
                            st.session_state['cmt_ai_topk'] = int(cmt_topk)
                    except Exception as e:
                        st.error(f'执行失败：{e}')

    # 已移除旧的 col6 区块（词云已在 Step 3）

    # —— 固定展示：抓取结束说明（跨重绘保留） ——
    try:
        _persist_info = st.session_state.get('fetch_end_info') or {}
        if _persist_info:
            st.caption(_format_end_info_cn(_persist_info))
    except Exception:
        pass

    st.markdown('---')
    st.subheader('结果与下载')

    def _preview_csv_table(csv_path: str, title: str):
        try:
            df_full = pd.read_csv(csv_path)
            st.markdown(f"**{title} 预览**")
            st.caption(f"共 {len(df_full)} 行，显示前 10 行")
            st.dataframe(df_full.head(10))
        except Exception as e:
            st.warning(f"无法加载 {title} 预览：{e}")

    # 第一行：评论 CSV 表格展示（含下载）——按点赞数排序，展示前50，分页（1-5页，每页10条）
    if st.session_state.raw_csv and os.path.exists(st.session_state.raw_csv):
        st.markdown('### 评论 CSV（按点赞数降序，前50条，分页展示）')
        with open(st.session_state.raw_csv, 'rb') as f:
            st.download_button('下载评论CSV', data=f, file_name=os.path.basename(st.session_state.raw_csv))
        try:
            df_full = pd.read_csv(st.session_state.raw_csv)
            if '点赞数量' in df_full.columns:
                df_full['点赞数量'] = pd.to_numeric(df_full['点赞数量'], errors='coerce').fillna(0).astype(int)
                df_sorted = df_full.sort_values('点赞数量', ascending=False)
            else:
                df_sorted = df_full
            top_k = min(50, len(df_sorted))
            df_top = df_sorted.head(top_k)
            per_page = 10
            total_pages = (top_k + per_page - 1) // per_page
            if total_pages <= 0:
                st.info('无数据可展示')
            else:
                # 分页控件（最多显示到第5页）
                options = list(range(1, min(5, total_pages) + 1))
                page = st.radio('选择页码', options=options, index=0, horizontal=True, key='cmt_page')
                start = (page - 1) * per_page
                end = min(start + per_page, top_k)
                st.caption(f"共 {len(df_full)} 行；当前显示第 {page}/{total_pages} 页（{start+1}-{end}）")
                st.dataframe(df_top.iloc[start:end], width='stretch')
        except Exception as e:
            st.warning(f"无法加载 评论 CSV 预览：{e}")

    # 第二行：左词频表格，右词云
    col_wf, col_wc = st.columns([1, 1])
    with col_wf:
        try:
            if st.session_state.get('word_freqs'):
                import pandas as _pd
                top10 = st.session_state['word_freqs'][:10]
                df_top = _pd.DataFrame(top10, columns=['词语', '词频'])
                df_top.index = _pd.RangeIndex(start=1, stop=len(df_top)+1, step=1)
                df_top.index.name = 'Top'
                st.markdown('### 词频 Top 10')
                st.dataframe(df_top, width='stretch')
                if st.session_state.get('wordfreq_csv') and os.path.exists(st.session_state['wordfreq_csv']):
                    with open(st.session_state['wordfreq_csv'], 'rb') as f:
                        st.download_button('下载词频CSV', data=f, file_name=os.path.basename(st.session_state['wordfreq_csv']))
        except Exception:
            pass
    with col_wc:
        if st.session_state.get('wordcloud_png') and os.path.exists(st.session_state['wordcloud_png']):
            st.markdown('### 词云')
            st.image(st.session_state['wordcloud_png'], caption='词云', width='stretch')
            with open(st.session_state['wordcloud_png'], 'rb') as f:
                st.download_button('下载词云', data=f, file_name=os.path.basename(st.session_state['wordcloud_png']))

    # 第三行：左共现 CSV 表格，右语义网络图
    col_co, col_net = st.columns([1, 1])
    with col_co:
        if st.session_state.processed_csv and os.path.exists(st.session_state.processed_csv):
            st.markdown('### 共现 CSV')
            with open(st.session_state.processed_csv, 'rb') as f:
                st.download_button('下载共现CSV', data=f, file_name=os.path.basename(st.session_state.processed_csv))
            _preview_csv_table(st.session_state.processed_csv, '共现 CSV')
    with col_net:
        if st.session_state.out_png and os.path.exists(st.session_state.out_png):
            st.markdown('### 语义网络图')
            st.image(st.session_state.out_png, caption='语义网络图预览', width='stretch')
            with open(st.session_state.out_png, 'rb') as f:
                st.download_button('下载语义网络图', data=f, file_name=os.path.basename(st.session_state.out_png))

    # 第四行：左各聚类词语，右 AI 总结（语义网络）
    col_comm, col_ai = st.columns([1, 1])
    with col_comm:
        if st.session_state.get('network_info'):
            st.markdown('### 各聚类的词语')
            try:
                for c in st.session_state['network_info'].get('communities', []):
                    st.markdown(f"- 聚类 {c['index']}（颜色 {c['color']}）：{', '.join(c['nodes'])}")
            except Exception:
                pass
    with col_ai:
        st.markdown('### AI 总结（每个聚类的话题）')
        # 始终展示已有结果，避免翻页导致内容消失
        if st.session_state.get('ai_summary_text'):
            st.markdown(st.session_state['ai_summary_text'])
        # 若触发标志打开，则继续流式追加
        if st.session_state.get('ai_trigger') and _get_effective_api_key() and st.session_state.get('network_info'):
            try:
                _ds = _load_deepseek_summary()
                ai_holder = st.empty()
                text_acc = st.session_state.get('ai_summary_text', '')
                ai_holder.markdown(text_acc)
                for piece in _ds.stream_summarize_with_deepseek(_get_effective_api_key(), st.session_state['network_info'], language='zh'):
                    text_acc += piece
                    st.session_state['ai_summary_text'] = text_acc
                    ai_holder.markdown(text_acc)
            except Exception:
                pass
            finally:
                st.session_state['ai_trigger'] = False

    # 第五行（保留为语义网络的AI总结输出区域）
    st.markdown('---')
    st.subheader('AI总结：评论主题与热门议题（按点赞Top100）')
    # 始终展示已有结果，避免翻页导致内容消失
    if st.session_state.get('cmt_ai_text'):
        st.markdown(st.session_state['cmt_ai_text'])
    # 若触发标志打开，则继续流式追加
    if st.session_state.get('cmt_ai_trigger') and _get_effective_api_key() and st.session_state.get('raw_csv'):
        try:
            _cs = _load_comment_summary()
            holder = st.empty()
            acc = st.session_state.get('cmt_ai_text', '')
            holder.markdown(acc)
            topk = int(st.session_state.get('cmt_ai_topk') or 100)
            for chunk in _cs.stream_summarize_comment_themes(_get_effective_api_key(), st.session_state['raw_csv'], top_k=topk, language='zh'):
                acc += chunk
                st.session_state['cmt_ai_text'] = acc
                holder.markdown(acc)
        except Exception as _e:
            st.error(f'评论主题总结失败：{_e}')
        finally:
            st.session_state['cmt_ai_trigger'] = False


    # —— 导出与快照 ——
    st.markdown('---')
    st.subheader('导出与快照')

    def _file_to_b64(path: str) -> str:
        try:
            with open(path, 'rb') as f:
                return base64.b64encode(f.read()).decode('ascii')
        except Exception:
            return ''

    def _df_html_safely(df) -> str:
        try:
            return df.to_html(index=False, border=0)
        except Exception:
            return '<p>表格生成失败</p>'

    def _build_report_html() -> str:
        title_txt = (st.session_state.get('report_title') or (st.session_state.get('raw_csv') and os.path.basename(st.session_state['raw_csv']).replace('.csv','')) or '报告')
        # 基础数据
        raw_csv = st.session_state.get('raw_csv')
        processed_csv = st.session_state.get('processed_csv')
        net_png = st.session_state.get('out_png')
        wc_png = st.session_state.get('wordcloud_png')
        wf_top = st.session_state.get('word_freqs') or []
        # 预览表
        raw_table_html = ''
        co_table_html = ''
        try:
            if raw_csv and os.path.exists(raw_csv):
                df_full = pd.read_csv(raw_csv)
                if '点赞数量' in df_full.columns:
                    df_full['点赞数量'] = pd.to_numeric(df_full['点赞数量'], errors='coerce').fillna(0).astype(int)
                    df_full = df_full.sort_values('点赞数量', ascending=False)
                raw_table_html = _df_html_safely(df_full.head(50))
        except Exception:
            pass
        try:
            if processed_csv and os.path.exists(processed_csv):
                df_co = pd.read_csv(processed_csv)
                co_table_html = _df_html_safely(df_co.head(100))
        except Exception:
            pass
        # 词频 top10
        wf_html = ''
        try:
            if wf_top:
                _pd = pd.DataFrame(wf_top[:10], columns=['词语','词频'])
                wf_html = _df_html_safely(_pd)
        except Exception:
            pass
        # 为交互表格添加 DataTables 包装
        def _dt_wrap(html: str, table_id: str, classes: str = 'display compact') -> str:
            try:
                if not html:
                    return ''
                return html.replace('<table', f'<table id="{table_id}" class="{classes}"', 1)
            except Exception:
                return html

        raw_table_html = _dt_wrap(raw_table_html, 'raw_table')
        co_table_html = _dt_wrap(co_table_html, 'co_table')
        wf_html = _dt_wrap(wf_html, 'wf_table')

        # 图片 base64
        wc_b64 = _file_to_b64(wc_png) if wc_png and os.path.exists(wc_png) else ''
        net_b64 = _file_to_b64(net_png) if net_png and os.path.exists(net_png) else ''
        # AI 文本
        ai1 = st.session_state.get('ai_summary_text') or ''
        ai2 = st.session_state.get('cmt_ai_text') or ''
        # 结束原因：优先读取 session_state
        end_info = st.session_state.get('fetch_end_info') or {}
        # 兜底：若 session_state 为空，尝试再次从模块读取一次（避免某些云端重绘导致的空白）
        if not end_info:
            try:
                cmod = _load_crawler_module()
                if hasattr(cmod, 'get_last_fetch_info'):
                    end_info = cmod.get_last_fetch_info() or {}
            except Exception:
                end_info = {}
        end_line = _format_end_info_cn(end_info)

        html = f"""
<!doctype html>
<html lang=zh>
<head>
  <meta charset="utf-8" />
  <title>{title_txt} - 报告</title>
  <!-- DataTables（CDN，需联网以获得交互分页；离线则回退为静态表） -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/datatables.net-dt@1.13.8/css/jquery.dataTables.min.css">
  <script src="https://cdn.jsdelivr.net/npm/jquery@3.7.1/dist/jquery.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/datatables.net@1.13.8/js/jquery.dataTables.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans SC', 'Microsoft YaHei', sans-serif; margin: 24px; }}
    h1, h2 {{ margin: 10px 0; }}
    .row {{ display:flex; gap:24px; align-items:flex-start; }}
    .col {{ flex:1; min-width:0; }}
    img {{ max-width:100%; height:auto; border:1px solid #eee; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #eee; padding: 6px 8px; }}
    caption {{ text-align:left; color:#666; margin-bottom:6px; }}
    .card {{ background:#fafafa; border:1px solid #eee; padding:12px; margin:12px 0; }}
  </style>
  </head>
  <body>
    <h1>{title_txt} - 运行报告</h1>
    <div class="card"><strong>{end_line}</strong></div>

    <h2>评论（按赞前50）</h2>
    {raw_table_html}

    <div class="row">
      <div class="col">
        <h2>词频 Top 10</h2>
        {wf_html}
      </div>
      <div class="col">
        <h2>词云</h2>
        {('<img src="data:image/png;base64,' + wc_b64 + '" />') if wc_b64 else '<p>无词云</p>'}
      </div>
    </div>

    <div class="row">
      <div class="col">
        <h2>共现 CSV（前100行）</h2>
        {co_table_html}
      </div>
      <div class="col">
        <h2>语义网络图</h2>
        {('<img src="data:image/png;base64,' + net_b64 + '" />') if net_b64 else '<p>无网络图</p>'}
      </div>
    </div>

    <h2>AI 总结（语义网络聚类）</h2>
    <div class="card">{ai1.replace('\n','<br/>')}</div>

    <h2>AI 总结：评论主题与热门议题（按点赞TopX）</h2>
    <div class="card">{ai2.replace('\n','<br/>')}</div>
    <script>
      (function(){{
        if (window.jQuery && $.fn && $.fn.dataTable) {{
          try {{
            if (document.getElementById('raw_table')) $('#raw_table').DataTable({{pageLength:10, deferRender:true}});
            if (document.getElementById('co_table')) $('#co_table').DataTable({{pageLength:25, deferRender:true}});
            if (document.getElementById('wf_table')) $('#wf_table').DataTable({{paging:false, searching:false, info:false}});
          }} catch(e) {{ console.warn(e); }}
        }}
      }})();
    </script>
  </body>
</html>
"""
        return html

    col_r1, col_r2 = st.columns([1,1])
    with col_r1:
        try:
            report_html = _build_report_html()
            file_name = (crawler_sanitize(st.session_state.get('report_title') or (st.session_state.get('raw_csv') and os.path.basename(st.session_state['raw_csv']).replace('.csv','')) or 'report') + '_report.html')
            st.download_button('下载报告HTML（快照）', data=report_html.encode('utf-8'), file_name=file_name, mime='text/html')
        except Exception:
            st.caption('报告生成失败')
    with col_r2:
        try:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                # 写入报告
                zf.writestr('report.html', (_build_report_html() or '').encode('utf-8'))
                # 附件文件
                for p in [st.session_state.get('raw_csv'), st.session_state.get('processed_csv'), st.session_state.get('out_png'), st.session_state.get('wordcloud_png'), st.session_state.get('wordfreq_csv')]:
                    if p and os.path.exists(p):
                        try:
                            zf.write(p, arcname=os.path.basename(p))
                        except Exception:
                            pass
            st.download_button('下载结果ZIP（含CSV/PNG/报告）', data=buf.getvalue(), file_name='results_bundle.zip')
        except Exception:
            st.caption('ZIP 打包失败')


if __name__ == '__main__':
    # 以 Streamlit 方式运行：streamlit run app.py
    render_streamlit_app()


