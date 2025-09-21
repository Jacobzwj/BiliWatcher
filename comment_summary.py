from typing import List, Dict, Generator
import pandas as pd
import re


def _load_top_comments(csv_path: str, top_k: int = 100) -> List[Dict[str, str]]:
    df = pd.read_csv(csv_path)
    # 标准化列名（兼容可能的英文字段名）
    like_col_candidates = ['点赞数量', 'like', 'likes', '点赞']
    text_col_candidates = ['评论内容', 'content', 'text', '评论']

    like_col = None
    for c in like_col_candidates:
        if c in df.columns:
            like_col = c
            break
    text_col = None
    for c in text_col_candidates:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        raise ValueError('评论CSV缺少文本列（如“评论内容”）')

    if like_col is not None:
        df[like_col] = pd.to_numeric(df[like_col], errors='coerce').fillna(0).astype(int)
        df = df.sort_values(like_col, ascending=False)
    else:
        # 无点赞列则保持原顺序
        like_col = None

    # 取前 top_k
    df = df.head(min(top_k, len(df)))

    # 清洗与限长，避免超长输入
    def _clean_text(t: str) -> str:
        s = str(t or '').strip()
        s = re.sub(r'\s+', ' ', s)
        # 截断到 160 字符，避免 tokens 过多
        return s[:160]

    records: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        rec = {
            'text': _clean_text(row[text_col])
        }
        if like_col is not None:
            rec['likes'] = int(row[like_col])
        records.append(rec)
    return records


def _build_prompt_from_comments(comments: List[Dict[str, str]], language: str = 'zh') -> Dict[str, str]:
    # 构造用户消息，将评论按 “编号. [赞X] 评论文本” 列表化
    lines: List[str] = []
    for idx, rec in enumerate(comments, start=1):
        like_str = f"[赞{rec.get('likes', 0)}] " if 'likes' in rec else ''
        text = rec.get('text', '')
        if not text:
            continue
        lines.append(f"{idx}. {like_str}{text}")
    corpus = "\n".join(lines)

    system_prompt = (
        "你是中文社媒分析专家。请阅读以下来自B站评论的文本（已按点赞数从高到低取前若干条）。"
        "请输出三部分内容，仅输出结果，不要解释或展示思考过程：\n"
        "1) 主题聚类：将评论划分为若干主题，每个主题用一句中文概括，并附上一个代表性原评论作为例子；\n"
        "2) 高频议题（严格最多3个）：找出被很多用户重复提到的议题，最多列出3个；若不足3个则按实际数量输出。相似或重复的议题必须合并为一个更通用的表述。每个议题用一句中文描述，并附上最多3个不同的原评论示例（不可复用同一评论，若不足3条则给现有条数）；\n"
        "3) 情感倾向（分组展示）：将第2步得到的每个议题分到以下三类之一——积极正面/消极负面/中立。每条需包含：议题名 + 简短理由（不超过20字）+ 一个示例原句（直接粘贴，不得用编号）。不得新增新的议题或示例。\n"
        "严禁输出模板或占位：不得出现尖括号< >、不得出现‘<原评论>’等占位；所有示例内容必须来自输入评论原句的片段，且尽量≤60字。\n"
        "输出前自检：高频议题条目数必须≤3，议题名称不重复不包含关系；若检测到‘<’或‘>’字符，必须改写为真实内容或删除该行。"
    )

    user_prompt = (
        "评论语料如下：\n" + corpus + "\n\n"
        "请严格按以下格式输出（不要多于下列条目；不得出现占位或尖括号）：\n"
        "【主题聚类】\n"
        "- 主题1：<一句话主题>；示例：<原评论>\n"
        "- 主题2：<一句话主题>；示例：<原评论>\n"
        "...\n"
        "【高频议题（最多3个，不足则按实际；相似合并；不得重复）】\n"
        "- 议题1：一句话描述；示例1：原评论片段；示例2：原评论片段；示例3：原评论片段\n"
        "- 议题2：一句话描述；示例1：原评论片段；示例2：原评论片段；示例3：原评论片段\n"
        "- 议题3：一句话描述；示例1：原评论片段；示例2：原评论片段；示例3：原评论片段\n"
        "【情感倾向（分组展示：议题 + 简短理由 + 示例原句）】\n"
        "- 积极正面：\n"
        "  - <议题名>；理由：<不超过20字>；示例：<原评论片段>\n"
        "  - <议题名>；理由：<不超过20字>；示例：<原评论片段>\n"
        "- 消极负面：\n"
        "  - <议题名>；理由：<不超过20字>；示例：<原评论片段>\n"
        "  - <议题名>；理由：<不超过20字>；示例：<原评论片段>\n"
        "- 中立：\n"
        "  - <议题名>；理由：<不超过20字>；示例：<原评论片段>\n"
    )

    return {"system": system_prompt, "user": user_prompt}


def summarize_comment_themes(api_key: str, csv_path: str, top_k: int = 100, language: str = 'zh', timeout: int = 60) -> str:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(f"缺少 openai 依赖，请先安装：pip install openai。错误: {e}")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    comments = _load_top_comments(csv_path, top_k=top_k)
    prompts = _build_prompt_from_comments(comments, language=language)

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]},
        ],
        stream=False,
        timeout=timeout,
        temperature=0.2,
        max_tokens=900,
    )

    choice = resp.choices[0]
    summary = ''
    if getattr(choice, 'message', None) is not None:
        summary = choice.message.content or ''
    return summary or ''


def stream_summarize_comment_themes(api_key: str, csv_path: str, top_k: int = 100, language: str = 'zh') -> Generator[str, None, None]:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(f"缺少 openai 依赖，请先安装：pip install openai。错误: {e}")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    comments = _load_top_comments(csv_path, top_k=top_k)
    prompts = _build_prompt_from_comments(comments, language=language)

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]},
        ],
        stream=True,
        temperature=0.2,
        max_tokens=1000,
    )

    for chunk in resp:
        delta = chunk.choices[0].delta
        if hasattr(delta, 'content') and delta.content:
            yield delta.content



