from typing import Dict, List


def _format_communities_text(info: Dict) -> str:
    lines: List[str] = []
    num = info.get('num_communities') or 0
    modularity = info.get('modularity')
    lines.append(f"聚类数: {num}")
    if modularity is not None:
        lines.append(f"模块度: {modularity:.3f}")
    for c in info.get('communities', []):
        idx = c.get('index')
        color = c.get('color')
        nodes = c.get('nodes') or []
        preview = ', '.join(nodes[:40]) + (" …" if len(nodes) > 40 else "")
        lines.append(f"聚类 {idx}（颜色 {color}）：{preview}")
    return '\n'.join(lines)


def summarize_with_deepseek(api_key: str, community_info: Dict, language: str = 'zh', timeout: int = 60) -> Dict[str, str]:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(f"缺少 openai 依赖，请先安装：pip install openai。错误: {e}")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    communities_text = _format_communities_text(community_info)

    system_prompt = (
        "你是一个图分析与文本挖掘专家。现在给你一个语义网络的聚类（社区）划分结果。"
        "请严格按照要求：对“每个聚类”仅输出“一句话”的简洁中文主题总结，"
        "聚焦用户主要在讨论什么；不要输出思考过程，不要扩展推理文本。"
    )

    user_prompt = (
        f"请阅读以下聚类信息，并输出“一句总结/聚类”，格式如下：\n"
        f"聚类 1：<一句话总结>\n聚类 2：<一句话总结>\n...\n\n"
        f"语料（供你推理）：\n{communities_text}\n\n"
        f"要求：\n- 每个聚类只给出一句中文总结；\n- 不要解释，不要列要点；\n- 按序号逐行输出。\n"
    )

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        timeout=timeout,
        temperature=0.2,
        max_tokens=700,
    )

    choice = resp.choices[0]
    summary = ''
    if getattr(choice, 'message', None) is not None:
        summary = choice.message.content or ''

    return {
        'thinking': '',
        'summary': summary or '',
    }


def stream_summarize_with_deepseek(api_key: str, community_info: Dict, language: str = 'zh'):
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(f"缺少 openai 依赖，请先安装：pip install openai。错误: {e}")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    communities_text = _format_communities_text(community_info)
    system_prompt = (
        "你是一个图分析与文本挖掘专家。给定聚类关键词，请为每个聚类输出一行、仅一句话的中文主题总结。"
        "只输出总结文本，不要解释或列要点。"
    )
    user_prompt = (
        f"聚类信息：\n{communities_text}\n\n"
        f"按以下格式输出：\n聚类 1：<一句话总结>\n聚类 2：<一句话总结>\n..."
    )

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        stream=True,
        max_tokens=900,
        temperature=0.2,
    )

    for chunk in resp:
        delta = chunk.choices[0].delta
        if hasattr(delta, 'content') and delta.content:
            yield delta.content


