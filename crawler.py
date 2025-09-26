import requests
import re
import time
import random
import csv
import os
import hashlib
import urllib.parse

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.bilibili.com'
}


# 最近一次抓取的结束信息
_last_fetch_info = {}

def get_last_fetch_info() -> dict:
    return dict(_last_fetch_info)


# ---- WBI 签名（用于 cursor 接口） ----
_MIXIN_KEY_ENC_TAB = [46,47,18,2,53,8,23,32,15,50,10,31,58,3,45,35,27,43,5,49,33,9,42,19,29,28,14,39,12,38,41,13,37,48,7,16,24,55,40,61,26,17,0,1,60,51,30,4,22,25,54,21,56,59,6,57,11,52,36,34,20]

def _get_wbi_keys() -> tuple[str, str]:
    url = 'https://api.bilibili.com/x/web-interface/nav'
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    j = r.json()
    img = j.get('data', {}).get('wbi_img', {})
    img_url = img.get('img_url', '')
    sub_url = img.get('sub_url', '')
    img_key = os.path.basename(img_url).split('.')[0]
    sub_key = os.path.basename(sub_url).split('.')[0]
    return img_key, sub_key

def _get_mixin_key(img_key: str, sub_key: str) -> str:
    s = (img_key + sub_key)
    mixed = ''.join([s[i] for i in _MIXIN_KEY_ENC_TAB if i < len(s)])
    return mixed[:32]

def _wbi_sign(params: dict) -> dict:
    img_key, sub_key = _get_wbi_keys()
    mixin_key = _get_mixin_key(img_key, sub_key)
    params = dict(params)
    params['wts'] = int(time.time())
    # 移除特殊字符，按 key 排序
    filtered = {k: ''.join([c for c in str(v) if c not in "!'()*"]) for k, v in params.items()}
    query = urllib.parse.urlencode(sorted(filtered.items()))
    w_rid = hashlib.md5((query + mixin_key).encode('utf-8')).hexdigest()
    filtered['w_rid'] = w_rid
    return filtered

def _fetch_comments_via_wbi(video_id: str, on_progress=None, hard_max_count: int | None = 20000):
    comments = []
    next_cursor = 0
    started = time.time()
    pages = 0
    while True:
        base = 'https://api.bilibili.com/x/v2/reply/wbi/main'
        params = {
            'oid': video_id,
            'type': 1,
            'mode': 3,  # 按时间
            'next': next_cursor,
            'ps': 20,
        }
        try:
            signed = _wbi_sign(params)
            resp = requests.get(base, params=signed, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get('code', 0) != 0:
                # 光标接口失败则中断，交给旧接口兜底
                raise RuntimeError(f"wbi 接口错误：code={data.get('code')}, message={data.get('message')}")
            d = data.get('data', {}) or {}
            replies = d.get('replies') or []
            if not isinstance(replies, list):
                replies = []
            for comment in replies:
                try:
                    comment_info = {
                        '用户昵称': comment['member']['uname'],
                        '评论内容': comment['content']['message'],
                        '被回复用户': '',
                        '评论层级': '一级评论',
                        '性别': comment['member']['sex'],
                        '用户当前等级': comment['member']['level_info']['current_level'],
                        '点赞数量': comment['like'],
                        '回复时间': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(comment['ctime']))
                    }
                    comments.append(comment_info)
                    if on_progress is not None:
                        try:
                            on_progress(1)
                        except Exception:
                            pass
                    rcount = int(comment.get('rcount', 0) or 0)
                    replies2 = fetch_comment_replies(video_id, comment['rpid'], comment['member']['uname'], expected_count=rcount, on_progress=on_progress)
                    comments.extend(replies2)
                except Exception:
                    continue

            cur = d.get('cursor', {}) or {}
            pages += 1
            if cur.get('is_end') is True:
                _last_fetch_info.update({'method': 'wbi', 'end_reason': 'is_end', 'pages': pages})
                break
            next_cursor = cur.get('next', 0)
            if hard_max_count is not None and len(comments) >= int(hard_max_count):
                _last_fetch_info.update({'method': 'wbi', 'end_reason': 'hard_limit', 'pages': pages})
                break
            time.sleep(random.uniform(0.1, 0.2))
        except Exception:
            # 交给旧接口兜底
            raise
    _last_fetch_info.update({'method': 'wbi', 'total_count': len(comments), 'duration_sec': time.time() - started})
    return comments

def sanitize_filename(name):
    return re.sub(r'[\\/:*?\"<>|]+', '_', str(name)).strip() or 'output'


def get_video_id(bv: str) -> str:
    url = f'https://www.bilibili.com/video/{bv}'
    html = requests.get(url, headers=headers)
    html.encoding = 'utf-8'
    content = html.text
    aid_regx = '"aid":(.*?),"bvid":"{}"'.format(bv)
    video_aid = re.findall(aid_regx, content)[0]
    return video_aid


def fetch_comment_replies(video_id, comment_id, parent_user_name, expected_count=0, max_pages=2000, on_progress=None):
    replies = []
    preLen = 0
    for page in range(1, max_pages + 1):
        url = f'https://api.bilibili.com/x/v2/reply/reply?oid={video_id}&type=1&root={comment_id}&ps=10&pn={page}'
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and data.get('code', 0) != 0:
                    # 部分视频在翻页过深时会返回 -400 max offset exceeded：视作自然终止
                    if data.get('code') == -400 and str(data.get('message', '')).lower().find('max offset') != -1:
                        print('二级评论达到最大偏移，已停止继续翻页。')
                        break
                    raise RuntimeError(f"二级评论接口错误，code={data.get('code')}, message={data.get('message')}")
                if data and data.get('data'):
                    replies_data = data.get('data', {}).get('replies') or []
                    if not isinstance(replies_data, list):
                        replies_data = []
                    if page == 1 and expected_count and len(replies_data) == 0:
                        raise RuntimeError('二级评论为空但预期存在，可能 Cookie 过期或被风控。')
                    for reply in replies_data:
                        reply_info = {
                            '用户昵称': reply['member']['uname'],
                            '评论内容': reply['content']['message'],
                            '被回复用户': parent_user_name,
                            '评论层级': '二级评论',
                            '性别': reply['member']['sex'],
                            '用户当前等级': reply['member']['level_info']['current_level'],
                            '点赞数量': reply['like'],
                            '回复时间': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(reply['ctime']))
                        }
                        replies.append(reply_info)
                        if on_progress is not None:
                            try:
                                on_progress(1)
                            except Exception:
                                pass
                    # 如果页数据没有增加，或接口提示 is_end，则认为结束
                    cursor = (data.get('data', {}) or {}).get('cursor', {}) or {}
                    if cursor.get('is_end') is True:
                        break
                    if preLen == len(replies):
                        break
                    preLen = len(replies)
                else:
                    return replies
        except requests.RequestException as e:
            print(f"请求出错: {e}")
            break
        # 随机延时，减少被限流概率
        time.sleep(random.uniform(0.2, 0.5))
    return replies


def fetch_comments(video_id, max_pages=1000, on_progress=None, hard_max_count: int | None = 12000):
    comments = []
    last_count = 0
    comment_counter = 0
    _last_fetch_info.clear()
    # 尝试更稳的光标接口（wbi）。若失败，则回退原分页接口
    try:
        comments = _fetch_comments_via_wbi(video_id, on_progress=on_progress, hard_max_count=hard_max_count or 20000)
        if comments:
            _last_fetch_info.setdefault('method', 'wbi')
            _last_fetch_info.setdefault('end_reason', 'is_end')
            _last_fetch_info['total_count'] = len(comments)
            return comments
    except Exception:
        comments = []
    for page in range(1, max_pages + 1):
        url = f'https://api.bilibili.com/x/v2/reply?pn={page}&type=1&oid={video_id}&sort=2'
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and data.get('code', 0) != 0:
                    # -400 max offset exceeded：翻页超过服务端上限，正常结束
                    if data.get('code') == -400 and str(data.get('message', '')).lower().find('max offset') != -1:
                        print('评论翻页达到最大偏移，停止抓取。')
                        _last_fetch_info.update({'method': 'legacy', 'end_reason': 'max_offset', 'pages': page})
                        break
                    raise RuntimeError(f"评论列表接口错误，code={data.get('code')}, message={data.get('message')}")
                if data and data.get('data'):
                    replies_data = data.get('data', {}).get('replies') or []
                    if not isinstance(replies_data, list):
                        replies_data = []
                    for comment in replies_data:
                        comment_info = {
                            '用户昵称': comment['member']['uname'],
                            '评论内容': comment['content']['message'],
                            '被回复用户': '',
                            '评论层级': '一级评论',
                            '性别': comment['member']['sex'],
                            '用户当前等级': comment['member']['level_info']['current_level'],
                            '点赞数量': comment['like'],
                            '回复时间': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(comment['ctime']))
                        }
                        comments.append(comment_info)
                        comment_counter += 1
                        print(f"\r已爬取评论数: {comment_counter}", end="", flush=True)
                        if on_progress is not None:
                            try:
                                on_progress(1)
                            except Exception:
                                pass
                        rcount = 0
                        try:
                            rcount = int(comment.get('rcount', 0))
                        except Exception:
                            rcount = 0
                        replies = fetch_comment_replies(video_id, comment['rpid'], comment['member']['uname'], expected_count=rcount, on_progress=on_progress)
                        comments.extend(replies)
                        comment_counter += len(replies)
                        if len(replies) > 0:
                            print(f"\r已爬取评论数: {comment_counter}", end="", flush=True)
                # 服务端游标/终止信号
                cursor = (data.get('data', {}) or {}).get('cursor', {}) or {}
                if cursor.get('is_end') is True:
                    _last_fetch_info.update({'method': 'legacy', 'end_reason': 'is_end', 'pages': page})
                    break
                if hard_max_count is not None and len(comments) >= int(hard_max_count):
                    print(f"达到本地安全上限 {hard_max_count} 条，停止抓取。")
                    _last_fetch_info.update({'method': 'legacy', 'end_reason': 'hard_limit', 'pages': page})
                    break
                if last_count == len(comments):
                    _last_fetch_info.update({'method': 'legacy', 'end_reason': 'no_progress', 'pages': page})
                    break
                last_count = len(comments)
            else:
                _last_fetch_info.update({'method': 'legacy', 'end_reason': 'http_error', 'http_status': response.status_code, 'pages': page})
                break
        except requests.RequestException as e:
            print(f"请求出错: {e}")
            _last_fetch_info.update({'method': 'legacy', 'end_reason': 'exception', 'error': str(e), 'pages': page})
            break
        except RuntimeError:
            raise
        # 随机延时，减少被限流概率
        time.sleep(random.uniform(0.1, 0.2))
    _last_fetch_info.setdefault('method', 'legacy')
    _last_fetch_info.setdefault('end_reason', 'exhausted' if page >= max_pages else _last_fetch_info.get('end_reason'))
    _last_fetch_info['total_count'] = len(comments)
    return comments


def save_comments_to_csv(comments, output_name):
    os.makedirs('./result', exist_ok=True)
    safe_name = sanitize_filename(output_name)
    with open(f'./result/{safe_name}.csv', mode='w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['用户昵称', '性别', '评论内容', '被回复用户', '评论层级', '用户当前等级', '点赞数量', '回复时间'])
        writer.writeheader()
        for comment in comments:
            writer.writerow(comment)


