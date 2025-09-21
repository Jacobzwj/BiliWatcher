import requests
import re
import time
import random
import csv
import os

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.bilibili.com'
}


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


def fetch_comments(video_id, max_pages=1000, on_progress=None):
    comments = []
    last_count = 0
    comment_counter = 0
    for page in range(1, max_pages + 1):
        url = f'https://api.bilibili.com/x/v2/reply?pn={page}&type=1&oid={video_id}&sort=2'
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and data.get('code', 0) != 0:
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
                if last_count == len(comments):
                    break
                last_count = len(comments)
            else:
                break
        except requests.RequestException as e:
            print(f"请求出错: {e}")
            break
        except RuntimeError:
            raise
        # 随机延时，减少被限流概率
        time.sleep(random.uniform(0.1, 0.2))
    return comments


def save_comments_to_csv(comments, output_name):
    os.makedirs('./result', exist_ok=True)
    safe_name = sanitize_filename(output_name)
    with open(f'./result/{safe_name}.csv', mode='w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['用户昵称', '性别', '评论内容', '被回复用户', '评论层级', '用户当前等级', '点赞数量', '回复时间'])
        writer.writeheader()
        for comment in comments:
            writer.writerow(comment)


