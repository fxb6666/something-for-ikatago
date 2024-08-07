#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
功能：此脚本可下载KataGo官网的所有权重文件，并支持使用第三方下载链接。为“CUDA”和“TENSORRT”引擎使用6b至28b的权重预设了适合于“Tesla T4”的最佳线程。
提示：'lxml'模块可用时效率更高。
注意：当存在“./data/weights”目录时，将无提示直接下载文件到此目录，同名文件会被覆盖。

一、基本参数
1. 参数示例：AUTO，NEW，18b，b18，18b-new，18b8526，18bs8526，b18s8526。仅以18b举例，其它块（block）数同理。
2. “AUTO”和“NEW”分别可以下载最强权重和最新权重。
3. 仅指定块数时将下载对应块数的最强权重（elo下限值最大）。
4. 块数后附加“-new”将下载指定块数中的最新权重。
5. 块数后附加s和采样数可下载指定权重。

二、特殊参数（加引号）
1. 支持正则表达式匹配特殊权重名称，表达式需要位于两条斜线中间，例如："/b18.*uec/"。如果有多个结果，只下载最新的一个。
2. 支持以 http/https 开头的下载链接。
3. 支持谷歌云端硬盘单个文件的分享链接和ID。 如果只有ID部分，需要以 id= 开头。
'''

import urllib.request
from urllib.parse import unquote
import gzip
import json
import re
import os
import sys
import subprocess
try:
    import lxml.html
    use_lxml = True
except:
    use_lxml = False
    from concurrent.futures import ThreadPoolExecutor

KATAGO_BACKEND = None
if len(sys.argv) == 1:
    WEIGHT_FILE = 'AUTO'
elif len(sys.argv) == 2:
    WEIGHT_FILE = sys.argv[1]
elif len(sys.argv) == 3:
    WEIGHT_FILE = sys.argv[1]
    KATAGO_BACKEND = sys.argv[2]
else:
    print('ERROR: too many arguments')
    sys.exit(1)

def get_group1(pattern, string):
    re_result = re.search(pattern, string, re.IGNORECASE)
    if re_result and re_result.group(1):
        return re_result.group(1)
    else:
        return None

def get_page(url, get_content=True):
    for i in range(2):
        try:
            user_agent = "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
            headers = {'User-Agent': user_agent, 'Accept-Encoding': 'gzip'}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as response:
                if not get_content:
                    return response
                content_ = response.read()
                if response.headers.get('Content-Encoding') == 'gzip':
                    content_ = gzip.decompress(content_)
                return response, content_
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: {e}")
            if i == 1:
                sys.exit(1)
            print("Retrying...")
        except:
            raise

def get_page_number(pattern):
    # 获取第一个匹配的模型所在页数
    url = 'https://katagotraining.org/api/networks-for-elo/?format=json' #更新较慢
    response, content_ = get_page(url)
    infos = json.loads(content_)
    numModels = len(infos)
    total_numPages = (numModels - 1) // 20 + 1
    model_num = 0
    page_number = ""
    for info in infos:
        model_num = model_num + 1
        model_name = info['name']
        if re.search(pattern, model_name, re.IGNORECASE):
            page_number = (model_num - 1) // 20 + 1
            break
    if not page_number:
        print(f'ERROR: No weights matching "{pattern}" were found.')
        sys.exit(1)
    return page_number, total_numPages

model_url = None
regexp_mode = False

# 特殊参数部分
if re.search(r'drive\.google\.com|^id=', WEIGHT_FILE, re.IGNORECASE) is not None:
    patterns = ['id=([^&/?]*)', '/file.*/d/([^&/?]*)']
    for pattern in patterns:
        file_id = get_group1(pattern, WEIGHT_FILE)
        if file_id is not None:
            break
    if file_id == None:
        print(f'ERROR: No match found for the file ID.')
        sys.exit(1)
    model_url = f'https://drive.usercontent.google.com/download?id={file_id}&confirm=t'
elif re.search('^http', WEIGHT_FILE, re.IGNORECASE):
    model_url = WEIGHT_FILE
elif re.search('^/.*/$', WEIGHT_FILE, re.IGNORECASE):
    regexp_mode = True
    regexp = get_group1('/(.*)/', WEIGHT_FILE)

BLOCK = None
SAMPLE = None
use_new = False

# 基本参数部分
if model_url == None and not regexp_mode:
    if WEIGHT_FILE.upper() in ['AUTO', 'NEW']:
        if WEIGHT_FILE.upper() == 'NEW':
            url="https://katagotraining.org/api/networks/newest_training/?format=json"
        else:
            url="https://katagotraining.org/api/networks/get_strongest/?format=json"
        response, content_ = get_page(url)
        info = json.loads(content_)
        model_url = info['model_file']
    else:
        BLOCK = get_group1('([0-9]+)b', WEIGHT_FILE)
        if BLOCK == None:
            BLOCK = get_group1('b([0-9]+)', WEIGHT_FILE)
            if BLOCK == None:
                BLOCK = get_group1('^([0-9]{1,2})(-new|s[0-9]+)?$', WEIGHT_FILE)
                if BLOCK == None:
                    print('ERROR: Block number matching failed.')
                    sys.exit(1)
        SAMPLE = get_group1('([0-9]{3,})', WEIGHT_FILE)
        if re.search('-new', WEIGHT_FILE, re.IGNORECASE):
            use_new = True
        if SAMPLE == None and not use_new:
            if BLOCK == '60':
                model_url = "https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b60c320-s9356080896-d3824355768.bin.gz"
            elif BLOCK == '30':
                model_url = "https://github.com/lightvector/KataGo/releases/download/v1.4.5/g170-b30c320x2-s4824661760-d1229536699.bin.gz"
            elif BLOCK == '20':
                model_url = "https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b20c256x2-s5303129600-d1228401921.bin.gz"
            elif BLOCK == '15':
                model_url = "https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b15c192-s1672170752-d466197061.txt.gz"
            elif BLOCK == '10':
                model_url = "https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b10c128-s1141046784-d204142634.txt.gz"
            elif BLOCK == '6':
                model_url = "https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b6c96-s175395328-d26788732.txt.gz"

# 从官网获取下载链接
if model_url == None:
    if regexp_mode:
        pattern = regexp
    elif SAMPLE:
        pattern = f'-b{BLOCK}c.*s{SAMPLE}'
    else:
        pattern = f'-b{BLOCK}c'
    # 方法1，'lxml'模块可用时，使用'lxml'模块获取模型链接
    if use_lxml:
        url = "https://katagotraining.org/networks/"
        response, content_ = get_page(url)
        # 解析 HTML 表格
        tree = lxml.html.fromstring(content_)
        table = tree.xpath('//table[@class="table mt-3"]')[0]
        max_lower_elo = -1
        i = 0
        # 遍历表格行
        for row in table.xpath('.//tr'):
            columns = row.xpath('.//td')
            if not len(columns):
                continue
            model_name = columns[0].text.strip()
            if re.search(pattern, model_name, re.IGNORECASE) == None:
                continue
            i = i + 1
            if i > 50:
                break
            if use_new or regexp_mode or SAMPLE:
                model_url = columns[3].xpath('.//a/@href')[0] if columns[3].xpath('.//a/@href') else None
                break
            elo_rating = columns[2].text.split()
            elo = float(elo_rating[0])
            uncertainty = float(elo_rating[2])
            lower_elo = round(elo - uncertainty, 1)
            if lower_elo < max_lower_elo:
                continue
            max_lower_elo = lower_elo
            model_url = columns[3].xpath('.//a/@href')[0]
    # 方法2，'lxml'模块不可用时，通过api方式获取模型链接
    else:
        base_url = "https://katagotraining.org/api/networks/?format=json&page={}" #每页20个模型
        page_number, total_numPages = get_page_number(pattern)
        numPagesSearch = 5 #第一个匹配到的模型所在的页面和之后的4页
        if use_new or regexp_mode or SAMPLE:
            numPagesSearch = 2
        if total_numPages - page_number + 1 < numPagesSearch: #避免超出最大页数
            numPagesSearch = total_numPages - page_number + 1
        urls = []
        for n in range(page_number, page_number + numPagesSearch):
            urls.append(base_url.format(n))
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(get_page, url) for url in urls] #多线程获取网页
            max_lower_elo = -1
            for future in futures:
                response, content_ = future.result()
                infos = json.loads(content_)['results']
                for info in infos:
                    model_name = info['name']
                    if re.search(pattern, model_name, re.IGNORECASE) == None:
                        continue
                    if use_new or regexp_mode or SAMPLE:
                        model_url = info['model_file']
                        break
                    lower_elo = info['log_gamma_lower_confidence']
                    if lower_elo < max_lower_elo:
                        continue
                    max_lower_elo = lower_elo
                    model_url = info['model_file']
                if (use_new or regexp_mode or SAMPLE) and model_url:
                    break
if model_url == None:
    print(f'ERROR: No URLs for weights matching "{pattern}" were found.')
    sys.exit(1)
print(f'model_url: {model_url}')

# 文件名处理
model_name = unquote(model_url.split("/")[-1].split("?")[0])
if BLOCK == None:
    BLOCK = get_group1("b([0-9]{1,2})c[0-9]{2,3}[^0-9]", model_name)
    if BLOCK == None:
        # 获取远程文件名
        response = get_page(model_url, False)
        content_disposition = response.headers.get('Content-Disposition')
        if content_disposition:
            filename = None
            parts = content_disposition.split(";")
            for part in parts:
                if part.strip().startswith('filename='):
                    filename = part.split('filename=')[-1].strip('"')
                    try:
                        filename = filename.encode("iso8859-1").decode("utf-8")
                    except:
                        pass
                    break
                elif part.strip().startswith('filename*='): #filename*=<charset>'<language>'<encoded-value>
                    filename = part.split('filename*=')[-1].strip('"')
                    encoding = filename.split("'")[0] if filename.split("'")[0] else 'utf-8'
                    filename = unquote(filename.split("'")[-1], encoding=encoding)
                    break
            if filename:
                model_name = filename
                BLOCK = get_group1("b([0-9]{1,2})c[0-9]{2,3}[^0-9]", model_name)
print(f"model_name: {model_name}")
if BLOCK:
    base_name = f'{BLOCK}b'
else:
    base_name = 'my_model'
ext = get_group1(r"(bin\.gz|txt\.gz|bin|txt|gz)$", model_name)
if ext == None:
    ext = "gz"
    print("\033[33;1mWARN\033[0m: Invalid extension. The extension has been changed to gz, which may cause an error.")
model_path = f'./data/weights/{base_name}.{ext}'

# 下载
if not os.path.isdir('./data/weights'):
    model_path = f'{base_name}.{ext}'
    if os.path.isfile(model_path):
        confirmation = input(f'\033[7m"{model_path}" already exists. Overwrite it? (Y/N) \033[0m\n')
    else:
        confirmation = input(f'\033[7mThe file will be downloaded to "{model_path}". Continue? (Y/N) \033[0m\n')
    if not confirmation.lower() == "y":
        sys.exit(1)
command = f'wget --retry-on-host-error --retry-connrefused -t3 -L "{model_url}" -O {model_path}'
status = os.system(command)
if not status == 0:
    print('ERROR: An error occurred during the download process.')
    sys.exit(1)

# 修改配置文件
if os.path.isfile('./change-config.sh') and os.path.isfile('./config/conf.yaml'):
    _ = os.system(f'sh ./change-config.sh {base_name} {model_path}')
cfg_path = './data/configs/default_gtp.cfg'
if os.path.isfile(cfg_path):
    try:
        command = 'nvidia-smi --query-gpu=name --format=csv,noheader'
        GPU_NAME = subprocess.check_output(command, shell=True, universal_newlines=True, stderr=subprocess.DEVNULL).strip()
    except:
        GPU_NAME = None
    THREADS_DICT = {}
    if GPU_NAME == "Tesla T4":
        THREADS_DICT = {
            ("CUDA", "28"): "9",
            ("CUDA", "18"): "18",
            ("CUDA", "60"): "8",
            ("CUDA", "40"): "10",
            ("CUDA", "30"): "10",
            ("CUDA", "20"): "13",
            ("CUDA", "15"): "18",
            ("CUDA", "10"): "22",
            ("CUDA", "6"): "28",
            ("TENSORRT", "28"): "13",
            ("TENSORRT", "18"): "18",
            ("TENSORRT", "60"): "10",
            ("TENSORRT", "40"): "12",
            ("TENSORRT", "30"): "14",
            ("TENSORRT", "20"): "13",
            ("TENSORRT", "15"): "20",
            ("TENSORRT", "10"): "23",
            ("TENSORRT", "6"): "28"
        }
    numThreads = THREADS_DICT.get((KATAGO_BACKEND, BLOCK), None)
    if numThreads is not None:
        _ = os.system(rf'sed -i -E "s/^(numSearchThreads =).*/\1 {numThreads}/" {cfg_path}')
    _ = os.system(f'sed -n "/^numSearchThreads/p" {cfg_path}')
