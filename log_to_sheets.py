import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# 若在 GitHub Actions 中，从环境变量读取凭证
if 'GDRIVE_CREDENTIALS' in os.environ:
    import json
    creds_dict = json.loads(os.environ['GDRIVE_CREDENTIALS'])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ])
else:
    creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ])

client = gspread.authorize(creds)
sheet = client.open("精锐哨兵预警记录").sheet1

def append_alert(symbol, confidence, suggestion, base_price, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [timestamp, symbol, confidence, base_price, suggestion]
    sheet.append_row(row)
    print(f"已记录 {symbol} 预警。")

# 测试一条
append_alert("0700.HK", 82, "若回踩320站稳可轻仓买入，止损317，目标335 [基于纯量价分析]", 320.5)
