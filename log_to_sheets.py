import os, json
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# 优先使用 GitHub Secret 中的凭证
if 'GDRIVE_CREDENTIALS' in os.environ:
    creds_dict = json.loads(os.environ['GDRIVE_CREDENTIALS'])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ])
else:
    # 本地调试备选：从文件读取
    CREDS_FILE = 'credentials.json'
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, [
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

if __name__ == "__main__":
    append_alert("0700.HK", 82, "若回踩320站稳可轻仓买入，止损317，目标335 [基于纯量价分析]", 320.5)
