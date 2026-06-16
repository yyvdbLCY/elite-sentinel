import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# 配置
CREDS_FILE = 'credentials.json'          # 下载的服务账号 JSON
SHEET_NAME = '精锐哨兵预警记录'          # 你的表格名称

def append_alert(symbol, confidence, suggestion, base_price, timestamp=None):
    """将一条预警追加到 Google Sheets"""
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, scope)
    client = gspread.authorize(creds)
    sheet = client.open(SHEET_NAME).sheet1

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = [timestamp, symbol, confidence, base_price, suggestion]
    sheet.append_row(row)
    print(f"已记录 {symbol} 预警。")

if __name__ == "__main__":
    # 示例：手动调用（可替换为从 Telegram 消息或哨兵输出读取）
    append_alert(
        symbol="0700.HK",
        confidence=82,
        suggestion="若回踩320站稳可轻仓买入，止损317，目标335，盈亏比1:5 [基于视觉分析]",
        base_price=320.5
    )
