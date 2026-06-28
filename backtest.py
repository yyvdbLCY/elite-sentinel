import os, json, re, time
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

print("📊 回測系統啟動")

# ==================== 配置 ====================
GDRIVE_CREDENTIALS = os.environ.get("GDRIVE_CREDENTIALS")
if not GDRIVE_CREDENTIALS:
    print("❌ 未設定 GDRIVE_CREDENTIALS")
    exit(1)

creds_dict = json.loads(GDRIVE_CREDENTIALS)
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
])
client = gspread.authorize(creds)
sheet = client.open("精锐哨兵预警记录").sheet1

# ==================== 读取预警记录 ====================
records = sheet.get_all_records()
print(f"📋 讀取到 {len(records)} 條預警記錄")

# ==================== 工具函数 ====================
def extract_numbers_from_suggestion(suggestion):
    """
    从建议文本中提取关键数字：入场价、止损位、目标位
    支持多种格式，如：
    - 建議在80.15附近買入，目標82.40，止損79.00
    - 入場位80.15，止損79.00，目標82.40
    """
    entry = None
    stop_loss = None
    target = None

    # 入场价
    entry_patterns = [
        r'(?:買入[價价]?|入場[價价]?|建議在|建議於)\s*([\d.]+)',
        r'(?:入場|買入)\s*[：:]\s*([\d.]+)',
        r'在\s*([\d.]+)\s*附近[買买]入',
    ]
    for pat in entry_patterns:
        match = re.search(pat, suggestion)
        if match:
            entry = float(match.group(1))
            break

    # 止损
    stop_patterns = [
        r'(?:止損|止蚀|止蚀位|空間止損)[：:\s]*[跌破]?\s*([\d.]+)',
        r'止損[設设]?[在於]?\s*([\d.]+)',
        r'跌破\s*([\d.]+)\s*[離离]場',
    ]
    for pat in stop_patterns:
        match = re.search(pat, suggestion)
        if match:
            stop_loss = float(match.group(1))
            break

    # 目标
    target_patterns = [
        r'(?:目標|目標位|獲利目標)[：:\s]*[看至]?\s*([\d.]+)',
        r'目標[價价]?[設设]?[在於]?\s*([\d.]+)',
        r'看至\s*([\d.]+)',
    ]
    for pat in target_patterns:
        match = re.search(pat, suggestion)
        if match:
            target = float(match.group(1))
            break

    return entry, stop_loss, target


def backtest_signal(alert_time, symbol, suggestion):
    """对单条预警进行回测，返回结果字典"""
    entry, stop_loss, target = extract_numbers_from_suggestion(suggestion)
    if not entry or not stop_loss or not target:
        return {"status": "數據不足", "reason": "無法提取入場/止損/目標"}

    # 回看周期：港股 10 个交易小时（2天），美股 13 个交易小时（2天）
    is_hk = symbol.endswith(".HK")
    hours = 10 if is_hk else 13
    end_time = alert_time + timedelta(hours=hours)

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=alert_time, end=end_time, interval="1h")
        if hist.empty:
            return {"status": "無數據", "reason": "yfinance 無返回數據"}
    except Exception as e:
        return {"status": "錯誤", "reason": str(e)}

    # 逐条遍历，判断先触达止损还是目标
    for idx, row in hist.iterrows():
        high = row['High']
        low = row['Low']
        if low <= stop_loss:
            return {
                "status": "止損",
                "entry": entry,
                "stop_loss": stop_loss,
                "target": target,
                "trigger_time": str(idx),
                "trigger_price": stop_loss
            }
        if high >= target:
            return {
                "status": "止盈",
                "entry": entry,
                "stop_loss": stop_loss,
                "target": target,
                "trigger_time": str(idx),
                "trigger_price": target
            }

    # 时间窗口内未触达
    return {
        "status": "橫盤失效",
        "entry": entry,
        "stop_loss": stop_loss,
        "target": target,
        "trigger_time": None,
        "trigger_price": None
    }


# ==================== 遍历所有预警并回测 ====================
results = []
for record in records:
    timestamp_str = record.get("timestamp", record.get("時間戳", ""))
    symbol = record.get("symbol", record.get("股票代碼", ""))
    suggestion = record.get("suggestion", record.get("建議", ""))
    confidence = record.get("confidence", record.get("置信度", ""))

    if not timestamp_str or not symbol or not suggestion:
        continue

    try:
        alert_time = pd.Timestamp(timestamp_str).tz_localize("Asia/Hong_Kong")
    except:
        continue

    result = backtest_signal(alert_time, symbol, suggestion)
    result["alert_time"] = timestamp_str
    result["symbol"] = symbol
    result["confidence"] = confidence
    result["suggestion"] = suggestion[:100]  # 截取前100字符
    results.append(result)

    print(f"  {symbol} @ {timestamp_str} → {result['status']}")
    time.sleep(0.5)  # 避免 yfinance 请求过快

# ==================== 统计与报告 ====================
total = len(results)
win = sum(1 for r in results if r["status"] == "止盈")
lose = sum(1 for r in results if r["status"] == "止損")
flat = sum(1 for r in results if r["status"] == "橫盤失效")
no_data = sum(1 for r in results if r["status"] in ("數據不足", "無數據", "錯誤"))

print(f"\n📊 回测统计：总 {total} 条")
print(f"  止盈: {win} | 止損: {lose} | 橫盤失效: {flat} | 無法回測: {no_data}")
if win + lose > 0:
    win_rate = win / (win + lose) * 100
    print(f"  有效信號勝率: {win_rate:.1f}%")

# ==================== 写入回测记录到新工作表 ====================
try:
    backtest_sheet = client.open("精锐哨兵预警记录").worksheet("回测记录")
except:
    backtest_sheet = client.open("精锐哨兵预警记录").add_worksheet("回测记录", 1000, 10)

# 写入表头
headers = ["预警时间", "股票", "状态", "入场", "止損", "目标", "触发时间", "触发价格", "置信度", "建议摘要"]
backtest_sheet.insert_row(headers, 1)

# 写入数据
for r in results:
    row = [
        r.get("alert_time", ""),
        r.get("symbol", ""),
        r.get("status", ""),
        r.get("entry", ""),
        r.get("stop_loss", ""),
        r.get("target", ""),
        r.get("trigger_time", ""),
        r.get("trigger_price", ""),
        r.get("confidence", ""),
        r.get("suggestion", ""),
    ]
    backtest_sheet.insert_row(row, 2)  # 插入到第二行（表头下方）

print("✅ 回测结果已写入 Google Sheets「回测记录」工作表")