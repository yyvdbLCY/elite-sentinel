import os, json
import gspread
from oauth2client.service_account import ServiceAccountCredentials

print("🧹 正在清理 Google Sheets...")

# 1. 認證
creds_json = os.environ.get("GDRIVE_CREDENTIALS")
if not creds_json:
    print("❌ 未設定 GDRIVE_CREDENTIALS")
    exit(1)

creds_dict = json.loads(creds_json)
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
])
client = gspread.authorize(creds)
sh = client.open("精锐哨兵预警记录")

# 2. 備份原始工作表
try:
    original = sh.sheet1
    backup = sh.add_worksheet("原始備份", rows=original.row_count, cols=original.col_count)
    all_data = original.get_all_values()
    if all_data:
        backup.insert_rows(all_data, 1)
        print("✅ 已備份原始數據到「原始備份」工作表")
except Exception as e:
    print(f"⚠️ 備份失敗（但不影響後續清理）: {e}")

# 3. 讀取現有數據，篩選有效行
sheet = sh.sheet1
raw_data = sheet.get_all_values()

# 有效行的條件：至少有 5 列，且第一列看起來像時間戳（包含 '-' 或 '/'）
valid_rows = []
for row in raw_data:
    if len(row) >= 5 and ('-' in row[0] or '/' in row[0]):
        valid_rows.append(row)

print(f"📋 原始行數: {len(raw_data)}，有效預警記錄: {len(valid_rows)}")

# 4. 清空工作表
sheet.clear()

# 5. 寫入表頭
headers = ["時間戳", "股票代碼", "置信度", "基準價", "建議"]
sheet.insert_row(headers, 1)

# 6. 寫入有效數據（從第 2 行開始）
if valid_rows:
    sheet.insert_rows(valid_rows, 2)

print(f"✅ 清理完成！已保留 {len(valid_rows)} 條有效記錄，表頭已設定。")