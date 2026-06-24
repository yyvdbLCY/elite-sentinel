import os, json, base64, io, requests, feedparser, time
import yfinance as yf
import mplfinance as mpf
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from openai import OpenAI
from datetime import datetime, timedelta

print("🚀 精銳哨兵正在啟動...")

# ==================== 配置 ====================
try:
    with open("stocks.txt", "r", encoding="utf-8") as f:
        STOCKS = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    print(f"📋 已載入 {len(STOCKS)} 隻監控標的")
except Exception as e:
    print(f"❌ 讀取 stocks.txt 失敗: {e}")
    STOCKS = []

DEEPSEEK_KEY = os.environ["DEEPSEEK_API_KEY"]
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

deepseek = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com/v1")

# Gemini 客户端
gemini_client = None
if GEMINI_KEY:
    try:
        from google import genai
        from google.genai import types
        gemini_client = genai.Client(api_key=GEMINI_KEY)
        print("✅ Gemini 客戶端已初始化")
    except Exception as e:
        print(f"⚠️ Gemini 初始化失敗: {e}")

# Hugging Face 视觉模型
HF_VISION_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_VISION_MODEL}"

# 状态记录
LAST_GEMINI_ANALYSIS = {}
LAST_GEMINI_CALL_TIME = 0
GEMINI_CALL_LOG = []
GEMINI_PER_RUN_LIMIT = 3
GEMINI_RUN_CALLS = 0

# ==================== 工具函数 ====================
def get_recent_data(symbol):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="5d", interval="1h")
    if hist.empty:
        return None, None, None, None
    if len(hist) >= 21:
        avg_vol = hist['Volume'].iloc[-21:-1].mean()
        last_vol = hist['Volume'].iloc[-1]
        vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0
    else:
        vol_ratio = 1.0
    turnover = None
    try:
        shares = ticker.info.get('sharesOutstanding')
        if shares and shares > 0:
            turnover = (hist['Volume'].iloc[-1] / shares) * 100
    except:
        pass
    open_hour_ratio = None
    try:
        today = hist[hist.index.date == hist.index[-1].date()]
        if len(today) > 0:
            first_hour_vol = today['Volume'].iloc[0]
            past_5 = hist[hist.index.date < hist.index[-1].date()]
            if len(past_5) >= 5:
                group_dates = sorted(set(past_5.index.date))[-5:]
                vols = []
                for d in group_dates:
                    day_data = past_5[past_5.index.date == d]
                    if len(day_data) > 0:
                        vols.append(day_data['Volume'].iloc[0])
                if vols:
                    avg_first = sum(vols) / len(vols)
                    open_hour_ratio = first_hour_vol / avg_first if avg_first > 0 else None
    except:
        pass
    return hist, vol_ratio, turnover, open_hour_ratio

def generate_chart_b64(symbol, hist):
    import io
    import base64
    import warnings
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import pandas as pd
    import numpy as np

    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*findfont.*')
        warnings.filterwarnings('ignore', message='.*Font.*not found.*')

        data = hist.tail(80).copy()
        if len(data) < 20:
            import mplfinance as mpf
            buf = io.BytesIO()
            savefig_config = dict(fname=buf, dpi=80, format='png', bbox_inches='tight')
            mpf.plot(data, type='candle', style='charles',
                     volume=True, figsize=(10,6), savefig=savefig_config)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode()

        # MACD
        exp12 = data['Close'].ewm(span=12, adjust=False).mean()
        exp26 = data['Close'].ewm(span=26, adjust=False).mean()
        macd_line = exp12 - exp26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line

        # RSI
        delta = data['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # 布局
        fig = plt.figure(figsize=(10, 8), dpi=100)
        gs = GridSpec(4, 1, height_ratios=[4, 1, 1, 1], hspace=0.05)

        # 主图：K线 + 三条均线
        ax1 = fig.add_subplot(gs[0])
        colors = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red' for i in range(len(data))]
        width = 0.6
        for i in range(len(data)):
            ax1.plot([i, i], [data['Low'].iloc[i], data['High'].iloc[i]], color=colors[i], linewidth=1)
            ax1.plot([i - width/2, i + width/2], [data['Open'].iloc[i], data['Open'].iloc[i]], color=colors[i], linewidth=4)
            ax1.plot([i - width/2, i + width/2], [data['Close'].iloc[i], data['Close'].iloc[i]], color=colors[i], linewidth=4)

        # 均线：MA10, MA20, MA50
        ma10 = data['Close'].rolling(window=10).mean()
        ma20 = data['Close'].rolling(window=20).mean()
        ma50 = data['Close'].rolling(window=50).mean()
        ax1.plot(range(len(data)), ma10, color='lime', linewidth=1, label='MA10')
        ax1.plot(range(len(data)), ma20, color='orange', linewidth=1, label='MA20')
        ax1.plot(range(len(data)), ma50, color='blue', linewidth=1, label='MA50')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.set_xticks(range(0, len(data), max(1, len(data)//6)))
        ax1.set_xticklabels([data.index[i].strftime('%m/%d %H:%M') for i in range(0, len(data), max(1, len(data)//6))], rotation=30, fontsize=7)
        ax1.set_ylabel('Price', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 成交量
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        colors_vol = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red' for i in range(len(data))]
        ax2.bar(range(len(data)), data['Volume'], color=colors_vol, alpha=0.7)
        ax2.set_ylabel('Volume', fontsize=8)
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.get_xticklabels(), visible=False)

        # MACD
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.plot(range(len(data)), macd_line, color='blue', linewidth=1, label='MACD')
        ax3.plot(range(len(data)), signal_line, color='red', linewidth=1, label='Signal')
        ax3.bar(range(len(data)), macd_hist, color=['green' if v >= 0 else 'red' for v in macd_hist], alpha=0.5)
        ax3.axhline(y=0, color='gray', linewidth=0.5)
        ax3.legend(loc='upper left', fontsize=7)
        ax3.set_ylabel('MACD', fontsize=8)
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.get_xticklabels(), visible=False)

        # RSI
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        ax4.plot(range(len(data)), rsi, color='purple', linewidth=1, label='RSI(14)')
        ax4.axhline(y=70, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
        ax4.axhline(y=30, color='green', linestyle='--', linewidth=0.8, alpha=0.7)
        ax4.set_ylim(0, 100)
        ax4.set_ylabel('RSI', fontsize=8)
        ax4.legend(loc='upper left', fontsize=7)
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(range(0, len(data), max(1, len(data)//6)))
        ax4.set_xticklabels([data.index[i].strftime('%m/%d %H:%M') for i in range(0, len(data), max(1, len(data)//6))], rotation=30, fontsize=7)

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

def should_skip_gemini(symbol, hist):
    if symbol not in LAST_GEMINI_ANALYSIS:
        return False
    last_time, last_price = LAST_GEMINI_ANALYSIS[symbol]
    current_price = hist['Close'].iloc[-1]
    if time.time() - last_time < 1800:
        if abs(current_price - last_price) / last_price < 0.01:
            print(f"{symbol} 近期已分析且波動小，跳過視覺分析")
            return True
    return False

# ---------- 调整后的 DeepSeek 初判（放宽阈值 + 保护视觉额度）----------
def deepseek_judge_alert(symbol, hist, vol_ratio, force=False, turnover=None, open_hour_ratio=None):
    data_text = hist.tail(24)[['Open','High','Low','Close','Volume']].to_string()
    extra_info = ""
    if turnover is not None:
        extra_info += f"\n當前換手率：{turnover:.2f}%"
    if open_hour_ratio is not None:
        extra_info += f"\n開盤第一小時量比（相對過去5日同時段均值）：{open_hour_ratio:.2f}"

    if force:
        hard_filter_block = "(用戶主動請求分析，忽略自動靜默閾值，但若以下硬指標未通過，請在 risk_factors 中註明，仍給出正常分析)"
    else:
        # 🟢 放宽后的规则：量能门槛降至1.5，且不强制 false，改为降低置信度并标注
        hard_filter_block = """## 硬性過濾規則（放寬版）
請先執行以下檢查，根據結果調整置信度，但不強制攔截（除非完全無異動）：
1. 若成交量倍數 < 1.5，且未出現明確反轉形態（頭肩底、雙底、楔形突破、早晨之星、鑷底等），請將置信度降低 15-20 點，並在 reason 中註明「量能偏弱」；若出現反轉形態，可維持原置信度。
2. 若價格未突破過去20小時最高價，且未出現破位下跌，則設置 alert: false, confidence: 10, reason: "價格未突破前高，無突破訊號"（此條為硬性過濾，除非出現上述反轉形態可覆蓋）。
3. 若換手率 > 5% 且價格漲幅 < 1%（滯漲），必須在 reason 中註明「派發風險」，但仍可繼續分析（alert 可為 true）"""

    prompt = f"""你是頂級交易員，嚴格遵循下述規則進行分析。

## 語言要求
- 你必須全程使用**繁體中文（台灣/香港習慣）**回答所有文字欄位。

## 認知誠實原則（指令約束）
- 你必須僅基於下方給出的數據回答。如果某個判斷缺乏數據依據，必須在對應的欄位中註明「無數據支持」，並將該結論的置信度設置為 0。
- 不允許編造趨勢、量價關係或形態，不允許假設數據之外的信息。若強行輸出無來源的信息，本次輸出將被視為無效。

## 事實錨定要求
在給出支撐位、壓力位、突破判斷、量價結論時，必須附帶信息來源標記。

{hard_filter_block}

## 趨勢摘要
用一句話總結過去24小時的價格運行軌跡及當前位置。

## 分析任務（僅當硬規則未觸發或已覆蓋時執行）
- 判斷是否出現放量突破、斷崖下跌、關鍵反轉等非正常邏輯異動
- 推算關鍵支撐位和壓力位（必須標明數據來源）
- 輸出置信度（0-100%），並提供3個可能誤導判斷的風險因素
- 當置信度 ≥ 80 時，你必須在 reason 中列舉至少3個相互印證的看多/看空訊號

以下是 {symbol} 近 24 小時數據：
{data_text}
當前成交量是過去 20 小時均值的 {vol_ratio:.1f} 倍。{extra_info}

嚴格輸出 JSON：
{{"alert": true/false, "reason":"...", "support": "支撐價 [來源]", "resistance": "壓力價 [來源]", "confidence": 0-100, "risk_factors": ["風險1","風險2","風險3"], "trend_summary": "趨勢摘要"}}"""

    resp = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role":"user","content":prompt}],
        temperature=0.1,
        response_format={"type":"json_object"}
    )
    return json.loads(resp.choices[0].message.content)

# ---------- 视觉分析相关函数保持不变 ----------
def gemini_vision_analysis(img_b64, symbol, trend_summary="", model='gemini-2.5-flash'):
    base_prompt = "作為首席宏觀分析師，嚴格審視以下視覺信息，並全程使用繁體中文回答。"
    if trend_summary:
        base_prompt += f"\n【背景趨勢】{trend_summary}"
    prompt = base_prompt + "\n請結合上述趨勢背景分析這張 K 線圖，重點回答：\n1. 當前形態及所處階段。\n2. 是否存在假突破、背離或騙線訊號？需與背景趨勢交叉驗證。\n3. 量價關係是否健康？給出簡潔結論。"
    image_bytes = base64.b64decode(img_b64)
    resp = gemini_client.models.generate_content(
        model=model,
        contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type="image/png")]
    )
    return resp.text

def can_call_gemini():
    global GEMINI_RUN_CALLS
    if GEMINI_RUN_CALLS >= GEMINI_PER_RUN_LIMIT:
        print(f"本次工作流已调用视觉 {GEMINI_PER_RUN_LIMIT} 次，跳过后续调用")
        return False
    return True

def record_gemini_call():
    global GEMINI_CALL_LOG, GEMINI_RUN_CALLS
    GEMINI_CALL_LOG.append(time.time())
    GEMINI_RUN_CALLS += 1

def call_gemini_with_fallback(img_b64, symbol, trend_summary=""):
    global LAST_GEMINI_CALL_TIME
    if not can_call_gemini():
        return None
    now = time.time()
    if now - LAST_GEMINI_CALL_TIME < 2:
        time.sleep(2 - (now - LAST_GEMINI_CALL_TIME))
    try:
        result = gemini_vision_analysis(img_b64, symbol, trend_summary, model='gemini-2.5-flash')
        LAST_GEMINI_CALL_TIME = time.time()
        record_gemini_call()
        return result
    except Exception as e:
        err_str = str(e)
        if '503' in err_str or 'UNAVAILABLE' in err_str:
            print(f"gemini-2.5-flash 不可用 (503)，回退到 gemini-2.0-flash...")
            try:
                result = gemini_vision_analysis(img_b64, symbol, trend_summary, model='gemini-2.0-flash')
                LAST_GEMINI_CALL_TIME = time.time()
                record_gemini_call()
                return result
            except Exception as e2:
                print(f"gemini-2.0-flash 也失败: {e2}")
                return None
        else:
            print(f"Gemini 视觉调用失败: {e}")
            return None

def hf_vision_analysis(img_b64, symbol):
    if not HF_TOKEN:
        return None
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": f"請分析這張 {symbol} 的 K 線圖，觀察形態、均線、MACD，指出假突破或騙線訊號，並用繁體中文給出簡潔結論。",
        "parameters": {"max_new_tokens": 200, "temperature": 0.1}
    }
    try:
        image_data = f"data:image/png;base64,{img_b64}"
        messages = [{"role": "user", "content": [{"type": "image", "image": image_data}, {"type": "text", "text": payload["inputs"]}]}]
        resp = requests.post(HF_API_URL, headers=headers, json={"inputs": messages, "parameters": payload["parameters"]}, timeout=20)
        if resp.status_code == 200:
            result = resp.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            elif isinstance(result, dict):
                return result.get("generated_text", "")
        else:
            print(f"Hugging Face 请求失败 ({resp.status_code}): {resp.text}")
    except Exception as e:
        print(f"Hugging Face 调用异常: {e}")
    return None

def call_vision_with_full_fallback(img_b64, symbol, trend_summary=""):
    if gemini_client:
        result = call_gemini_with_fallback(img_b64, symbol, trend_summary)
        if result:
            return result
        print("Gemini 系列失败，尝试 Hugging Face 备选引擎...")
    else:
        print("Gemini 未配置，直接尝试 Hugging Face 视觉引擎...")
    if HF_TOKEN:
        print("正在调用 Hugging Face 视觉模型...")
        result = hf_vision_analysis(img_b64, symbol)
        if result:
            print("Hugging Face 视觉分析成功")
            return result
        else:
            print("Hugging Face 视觉分析失败")
    else:
        print("未设置 HF_TOKEN，跳过 Hugging Face 视觉分析")
    return None

# ---------- 最终辩论（无变化）----------
def deepseek_debate(symbol, initial_judge, gemini_vision):
    if gemini_vision:
        expert_input = f"另一位專家（視覺分析）看完 K 線圖後指出：\n{gemini_vision}\n請結合視覺分析修正你的判斷。"
    else:
        expert_input = "系統暫無視覺分析數據。請僅基於上述量價數據，獨立給出最終的交易決策與具體建議。"

    debate_prompt = f"""你是頂級交易員，正在對一份初始分析進行最終裁決。

## 語言要求
- **重要：你必須完全使用繁體中文回覆所有 JSON 欄位。

## 認知誠實原則（指令約束）
- 你必須僅基於量價數據、視覺分析結論以及風險因素進行判斷。
- 如果某個操作建議缺乏直接的數據或圖形支撐，必須在 suggestion 中註明「該建議基於綜合經驗，缺乏直接量化指標」。
- 不得編造未在上下文中出現的支撐/壓力位或趨勢。

## 事實錨定要求
最終輸出的 suggestion 必須指明其邏輯來源，例如：
- "[基於純量價分析]"
- "[基於視覺分析對假突破的確認]"
- "[基於新聞情緒與量價共振]"

## 高級交易計劃要求
你必須輸出一個完整的交易計劃，並在 suggestion 字串中**嚴格使用換行 (\n) 將以下每個要素獨立成行**：
- 第一行：核心操作建議與方向（包括入場點、止損點、目標位、盈虧比結果，若低於 1:3 標註「博弈性價比低」）
- 第二行：
- 第三行：空間止損條件
- 第四行：時間止損條件
- 第五行：A路徑（達標）
- 第六行：B路徑（失效）
- 第七行：C路徑（橫盤）

你之前對 {symbol} 的初步判斷是：
{json.dumps(initial_judge, ensure_ascii=False)}

{expert_input}

輸出最終結論，嚴格 JSON 格式（確保 Value 為繁體中文）：
{{
  "action": "BUY/SELL/HOLD",
  "confidence": 0-100,
  "signal_breakdown": {{
    "price_action": "...",
    "volume_confirmation": "...",
    "visual_pattern": "若無視覺分析請填寫 '無視覺數據，基於純量價分析'"
  }},
  "risk_factors": ["..."],
  "suggestion": "必須用換行分隔的六行交易計劃，範例格式：
建議：基於純量價分析，建議在80.15附近買入，目標82.40，止損79.00，盈虧比1.96:1（低於3:1，博弈性價比低）

空間止損：跌破79.00離場
時間止損：若在80.00上方橫盤2個交易日無法拉回則失效
A路徑（達標）：若站穩82.40，可加倉至84.00
B路徑（失效）：若跌破79.00或時間止損觸發，訊號作廢
C路徑（橫盤）：若在79.30-80.50震盪，建議持有最多3個交易日等待突破"
}}"""
    resp = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role":"user","content":debate_prompt}],
        temperature=0.1,
        response_format={"type":"json_object"}
    )
    return json.loads(resp.choices[0].message.content)

def search_news(symbol):
    url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&sort=date"
    feed = feedparser.parse(url)
    return [{"title": e.title, "link": e.link} for e in feed.entries[:5]]

def deepseek_sentiment(symbol, news_items):
    if not news_items:
        return "暫無相關新聞"
    titles = "\n".join([n['title'] for n in news_items])
    prompt = f"關於 {symbol} 的新聞標題：\n{titles}\n判斷消息是利好出盡、真正反轉或其他，請用繁體中文一句話總結。"
    resp = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2
    )
    return resp.choices[0].message.content

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"})
    if not resp.ok:
        print(f"⚠️ Markdown 发送失败，降级纯文本")
        safe = text.replace("*", "").replace("_", "").replace("`", "")
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": safe})

def init_gsheet():
    if 'GDRIVE_CREDENTIALS' not in os.environ:
        return None
    try:
        creds_dict = json.loads(os.environ['GDRIVE_CREDENTIALS'])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ])
        client = gspread.authorize(creds)
        sheet = client.open("精锐哨兵预警记录").sheet1
        return sheet
    except Exception as e:
        print(f"Google Sheets 初始化失败: {e}")
        return None

def append_alert(sheet, symbol, confidence, suggestion, base_price):
    if sheet is None:
        return
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [timestamp, symbol, confidence, base_price, suggestion]
        sheet.append_row(row)
        print(f"已记录 {symbol} 预警到 Sheets")
    except Exception as e:
        print(f"写入 Sheets 失败: {e}")

# ==================== Telegram 指令處理 ====================
def check_telegram_commands():
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    offset_file = "last_update_id.txt"
    last_id = 0
    if os.path.exists(offset_file):
        with open(offset_file, "r") as f:
            try:
                last_id = int(f.read().strip())
            except:
                pass

    params = {"timeout": 5, "offset": last_id + 1}
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if not data.get("ok"):
            return
        for update in data["result"]:
            update_id = update["update_id"]
            message = update.get("message", {})
            text = message.get("text", "")
            chat_id = str(message.get("chat", {}).get("id", ""))
            if chat_id != TELEGRAM_CHAT_ID:
                continue

            if text.startswith("/analyze"):
                parts = text.split()
                if len(parts) >= 2:
                    target = parts[1].upper()
                    send_telegram(f"🔍 收到強制分析指令：{target}，開始分析…")
                    force_analyze(target)
                else:
                    send_telegram("⚠️ 格式錯誤，請使用：/analyze <代號>")

            with open(offset_file, "w") as f:
                f.write(str(update_id))
    except Exception as e:
        print(f"檢查 Telegram 指令時發生錯誤: {e}")

def force_analyze(symbol):
    try:
        hist, vol_ratio, turnover, open_hour_ratio = get_recent_data(symbol)
        if hist is None:
            send_telegram(f"❌ 無法取得 {symbol} 的數據，請檢查代號是否正確。")
            return

        # 強制初判（force=True）
        initial = deepseek_judge_alert(symbol, hist, vol_ratio, force=True, turnover=turnover, open_hour_ratio=open_hour_ratio)

        # 尝试视觉分析
        gemini_vision = None
        if (gemini_client or HF_TOKEN) and initial.get("confidence", 0) >= 80:
            if not should_skip_gemini(symbol, hist):
                try:
                    img_b64 = generate_chart_b64(symbol, hist)
                    trend_desc = initial.get("trend_summary", "")
                    gemini_vision = call_vision_with_full_fallback(img_b64, symbol, trend_desc)
                    if gemini_vision:
                        LAST_GEMINI_ANALYSIS[symbol] = (time.time(), hist['Close'].iloc[-1])
                except Exception as e:
                    print(f"强制分析视觉异常: {e}")

        # 最终辩论
        final = deepseek_debate(symbol, initial, gemini_vision)

        # 新闻情绪
        news = search_news(symbol)
        sentiment = deepseek_sentiment(symbol, news)

        conf_val = final.get("confidence", 50)
        if conf_val >= 80:
            conf_tag = "🟢強信號"
        elif conf_val >= 50:
            conf_tag = "🟡弱信號（未經視覺驗證）"
        else:
            conf_tag = "🔴微弱信號"

        action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(final["action"], "⚪")
        base_price = hist['Close'].iloc[-1]

        # 构建与常规预警一致的消息格式
        message = f"""
🚨 *【{symbol}】強制分析結果* | 置信度：{conf_val}% {conf_tag}

📊 *訊號拆解*
  ▪ 價格行為：{final['signal_breakdown'].get('price_action','')}
  ▪ 量能確認：{final['signal_breakdown'].get('volume_confirmation','')}
  ▪ 圖形形態：{final['signal_breakdown'].get('visual_pattern','')}

⚠️ *風險提示*
  {chr(10).join(f'  {i+1}. {r}' for i, r in enumerate(final.get('risk_factors', [])))}

📰 *新聞情緒*：{sentiment}

{action_emoji} *建議*：
{final.get('suggestion','')}

🔑 *追蹤密鑰*：`{symbol} | 觀察中 | 基準價 {base_price:.2f}`
"""
        send_telegram(message.strip())
        append_alert(None, symbol, conf_val, final.get('suggestion', ''), base_price)

    except Exception as e:
        send_telegram(f"❌ 強制分析 {symbol} 時發生錯誤：{e}")

# ==================== 主流程 ====================
def main():
    check_telegram_commands()

    sheet = init_gsheet()
    for symbol in STOCKS:
        try:
            hist, vol_ratio, turnover, open_hour_ratio = get_recent_data(symbol)
            if hist is None:
                continue

            initial = deepseek_judge_alert(symbol, hist, vol_ratio, turnover=turnover, open_hour_ratio=open_hour_ratio)
            if not initial.get("alert"):
                continue

            confidence = initial.get("confidence", 50)
            gemini_vision = None

            # 🟢 视觉分析触发门槛提升至 80
            if (gemini_client or HF_TOKEN) and confidence >= 70:
                if not should_skip_gemini(symbol, hist):
                    try:
                        img_b64 = generate_chart_b64(symbol, hist)
                        trend_desc = initial.get("trend_summary", "")
                        gemini_vision = call_vision_with_full_fallback(img_b64, symbol, trend_desc)
                        if gemini_vision:
                            LAST_GEMINI_ANALYSIS[symbol] = (time.time(), hist['Close'].iloc[-1])
                    except Exception as e:
                        print(f"视觉分析流程异常: {e}")
                else:
                    print(f"跳过 {symbol} 的视觉分析 (近期已分析且波动小)")

            final = deepseek_debate(symbol, initial, gemini_vision)

            news = search_news(symbol)
            sentiment = deepseek_sentiment(symbol, news)

            conf_val = final.get("confidence", 50)
            if conf_val >= 80:
                conf_tag = "🟢強信號"
            elif conf_val >= 50:
                conf_tag = "🟡弱信號（未經視覺驗證）"
            else:
                conf_tag = "🔴微弱信號"

            action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(final["action"], "⚪")
            base_price = hist['Close'].iloc[-1]

            message = f"""
🚨 *【{symbol}】異動預警* | 置信度：{conf_val}% {conf_tag}

📊 *訊號拆解*
  ▪ 價格行為：{final['signal_breakdown'].get('price_action','')}
  ▪ 量能確認：{final['signal_breakdown'].get('volume_confirmation','')}
  ▪ 圖形形態：{final['signal_breakdown'].get('visual_pattern','')}

⚠️ *風險提示*
  {chr(10).join(f'  {i+1}. {r}' for i, r in enumerate(final.get('risk_factors', [])))}

📰 *新聞情緒*：{sentiment}

{action_emoji} *建議*：
{final.get('suggestion','')}

🔑 *追蹤密鑰*：`{symbol} | 觀察中 | 基準價 {base_price:.2f}`
"""
            send_telegram(message.strip())
            append_alert(sheet, symbol, conf_val, final.get('suggestion', ''), base_price)
            time.sleep(1)

        except Exception as e:
            print(f"处理 {symbol} 时发生错误: {e}")

if __name__ == "__main__":
    main()
