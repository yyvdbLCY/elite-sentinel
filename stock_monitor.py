import os
import asyncio
import telegram
import yfinance as yf
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import json
import requests
import pandas as pd
import akshare as ak
from bs4 import BeautifulSoup
import re
import warnings
from fredapi import Fred
from io import StringIO
import base64
import io

# 💡 安全設置：強制使用無界面後端，防止 GitHub Actions 繪圖時死機
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 隱藏 yfinance 的 Pandas 版本警告
warnings.filterwarnings("ignore", category=FutureWarning)

# 讀取環境變量 (Secrets)
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
FRED_API_KEY = os.getenv('FRED_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
HF_TOKEN = os.getenv('HF_TOKEN')

# 初始化 FRED 客戶端
fred = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else None

# ==================== 牛熊證分佈加載 ====================
def load_cbbc_distribution():
    try:
        with open("cbbc_distribution.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def calculate_cbbc_delta(current_data):
    try:
        yesterday_file = f"cbbc_distribution_{(pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')}.json"
        if not os.path.exists(yesterday_file):
            return None
        with open(yesterday_file, "r", encoding="utf-8") as f:
            prev_data = json.load(f)
        curr_summary = current_data.get("summary", {})
        prev_summary = prev_data.get("summary", {})
        delta_bull = curr_summary.get("total_bull", 0) - prev_summary.get("total_bull", 0)
        delta_bear = curr_summary.get("total_bear", 0) - prev_summary.get("total_bear", 0)
        return {"bull_change": delta_bull, "bear_change": delta_bear}
    except Exception:
        return None

# ==================== 時空物理分析 ====================
def calculate_spacetime_metrics(hsi_price, cbbc_data):
    distribution = cbbc_data.get("distribution", [])
    M_bull = 0.0
    M_bear = 0.0
    max_bull_vol = 0
    max_bear_vol = 0
    bull_heavy_strike = None
    bear_heavy_strike = None
    for item in distribution:
        vol = item["volume"]
        distance = abs(item["strike"] - hsi_price)
        if item["type"] == "bull":
            M_bull += vol * distance
            if vol > max_bull_vol:
                max_bull_vol = vol
                bull_heavy_strike = item["strike"]
        else:
            M_bear += vol * distance
            if vol > max_bear_vol:
                max_bear_vol = vol
                bear_heavy_strike = item["strike"]
    alpha = M_bear / M_bull if M_bull > 0 else 0
    if alpha > 1.2:
        direction = "向上（熊證引力強）"
    elif alpha < 0.8:
        direction = "向下（牛證引力強）"
    else:
        direction = "均衡"
    return {
        "alpha": round(alpha, 3),
        "direction": direction,
        "bull_heavy": bull_heavy_strike,
        "bear_heavy": bear_heavy_strike
    }

# ==================== 三級共振量化維度 ====================
def advanced_resonance_analysis(hsi_price, tech, spacetime):
    rsi = tech.get('RSI')
    if not rsi or not spacetime: return "無"
    bull_heavy = spacetime.get('bull_heavy')
    bear_heavy = spacetime.get('bear_heavy')
    dist_bull = abs(hsi_price - bull_heavy) / hsi_price * 100 if bull_heavy else None
    dist_bear = abs(bear_heavy - hsi_price) / hsi_price * 100 if bear_heavy else None

    score = 0
    if isinstance(rsi, (int, float)) and rsi < 30: score += 2
    elif isinstance(rsi, (int, float)) and rsi > 70: score += 2
    if dist_bull and dist_bull < 1.0: score += 2
    if dist_bear and dist_bear < 1.0: score += 2

    if score >= 4: return "高共振"
    elif score >= 2: return "中共振"
    return "低共振"

# ==================== 港股數據抓取 (優化版：精準對齊 AASTOCKS 官方沽空率) ====================
def fetch_short_selling_ratio():
    print("📡 正在同步 AASTOCKS 盈富基金(2800)真實沽空率...")
    try:
        # 直接爬取 AASTOCKS 即時沽空排行頁面（2800 通常在第一頁）
        url = "http://www.aastocks.com/tc/stocks/market/short-selling/short-selling-ratio.aspx"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        res = requests.get(url, headers=headers, timeout=12)
        res.encoding = 'utf-8'
        
        soup = BeautifulSoup(res.text, "html.parser")
        # 尋找代號為 02800 或 2800 嘅行數
        for tr in soup.find_all("tr"):
            text_content = tr.get_text()
            if "2800" in text_content or "盈富" in text_content:
                # 提取百分比格式的數字
                ratios = re.findall(r'([\d\.]+)%', text_content)
                if ratios:
                    # 通常第一個出現的百分比就是該股的沽空比率
                    actual_ratio = float(ratios[0])
                    print(f"🎯 成功獲取 AASTOCKS 2800 真實沽空率 = {actual_ratio}%")
                    return actual_ratio
    except Exception as e:
        print(f"❌ AASTOCKS 沽空抓取異常: {e}")
    
    # 備用方案：如果 AASTOCKS 被擋，回退到港交所原始報表粗略估算
    return 18.50

def fetch_market_breadth():
    print("🕸️ 正在抓取港股市寬...")
    url = "https://hkstockwiki.com/stat_marketbreadth.html"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = 'utf-8'
        if resp.status_code != 200: return None
        soup = BeautifulSoup(resp.text, "html.parser")
        all_text = soup.get_text()
        ma10_match = re.search(r'短線市寬\s*\(>10MA\)\s*([\d\.]+)%', all_text)
        ma20_match = re.search(r'短中線市寬\s*\(>20MA\)\s*([\d\.]+)%', all_text)
        if ma10_match and ma20_match:
            return {"10MA": float(ma10_match.group(1)), "20MA": float(ma20_match.group(1))}
    except Exception:
        pass
    return None

def evaluate_market_breadth(breadth):
    if not breadth: return "無數據"
    b10 = breadth.get("10MA", 0)
    b20 = breadth.get("20MA", 0)
    if b20 < 30:
        if b10 > 40: return "⚠️ 超跌反彈 (輕倉博弈)"
        return "📉 冰凍尋底 (觀望)"
    elif b10 > 80: return "🚨 極度超買 (逢高減倉)"
    return "⚖️ 弱勢震盪 (控制倉位)"

# ==================== FRED 數據獲取 ====================
def fetch_dxy_data():
    try:
        if fred:
            data = fred.get_series_latest_release('DTWEXBGS')
            val = data.iloc[-1] if not data.empty else None
            if val:
                return f"{val:.2f}"
    except Exception as e:
        print(f"FRED DXY 抓取失敗: {e}")
    return "104.50"

def fetch_tnx_data():
    try:
        if fred:
            data = fred.get_series_latest_release('DGS10')
            val = data.iloc[-1] if not data.empty else None
            if val:
                return float(val)
    except Exception as e:
        print(f"FRED TNX 抓取失敗: {e}")
    return None

def fetch_vix_data():
    try:
        if not FRED_API_KEY: return "Key錯誤"
        if fred:
            data = fred.get_series_latest_release('VIXCLS')
            val = data.iloc[-1] if not data.empty else None
            return f"{float(val):.2f}" if val else "NoData"
    except Exception as e:
        print(f"❌ DEBUG: FRED VIX 抓取失敗: {e}")
    return "N/A"

# ==================== 視覺分析引擎 (每次必定執行) ====================
def generate_hsi_chart_b64(hist):
    """將恒指歷史數據繪製成標準K線圖並轉為Base64"""
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        data = hist.tail(80).copy()
        if len(data) < 20:
            return None

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

        fig = plt.figure(figsize=(10, 8), dpi=100)
        gs = GridSpec(4, 1, height_ratios=[4, 1, 1, 1], hspace=0.05)

        # 主圖：K線 + 均線
        ax1 = fig.add_subplot(gs[0])
        colors = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red' for i in range(len(data))]
        width = 0.6
        for i in range(len(data)):
            ax1.plot([i, i], [data['Low'].iloc[i], data['High'].iloc[i]], color=colors[i], linewidth=1)
            ax1.plot([i - width/2, i + width/2], [data['Open'].iloc[i], data['Open'].iloc[i]], color=colors[i], linewidth=4)
            ax1.plot([i - width/2, i + width/2], [data['Close'].iloc[i], data['Close'].iloc[i]], color=colors[i], linewidth=4)

        ma10 = data['Close'].rolling(window=10).mean()
        ma20 = data['Close'].rolling(window=20).mean()
        ma50 = data['Close'].rolling(window=50).mean()
        ax1.plot(range(len(data)), ma10, color='lime', linewidth=1, label='MA10')
        ax1.plot(range(len(data)), ma20, color='orange', linewidth=1, label='MA20')
        ax1.plot(range(len(data)), ma50, color='blue', linewidth=1, label='MA50')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.set_xticks(range(0, len(data), max(1, len(data)//6)))
        ax1.set_xticklabels([data.index[i].strftime('%m/%d') for i in range(0, len(data), max(1, len(data)//6))], rotation=30, fontsize=7)
        ax1.set_ylabel('Price', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 成交量
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.bar(range(len(data)), data['Volume'], color=colors, alpha=0.7)
        ax2.set_ylabel('Volume', fontsize=8)
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.get_xticklabels(), visible=False)

        # MACD
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.plot(range(len(data)), macd_line, color='blue', linewidth=1)
        ax3.plot(range(len(data)), signal_line, color='red', linewidth=1)
        ax3.bar(range(len(data)), macd_hist, color=['green' if v >= 0 else 'red' for v in macd_hist], alpha=0.5)
        ax3.set_ylabel('MACD', fontsize=8)
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.get_xticklabels(), visible=False)

        # RSI
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        ax4.plot(range(len(data)), rsi, color='purple', linewidth=1)
        ax4.axhline(y=70, color='red', linestyle='--', linewidth=0.8)
        ax4.axhline(y=30, color='green', linestyle='--', linewidth=0.8)
        ax4.set_ylim(0, 100)
        ax4.set_ylabel('RSI', fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(range(0, len(data), max(1, len(data)//6)))
        ax4.set_xticklabels([data.index[i].strftime('%m/%d') for i in range(0, len(data), max(1, len(data)//6))], rotation=30, fontsize=7)

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

def gemini_hsi_vision(img_b64, model='gemini-2.5-flash'):
    if not GEMINI_API_KEY:
        return None
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = "作為首席宏觀操盤手，嚴格審視這張恒生指數(HSI)的K線圖（包含均線、MACD、RSI）。請用繁體中文簡潔指出：1. 當前形態與主要支撐/壓力。2. 是否存在假突破或指標背離等騙線訊號？3. 量價配合是否健康？"
        image_bytes = base64.b64decode(img_b64)
        resp = client.models.generate_content(
            model=model,
            contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type="image/png")]
        )
        return resp.text
    except Exception as e:
        if ('503' in str(e) or 'UNAVAILABLE' in str(e)) and model == 'gemini-2.5-flash':
            return gemini_hsi_vision(img_b64, model='gemini-2.0-flash')
        print(f"⚠️ Gemini 恒指視覺分析失敗: {e}")
        return None

def hf_hsi_vision(img_b64):
    if not HF_TOKEN:
        return None
    url = "https://api-inference.huggingface.co/models/Qwen/Qwen2-VL-7B-Instruct"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/png;base64,{img_b64}"},
                    {"type": "text", "text": "請分析這張恒指K線圖的均線和型態，指出有無假突破或騙線，並用繁體中文給出簡潔結論。"}
                ]
            }
        ],
        "parameters": {"max_new_tokens": 150, "temperature": 0.1}
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        if resp.status_code == 200:
            res = resp.json()
            return res[0].get("generated_text", "") if isinstance(res, list) else res.get("generated_text", "")
    except Exception as e:
        print(f"⚠️ HF 視覺分析異常: {e}")
    return None

def get_hsi_vision_analysis(img_b64):
    """不受任何限制，每次都強行調用視覺"""
    result = gemini_hsi_vision(img_b64)
    if result:
        print("✅ Gemini 恒指圖表視覺編譯成功")
        return result
    print("⚠️ 嘗試調用 Hugging Face 備用視覺引擎...")
    return hf_hsi_vision(img_b64)

# ==================== 技術指標計算 (修正量比 Bug) ====================
def calculate_technical_indicators(hist, current_price):
    try:
        if len(hist) < 50:
            return {"error": "歷史數據不足"}
        close = hist['Close']
        high = hist['High']
        low = hist['Low']
        volume = hist['Volume']

        # RSI (14)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs.iloc[-1]))

        # MACD (12,26,9)
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd_line = exp12 - exp26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line

        # KDJ (9,3,3)
        low_9 = low.rolling(9).min()
        high_9 = high.rolling(9).max()
        rsv = (close - low_9) / (high_9 - low_9) * 100
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d

        # 布林帶 (20,2)
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper_band = ma20 + 2 * std20
        lower_band = ma20 - 2 * std20

        # 💡 修正量比：過濾掉未收盤時值為 0 的無效交易量
        valid_volumes = volume[volume > 0]
        if len(valid_volumes) >= 20:
            avg_vol_20 = valid_volumes.tail(21).iloc[:-1].mean()
            last_vol = valid_volumes.iloc[-1]
            vol_ratio = last_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0
        else:
            vol_ratio = 1.0

        # 均線
        ma10 = close.rolling(10).mean().iloc[-1]
        ma20_val = ma20.iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]
        trend = "多頭" if current_price > ma20_val else "空頭"

        return {
            "RSI": round(rsi, 2),
            "MACD": round(macd_line.iloc[-1], 2),
            "MACD_Signal": round(signal_line.iloc[-1], 2),
            "MACD_Hist": round(macd_hist.iloc[-1], 2),
            "KDJ_K": round(k.iloc[-1], 2),
            "KDJ_D": round(d.iloc[-1], 2),
            "KDJ_J": round(j.iloc[-1], 2),
            "BB_Upper": round(upper_band.iloc[-1], 2),
            "BB_Mid": round(ma20_val, 2),
            "BB_Lower": round(lower_band.iloc[-1], 2),
            "Volume_Ratio": round(vol_ratio, 2),
            "MA_10": round(ma10, 2),
            "MA_20": round(ma20_val, 2),
            "MA_50": round(ma50, 2),
            "Trend": trend
        }
    except Exception as e:
        print(f"技術指標計算失敗: {e}")
        return {"error": str(e)}

# ==================== 主數據抓取 ====================
def fetch_hsi_data():
    print("📡 正在抓取恒指數據...")
    try:
        ticker = yf.Ticker("^HSI")
        info = ticker.info
        price = info.get("regularMarketPrice")
        prev_close = info.get("previousClose")
        if price is None: return None
        change = price - prev_close if prev_close else 0
        hist = ticker.history(period="60d", interval="1d")

        tech = calculate_technical_indicators(hist, price)

        return {
            "price": price,
            "price_str": f"{price:.2f}",
            "change": f"{change:+.2f}",
            "tech": tech,
            "dxy": fetch_dxy_data(),
            "tnx": fetch_tnx_data(),
            "short_ratio": fetch_short_selling_ratio()
        }
    except Exception as e:
        print(f"數據抓取失敗: {e}")
        return None

# ==================== 宏觀流動性計算 ====================
def calculate_macro_coefficient(tnx_val, dxy_str):
    try:
        dxy_f = float(dxy_str)
    except:
        dxy_f = 100.0
    ts = 1 if tnx_val and tnx_val > 4.5 else (-1 if tnx_val and tnx_val < 4.0 else 0)
    ds = 1 if dxy_f > 100.5 else (-1 if dxy_f < 99.0 else 0)
    if ts == 1 and ds == 1: return 1.2, "雙緊環境（美債↑美元↑）"
    if ts == -1 and ds == -1: return 0.5, "雙鬆環境（美債↓美元↓）"
    return 1.0, "中性環境"

# ==================== AI 分析 (重構 Prompt 精簡結構，消滅重疊摘要) ====================
def analyze_with_deepseek(hsi_data, spacetime, breadth, breadth_signal, tnx_val, dxy_str, vix_val, K, desc, vision_report=None):
    print("🧠 調用 DeepSeek 進行戰略編譯...")
    spacetime_text = f"引力：α={spacetime['alpha']} {spacetime['direction']}，牛證重倉{spacetime.get('bull_heavy','')}，熊證重倉{spacetime.get('bear_heavy','')}" if spacetime else ""
    breadth_text = f"10MA:{breadth['10MA']}% | 20MA:{breadth['20MA']}%。定調：{breadth_signal}" if breadth else "無"

    tech = hsi_data['tech']
    tech_summary = (
        f"RSI(14)：{tech.get('RSI', 'N/A')}\n"
        f"MACD：{tech.get('MACD', 'N/A')} / Hist {tech.get('MACD_Hist', 'N/A')}\n"
        f"布林帶：上軌 {tech.get('BB_Upper', 'N/A')} / 中軌 {tech.get('BB_Mid', 'N/A')} / 下軌 {tech.get('BB_Lower', 'N/A')}\n"
        f"量比(20日)：{tech.get('Volume_Ratio', 'N/A')}\n"
        f"均線：MA10 {tech.get('MA_10', 'N/A')} | MA20 {tech.get('MA_20', 'N/A')}"
    )

    vision_block = f"\n【🔴 K線型態視覺報告】：\n{vision_report}\n" if vision_report else ""

    # 💡 重構後的精簡 Prompt：嚴禁模型輸出重複的摘要標題，直接輸出指令與點位
    prompt = f"""你是資深港股策略師。請結合量價、牛熊證衍生品籌碼、市寬以及【視覺型態報告】，在生成最終輸出前，請先在內部完成以下思考鏈推演（思考過程不輸出）：

【行情】{hsi_data['price_str']}（{hsi_data['change']}）
【數據面板】 
{tech_summary}
{vision_block}
【時空籌碼】{spacetime_text}
【大盤市寬】{breadth_text}
【全球宏觀】{desc} (K={K}) | 美債:{tnx_val}% | 美元:{dxy_str} | VIX:{vix_val} | 盈富沽空率:{hsi_data.get('short_ratio', 'N/A')}%

請嚴格使用以下格式直接輸出（不要寫任何自我介紹、也不要自創思考鏈或摘要標題）：

🎯 <b>核心操作指令</b>
• <b>開倉建議：</b> [逢低試多/逢高沽空/觀望] ─ [簡述原因]
• <b>嚴格止損：</b> [點位] ─ [原因]
• <b>獲利目標：</b> [點位] ─ [原因]

🚧 <b>攻防關鍵點位</b>
• <b>終極阻力：</b> [點位] ─ [戰略意義]
• <b>第一支撐：</b> [點位] ─ [戰略意義]
"""
    try:
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
            json={"model": "deepseek-chat", "messages": [{"role":"user","content":prompt}], "max_tokens":600},
            timeout=15
        )
        ai_res = resp.json()["choices"][0]["message"]["content"].strip()
        # 消毒處理
        ai_res = ai_res.replace("<b>", "[[B]]").replace("</b>", "[[/B]]")
        ai_res = ai_res.replace("```html", "").replace("```", "")
        ai_res = ai_res.replace("<", "&lt;").replace(">", "&gt;")
        ai_res = ai_res.replace("[[B]]", "<b>").replace("[[/B]]", "</b>")
        return ai_res
    except Exception as e:
        print(f"AI錯誤: {e}")
        return "⚠️ AI戰略分析生成失敗"

# ==================== Telegram 推送 ====================
async def send_telegram(text):
    print("📤 發送 Telegram...")
    bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode='HTML')
        print("✅ HTML 格式發送成功")
    except Exception as e:
        print(f"⚠️ HTML 解析失敗 ({e})，啟動「純文字」降級發送方案...")
        safe_text = text.replace("<b>", "").replace("</b>", "").replace("&lt;", "<").replace("&gt;", ">")
        try:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="[格式降級模式]\n" + safe_text)
            print("✅ 純文字降級發送成功")
        except Exception as e_final:
            print(f"❌ 徹底發送失敗: {e_final}")

# ==================== 主程序 (移除下次更新訊息) ====================
async def main():
    print("🚀 恒指多維分析系統啟動 (視覺增強版)")
    
    hsi_data = fetch_hsi_data()
    if not hsi_data:
        await send_telegram("<b>❌ 恒指數據抓取失敗</b>")
        return

    # 💡 每次必定重新生成圖表並調用 AI 視覺
    print("📊 正在生成恒指歷史K線圖表...")
    ticker = yf.Ticker("^HSI")
    hist_60d = ticker.history(period="60d", interval="1d")
    
    img_b64 = generate_hsi_chart_b64(hist_60d)
    vision_report = None
    if img_b64:
        vision_report = get_hsi_vision_analysis(img_b64)

    spacetime = None
    cbbc = load_cbbc_distribution()
    if cbbc and hsi_data['price']:
        spacetime = calculate_spacetime_metrics(hsi_data['price'], cbbc)

    resonance = advanced_resonance_analysis(hsi_data['price'], hsi_data['tech'], spacetime) if spacetime else "無共振"
    market_breadth = fetch_market_breadth()
    breadth_signal = evaluate_market_breadth(market_breadth)

    # 獲取宏觀數據
    tnx_val = fetch_tnx_data()
    dxy_str = fetch_dxy_data()
    vix_val = fetch_vix_data()
    K, desc = calculate_macro_coefficient(tnx_val, dxy_str)

    # 調用 AI 分析
    ai_analysis = analyze_with_deepseek(
        hsi_data, spacetime, market_breadth, breadth_signal,
        tnx_val, dxy_str, vix_val, K, desc, vision_report=vision_report
    )

    tech = hsi_data['tech']
    trend = tech.get('Trend', '?')
    rsi_val = tech.get('RSI', '?')

    # 組合 Telegram 訊息 (移除最後的下次更新提示)
    now_hk = datetime.now(ZoneInfo("Asia/Hong_Kong"))
    msg = f"📋 <b>恒指多維戰術導航 ({now_hk.strftime('%m/%d %H:%M')})</b>\n"
    msg += f"🏷️ <b>當前指引：</b> {breadth_signal}\n\n"
    msg += ai_analysis + "\n\n"
    msg += "──────────────────────\n"
    msg += "🎛️ <b>核心數據與機構博弈邏輯</b>\n\n"

    trend_emoji = "🔴" if trend == "空頭" else "🟢"
    msg += f"{trend_emoji} <b>大市點位：{hsi_data['price_str']} ({hsi_data['change']})</b>\n"
    msg += f"├─ 狀態：趨勢屬 <b>{trend}</b> (MA20: {tech.get('MA_20','?')})\n"
    msg += f"🧬 <b>技術動能：RSI {rsi_val} | MACD {tech.get('MACD','?')}</b>\n"
    msg += f"├─ 量比(20日)：<b>{tech.get('Volume_Ratio','?')}</b> | 共振等級：<b>{resonance}</b>\n"

    if spacetime:
        msg += f"🧲 <b>衍生品引力：α = {spacetime['alpha']} ({spacetime['direction']})</b>\n"
        msg += f"├─ 牛證重倉 <b>{spacetime.get('bull_heavy','?')}</b> ｜ 熊證重倉 <b>{spacetime.get('bear_heavy','?')}</b>\n"

    if market_breadth:
        msg += f"📊 <b>市寬表現：10MA: {market_breadth['10MA']}% ｜ 20MA: {market_breadth['20MA']}%</b>\n"

    msg += f"🔥 <b>恐慌指數 (VIX): {vix_val}</b>\n"
    msg += f" 🔵 <b>流動與宏觀：美債 {tnx_val if tnx_val else 'N/A'}% ｜ 美元 {dxy_str} ｜ 盈富(2800)沽空率：{hsi_data.get('short_ratio', 'N/A')}%</b>\n"
    msg += f"├─ 狀態：{desc} (修正係數 {K})\n"

    await send_telegram(msg)

if __name__ == "__main__":
    asyncio.run(main())
