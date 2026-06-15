import os, json, base64, io, requests, feedparser, time
import yfinance as yf
import mplfinance as mpf
from openai import OpenAI
from datetime import datetime, timedelta

# ==================== 配置 ====================
# 從倉庫根目錄的 stocks.txt 讀取股票清單 (每行一個代碼，支援 # 註解)
with open("stocks.txt", "r", encoding="utf-8") as f:
    STOCKS = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

DEEPSEEK_KEY = os.environ["DEEPSEEK_API_KEY"]
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")         # 可選
TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# 初始化 AI 客戶端
deepseek = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com/v1")

# 修復 2：更新為最新的 google-genai SDK 寫法
gemini_client = None
if GEMINI_KEY:
    from google import genai
    from google.genai import types
    gemini_client = genai.Client(api_key=GEMINI_KEY)

# ==================== 工具函數 ====================
def get_recent_data(symbol):
    """拉取近5天小時線，並計算成交量異常倍數"""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="5d", interval="1h")
    if hist.empty:
        return None, None

    # 成交量異常檢測：最後一小時 vs 過去20小時均值
    if len(hist) >= 21:
        avg_vol = hist['Volume'].iloc[-21:-1].mean()
        last_vol = hist['Volume'].iloc[-1]
        vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0
    else:
        vol_ratio = 1.0
    return hist, vol_ratio

def generate_chart_b64(symbol, hist):
    """生成 K 線圖並轉 base64 (溫和壓縮版：節省流量但保留清晰度)"""
    buf = io.BytesIO()
    
    # 修復 3：設定縮小後的畫布與 DPI，保留 png 格式確保影線清晰
    savefig_config = dict(fname=buf, dpi=100, format='png')
    mpf.plot(hist.tail(50), type='candle', style='charles',
             volume=True, figsize=(8,4), savefig=savefig_config)
             
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def deepseek_judge_alert(symbol, hist, vol_ratio):
    """DeepSeek 初始判斷：是否異動 + 支撐壓力 + 置信度 + 風險"""
    data_text = hist.tail(24)[['Open','High','Low','Close','Volume']].to_string()
    prompt = f"""你是頂級交易員。以下是 {symbol} 近 24 小時數據：
{data_text}
當前成交量是過去 20 小時均值的 {vol_ratio:.1f} 倍。

請判斷是否出現需要提醒的異動（普通小波動忽略）。僅當出現放量突破、斷崖下跌、關鍵反轉等非正常邏輯時，才標記 alert: true。
推算關鍵支撐位和壓力位。
輸出你的判斷置信度（0-100%），並列出可能讓你誤判的 3 個風險因素。

嚴格輸出 JSON：
{{"alert": true/false, "reason":"...", "support": 支撐價, "resistance": 壓力價, "confidence": 85, "risk_factors": ["風險1","風險2","風險3"]}}"""

    resp = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role":"user","content":prompt}],
        temperature=0.1,
        response_format={"type":"json_object"}
    )
    return json.loads(resp.choices[0].message.content)

def gemini_vision_analysis(img_b64, symbol):
    """Gemini 視覺看圖，尋找騙線信號 (使用最新 SDK)"""
    prompt = ("請像人類專家一樣分析這張 K 線圖，觀察形態、均線、MACD，"
              "特別注意是否存在假突破、背離或騙線信號，給出簡潔結論。")
              
    # 修復 4：將 Base64 轉回 bytes 交給新版 SDK 處理
    image_bytes = base64.b64decode(img_b64)
    
    resp = gemini_client.models.generate_content(
        model='gemini-1.5-flash',
        contents=[
            prompt,
            types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        ]
    )
    return resp.text

def deepseek_debate(symbol, initial_judge, gemini_vision):
    """最終決策：優先參考 Gemini 視覺，若無則由 DeepSeek 獨立給出具體建議"""
    
    # 動態調整提示詞：有視覺就辯論，沒視覺就單獨總結
    if gemini_vision:
        expert_input = f"另一位專家（Gemini 視覺）看完 K 線圖後指出：\n{gemini_vision}\n請結合視覺分析修正你的判斷。"
    else:
        expert_input = "系統暫無視覺分析數據。請你僅基於上述量價數據（包含支撐壓力與異動原因），獨立給出最終的交易決策與具體建議。"

    debate_prompt = f"""你之前對 {symbol} 的判斷是：
{json.dumps(initial_judge, ensure_ascii=False)}

{expert_input}

輸出最終結論，嚴格 JSON 格式：
{{
  "action": "BUY/SELL/HOLD",
  "confidence": 0-100,
  "signal_breakdown": {{
    "price_action": "...",
    "volume_confirmation": "...",
    "visual_pattern": "若無視覺分析請填寫'無視覺數據，基於純量價分析'"
  }},
  "risk_factors": ["..."],
  "suggestion": "給交易員的一句明確建議（必須具體，例如：量縮回踩，建議等待突破xx元再介入，或跌破xx元止損）"
}}"""
    resp = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role":"user","content":debate_prompt}],
        temperature=0.1,
        response_format={"type":"json_object"}
    )
    return json.loads(resp.choices[0].message.content)

def search_news(symbol):
    """Google News RSS 標題"""
    url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&sort=date"
    feed = feedparser.parse(url)
    items = [{"title": e.title, "link": e.link} for e in feed.entries[:5]]
    return items

def deepseek_sentiment(symbol, news_items):
    if not news_items:
        return "暫無相關新聞"
    titles = "\n".join([n['title'] for n in news_items])
    prompt = f"關於 {symbol} 的新聞標題：\n{titles}\n判斷消息是利好出盡、真正反轉或其他，一句話總結。"
    resp = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2
    )
    return resp.choices[0].message.content

def send_telegram(text):
    """發送 Telegram 訊息"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"})

# ==================== 主流程 ====================
def main():
    for symbol in STOCKS:
        try:
            hist, vol_ratio = get_recent_data(symbol)
            if hist is None:
                continue

            # 1. DeepSeek 初判
            initial = deepseek_judge_alert(symbol, hist, vol_ratio)
            if not initial.get("alert"):
                continue

            # 2. Gemini 視覺（若有）
            gemini_vision = None
            if gemini_client:
                try:
                    img_b64 = generate_chart_b64(symbol, hist)
                    gemini_vision = gemini_vision_analysis(img_b64, symbol)
                except Exception as e:
                    print(f"Gemini 視覺失敗: {e}")
            
            # 3. 最終決策（注意：這行必須與上方的 'if gemini_client:' 對齊）
            final = deepseek_debate(symbol, initial, gemini_vision)

            # 4. 新聞情緒
            news = search_news(symbol)
            sentiment = deepseek_sentiment(symbol, news)

            # 5. 組裝結構化推送
            action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(final["action"], "⚪")
            message = f"""
🚨 *【{symbol}】異動預警* | 置信度：{final['confidence']}%

📊 *信號拆解*
  ▪ 價格行為：{final['signal_breakdown'].get('price_action','')}
  ▪ 量能確認：{final['signal_breakdown'].get('volume_confirmation','')}
  ▪ 圖形形態：{final['signal_breakdown'].get('visual_pattern','')}

⚠️ *風險提示*
  {chr(10).join(f'  {i+1}. {r}' for i, r in enumerate(final.get('risk_factors', [])))}

📰 *新聞情緒*：{sentiment}

{action_emoji} *建議*：{final.get('suggestion','')}
"""
            send_telegram(message.strip())
            time.sleep(1)  # 禮貌間隔

        except Exception as e:
            print(f"處理 {symbol} 時發生錯誤: {e}")

if __name__ == "__main__":
    main()
