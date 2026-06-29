import os, json, base64, io, requests, feedparser, time
import yfinance as yf
import mplfinance as mpf
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from openai import OpenAI
from datetime import datetime, timedelta

print("🚀 精銳哨兵正在啟動...")

# ------- 配置與初始化 -------
def load_stock_list(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    except Exception:
        return []

STOCKS_HK = load_stock_list("stocks_hk.txt")
STOCKS_US = load_stock_list("stocks_us.txt")
print(f"📋 港股 {len(STOCKS_HK)} 隻 | 美股 {len(STOCKS_US)} 隻")

DEEPSEEK_KEY = os.environ["DEEPSEEK_API_KEY"]
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY")  # 智譜備用視覺引擎
TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
MARKET = os.environ.get("MARKET", "").upper()

deepseek = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com/v1")

# ------- Gemini 客戶端初始化 -------
gemini_client = None
if GEMINI_KEY:
    try:
        from google import genai
        from google.genai import types
        gemini_client = genai.Client(api_key=GEMINI_KEY)
        print("✅ Gemini 客戶端已初始化")
    except Exception as e:
        print(f"⚠️ Gemini 初始化失敗: {e}")

# ------- 智譜視覺模型配置 -------
ZHIPU_VISION_MODEL = "glm-4.6v-flash"
ZHIPU_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

# ------- 狀態記錄 -------
LAST_GEMINI_ANALYSIS = {}
LAST_GEMINI_CALL_TIME = 0
GEMINI_CALL_LOG = []
GEMINI_PER_RUN_LIMIT = 3
GEMINI_RUN_CALLS = 0
LAST_GEMINI_ERROR = {}

# ------- 工具函數：數據獲取 -------
def get_recent_data(symbol):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="5d", interval="1h")
    if hist.empty:
        return None, None, None, None, None
    if len(hist) >= 21:
        avg_vol = hist['Volume'].iloc[-21:-1].mean()
        last_vol = hist['Volume'].iloc[-1]
        vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0
    else:
        vol_ratio = 1.0
    turnover = None
    try:
        shares = ticker.fast_info.get('shares') or ticker.info.get('sharesOutstanding')
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
    # 新增：獲取日線數據用於多時間框架過濾
    daily_trend = None
    try:
        daily = ticker.history(period="3mo", interval="1d")
        if not daily.empty and len(daily) >= 20:
            daily_ma20 = daily['Close'].rolling(window=20).mean().iloc[-1]
            daily_close = daily['Close'].iloc[-1]
            daily_trend = {
                'ma20': daily_ma20,
                'close': daily_close,
                'above_ma20': daily_close > daily_ma20,
            }
    except:
        pass
    return hist, vol_ratio, turnover, open_hour_ratio, daily_trend

# ------- 工具函數：K線圖生成 -------
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

# ------- 視覺分析輔助函數 -------
def process_symbol(symbol, sheet):
    try:
        hist, vol_ratio, turnover, open_hour_ratio, daily_trend = get_recent_data(symbol)
        if hist is None:
            return

        # ---- 第1步：Python硬性過濾（攔截無效信號，節省API額度）----
        passed, filter_reason = hard_filter_signals(hist, vol_ratio, turnover)
        if not passed:
            print(f"⏭️ {symbol} 被硬性過濾：{filter_reason}")
            return

        # ---- 第2步：計算量化指標（形態、支撐壓力）----
        patterns = detect_patterns(hist)
        key_levels = calculate_key_levels(hist)

        # ---- 第3步：DeepSeek初判（基於預計算指標）----
        initial = deepseek_judge_alert(
            symbol, hist, vol_ratio,
            turnover=turnover,
            open_hour_ratio=open_hour_ratio,
            daily_trend=daily_trend,
            patterns=patterns,
            key_levels=key_levels
        )

        if not initial.get("alert"):
            return

        # ---- 第3.5步：多時間框架趨勢過濾（逆勢信號降級）----
        is_counter_trend, mtf_reason = mtf_trend_filter(initial.get('action', ''), daily_trend)
        if is_counter_trend:
            current_conf = initial.get("confidence", 50)
            initial["confidence"] = max(10, current_conf - 30)
            initial["reason"] = f"{initial.get('reason', '')} | {mtf_reason}"
            if initial["confidence"] < 50:
                print(f"⏭️ {symbol} 逆勢信號置信度過低，跳過：{mtf_reason}")
                return

        confidence = initial.get("confidence", 50)

        # ---- 第4步：新聞情緒分析（前置，供辯論使用）----
        news = search_news(symbol)
        sentiment = deepseek_sentiment(symbol, news)

        # ---- 第5步：視覺分析（提高門檻至80，增加形態校驗）----
        gemini_vision = None
        if (gemini_client or ZHIPU_API_KEY) and confidence >= 80 and has_visual_worthy_patterns(patterns):
            if not should_skip_gemini(symbol, hist):
                try:
                    img_b64 = generate_chart_b64(symbol, hist)
                    trend_desc = initial.get("trend_summary", "")
                    gemini_vision = call_vision_with_full_fallback(img_b64, symbol, trend_desc)
                    if gemini_vision:
                        LAST_GEMINI_ANALYSIS[symbol] = (time.time(), hist['Close'].iloc[-1])
                except Exception as e:
                    print(f"視覺分析流程異常: {e}")
            else:
                print(f"跳過 {symbol} 的視覺分析（近期已分析且波動小）")

        # ---- 第6步：DeepSeek最終辯論（結合新聞情緒與視覺分析）----
        final = deepseek_debate(symbol, initial, gemini_vision, sentiment)

        # ---- 第7步：Python盈虧比複核（低於1.5則降級）----
        final = check_risk_reward(final, key_levels)

        # 視覺形態補充：若視覺未提供或置信度不足，顯示純量價說明或視覺錯誤原因
        final.setdefault('signal_breakdown', {})
        vis_pattern = final['signal_breakdown'].get('visual_pattern', '')
        if not vis_pattern or vis_pattern.strip() == '':
            if final.get('confidence', 0) < 80:
                if symbol in LAST_GEMINI_ERROR:
                    final['signal_breakdown']['visual_pattern'] = f"視覺模型失敗：{LAST_GEMINI_ERROR[symbol]}，因此基於純量價分析"
                else:
                    final['signal_breakdown']['visual_pattern'] = "由於置信度不足基於純量價分析"

        # ---- 第8步：發送預警----
        conf_val = final.get("confidence", 50)
        if conf_val >= 80:
            conf_tag = "🟢強信號"
        elif conf_val >= 50:
            conf_tag = "🟡弱信號（未經視覺驗證）"
        else:
            conf_tag = "🔴微弱信號"

        action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(final.get("action"), "⚪")
        base_price = hist['Close'].iloc[-1]

        message = f"""
🚨 *【{symbol}】異動預警* | 置信度：{conf_val}% {conf_tag} 📊 *訊號拆解*  ▪ 價格行為：{final['signal_breakdown'].get('price_action','')}  ▪ 量能確認：{final['signal_breakdown'].get('volume_confirmation','')}  ▪ 圖形形態：{final['signal_breakdown'].get('visual_pattern','')} ⚠️ *風險提示*  {chr(10).join(f'  {i+1}. {r}' for i, r in enumerate(final.get('risk_factors', [])))} 📰 *新聞情緒*：{sentiment} {action_emoji} *建議*：{final.get('suggestion','')} 🔑 *追蹤密鑰*：`{symbol} | 觀察中 | 基準價 {base_price:.2f}`"""
        send_telegram(message.strip())
        append_alert(sheet, symbol, conf_val, final.get('suggestion', ''), base_price)
        time.sleep(1)

    except Exception as e:
        print(f"處理 {symbol} 時發生錯誤: {e}")
    # 吞沒形態
    if len(c) >= 2:
        if c[-1] > o[-1] and c[-2] < o[-2] and c[-1] >= o[-2] and o[-1] <= c[-2]:
            patterns.append("看漲吞沒")
        elif c[-1] < o[-1] and c[-2] > o[-2] and c[-1] <= o[-2] and o[-1] >= c[-2]:
            patterns.append("看跌吞沒")
    # 十字星
    if len(c) >= 1:
        body = abs(c[-1] - o[-1])
        range_ = h[-1] - l[-1]
        if range_ > 0 and body / range_ < 0.1:
            patterns.append("十字星")
    # 長上影線 / 長下影線
    if len(c) >= 1:
        body = abs(c[-1] - o[-1])
        if body > 0:
            upper_shadow = h[-1] - max(c[-1], o[-1])
            lower_shadow = min(c[-1], o[-1]) - l[-1]
            if upper_shadow / body > 2:
                patterns.append("長上影線")
            if lower_shadow / body > 2:
                patterns.append("長下影線")
    # 早晨之星 / 黃昏之星
    if len(c) >= 3:
        body1 = abs(c[-3] - o[-3])
        body2 = abs(c[-2] - o[-2])
        if body1 > 0:
            if c[-3] < o[-3] and body2 < body1 * 0.5 and c[-1] > o[-1] and c[-1] > (o[-3] + c[-3]) / 2:
                patterns.append("早晨之星")
            if c[-3] > o[-3] and body2 < body1 * 0.5 and c[-1] < o[-1] and c[-1] < (o[-3] + c[-3]) / 2:
                patterns.append("黃昏之星")
    # MACD背離檢測
    if len(hist) >= 35:
        exp12 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp26 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd_line = exp12 - exp26
        lookback = min(15, len(hist) - 1)
        price_now = hist['High'].iloc[-1]
        price_past = hist['High'].iloc[-lookback]
        macd_now = macd_line.iloc[-1]
        macd_past = macd_line.iloc[-lookback]
        if price_now > price_past and macd_now < macd_past:
            patterns.append("MACD頂背離")
        elif price_now < price_past and macd_now > macd_past:
            patterns.append("MACD底背離")
    return patterns if patterns else ["無明顯形態"]

# ------- 新增：支撐壓力計算 -------
def calculate_key_levels(hist):
    """計算關鍵支撐位和壓力位"""
    if len(hist) < 20:
        return None
    recent = hist.tail(20)
    resistance = recent['High'].max()
    support = recent['Low'].min()
    current = hist['Close'].iloc[-1]
    return {
        'resistance': resistance,
        'support': support,
        'current_price': current,
        'dist_to_resistance': (resistance - current) / current * 100 if current > 0 else 0,
        'dist_to_support': (current - support) / current * 100 if current > 0 else 0,
    }

# ------- 新增：盈虧比複核 -------
def check_risk_reward(final, key_levels):
    """複核盈虧比，若低於1.5則降級置信度"""
    if not key_levels or not final.get('action'):
        return final
    current = key_levels['current_price']
    support = key_levels['support']
    resistance = key_levels['resistance']
    if final['action'] == 'BUY':
        potential_loss = current - support
        potential_gain = resistance - current
        if potential_loss > 0 and potential_gain > 0:
            rr = potential_gain / potential_loss
            if rr < 1.5:
                final['confidence'] = min(final.get('confidence', 50), 40)
                final['suggestion'] = f"⚠️ 盈虧比僅{rr:.1f}:1（低於1.5:1標準），信號已降級\n" + final.get('suggestion', '')
    elif final['action'] == 'SELL':
        potential_loss = resistance - current
        potential_gain = current - support
        if potential_loss > 0 and potential_gain > 0:
            rr = potential_gain / potential_loss
            if rr < 1.5:
                final['confidence'] = min(final.get('confidence', 50), 40)
                final['suggestion'] = f"⚠️ 盈虧比僅{rr:.1f}:1（低於1.5:1標準），信號已降級\n" + final.get('suggestion', '')
    return final

# ------- 新增：多時間框架趨勢過濾 -------
def mtf_trend_filter(action, daily_trend):
    """多時間框架過濾：1h信號與日線趨勢相反時返回降級標記"""
    if not daily_trend:
        return False, "無日線數據，跳過MTF過濾"
    if action == 'BUY' and not daily_trend['above_ma20']:
        return True, f"逆勢信號（1h看多但日線在MA20下方={daily_trend['ma20']:.2f}）"
    if action == 'SELL' and daily_trend['above_ma20']:
        return True, f"逆勢信號（1h看空但日線在MA20上方={daily_trend['ma20']:.2f}）"
    return False, "順勢信號"

# ------- 新增：判斷是否有值得視覺分析的形態 -------
def has_visual_worthy_patterns(patterns):
    """只有出現關鍵形態才調用視覺模型，節省API額度"""
    key_patterns = {"看漲吞沒", "看跌吞沒", "早晨之星", "黃昏之星", "MACD頂背離", "MACD底背離", "十字星"}
    return any(p in key_patterns for p in patterns)

# ------- DeepSeek 初判引擎 -------
def deepseek_judge_alert(symbol, hist, vol_ratio, force=False, turnover=None,
                         open_hour_ratio=None, daily_trend=None, patterns=None, key_levels=None):
    data_text = hist.tail(24)[['Open','High','Low','Close','Volume']].to_string()
    # 構建日線趨勢背景
    trend_info = ""
    if daily_trend:
        trend_status = "多頭排列（日線收盤在MA20上方）" if daily_trend['above_ma20'] else "空頭排列（日線收盤在MA20下方）"
        trend_info = f"\n日線趨勢：{trend_status}，日線MA20={daily_trend['ma20']:.2f}，日線收盤={daily_trend['close']:.2f}"
    # 構建形態信息
    pattern_info = f"\n代碼檢測到的K線形態：{', '.join(patterns) if patterns else '無明顯形態'}"
    # 構建支撐壓力信息
    level_info = ""
    if key_levels:
        level_info = f"\n關鍵支撐位：{key_levels['support']:.2f}（距當前價格{key_levels['dist_to_support']:.1f}%）"
        level_info += f"\n關鍵壓力位：{key_levels['resistance']:.2f}（距當前價格{key_levels['dist_to_resistance']:.1f}%）"
    extra_info = ""
    if turnover is not None:
        extra_info += f"\n當前換手率：{turnover:.2f}%"
    if open_hour_ratio is not None:
        extra_info += f"\n開盤第一小時量比：{open_hour_ratio:.2f}"
    if force:
        filter_note = "（用戶主動請求分析，硬性過濾已跳過，但以下量化指標仍供參考）"
    else:
        filter_note = "（已通過代碼層硬性過濾，量價基礎條件已滿足）"
    prompt = f"""你是頂級交易員，基於以下已計算好的量化指標進行分析。 ## 語言要求- 全程使用繁體中文（台灣/香港習慣）回答所有文字欄位。 ## 認知誠實原則（指令約束）- 僅基於下方給出的數據回答。缺乏數據依據的判斷必須註明「無數據支持」，置信度設為0。- 不允許編造趨勢、量價關係或形態。 ## 已計算的量化指標{filter_note}{trend_info}{pattern_info}{level_info}{extra_info} 以下是 {symbol} 近 24 小時數據：{data_text}當前成交量是過去 20 小時均值的 {vol_ratio:.1f} 倍。 ## 分析任務- 基於上述已計算好的形態、支撐壓力和趨勢數據，判斷是否存在有效信號- 結合日線趨勢判斷是順勢還是逆勢（逆勢信號置信度應大幅降低）- 輸出置信度（0-100%），並提供3個可能誤導判斷的風險因素- 當置信度 ≥ 80 時，必須列舉至少3個相互印證的看多/看空訊號 嚴格輸出 JSON：{{"alert": true/false, "action": "BUY/SELL/HOLD", "reason":"...", "support": "支撐價 [來源]", "resistance": "壓力價 [來源]", "confidence": 0-100, "risk_factors": ["風險1","風險2","風險3"], "trend_summary": "趨勢摘要"}}"""
    resp = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role":"user","content":prompt}],
        temperature=0.1,
        response_format={"type":"json_object"}
    )
    return json.loads(resp.choices[0].message.content)

# ------- Gemini 視覺分析引擎 -------
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

# ------- 視覺分析配額與回退機制 -------
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
        LAST_GEMINI_ERROR.pop(symbol, None)
        return result
    except Exception as e:
        err_str = str(e)
        LAST_GEMINI_ERROR[symbol] = err_str
        if '503' in err_str or 'UNAVAILABLE' in err_str:
            print(f"gemini-2.5-flash 不可用 (503)，回退到 gemini-2.0-flash...")
            try:
                result = gemini_vision_analysis(img_b64, symbol, trend_summary, model='gemini-2.0-flash')
                LAST_GEMINI_CALL_TIME = time.time()
                record_gemini_call()
                LAST_GEMINI_ERROR.pop(symbol, None)
                return result
            except Exception as e2:
                err2 = str(e2)
                print(f"gemini-2.0-flash 也失败: {e2}")
                LAST_GEMINI_ERROR[symbol] = err2
                return None
        else:
            print(f"Gemini 视觉调用失败: {e}")
            return None

# ------- 智譜視覺分析（備用引擎） -------
def zhipu_vision_analysis(img_b64, symbol):
    if not ZHIPU_API_KEY:
        return None
    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": ZHIPU_VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/png;base64,{img_b64}"},
                    {"type": "text", "text": f"請分析這張 {symbol} 的 K 線圖，觀察形態、均線、MACD，指出假突破或騙線訊號，並用繁體中文給出簡潔結論。"}
                ]
            }
        ],
        "max_tokens": 200,
        "temperature": 0.1
    }
    try:
        resp = requests.post(ZHIPU_API_URL, headers=headers, json=payload, timeout=20)
        if resp.status_code == 200:
            result = resp.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            print(f"智譜 API 請求失敗 ({resp.status_code}): {resp.text}")
    except Exception as e:
        print(f"智譜視覺調用異常: {e}")
    return None

# ------- 視覺分析完整回退鏈（Gemini → 智譜） -------
def call_vision_with_full_fallback(img_b64, symbol, trend_summary=""):
    if gemini_client:
        result = call_gemini_with_fallback(img_b64, symbol, trend_summary)
        if result:
            LAST_GEMINI_ERROR.pop(symbol, None)
            return result
        print("Gemini 系列失败，尝试智譜备用引擎...")
    else:
        print("Gemini 未配置，直接尝试智譜视觉引擎...")
    if ZHIPU_API_KEY:
        print("正在调用智譜视觉模型...")
        try:
            result = zhipu_vision_analysis(img_b64, symbol)
            if result:
                print("智譜视觉分析成功")
                LAST_GEMINI_ERROR.pop(symbol, None)
                return result
            else:
                print("智譜视觉分析失败")
                LAST_GEMINI_ERROR[symbol] = "智譜返回空結果"
        except Exception as e:
            LAST_GEMINI_ERROR[symbol] = str(e)
            print(f"智譜视觉調用異常: {e}")
    else:
        print("未设置 ZHIPU_API_KEY，跳过备用视觉分析")
    return None

# ------- 最終辯論與交易計劃生成 -------
def deepseek_debate(symbol, initial_judge, gemini_vision, sentiment="暫無相關新聞"):
    if gemini_vision:
        expert_input = f"另一位專家（視覺分析）看完 K 線圖後指出：\n{gemini_vision}\n請結合視覺分析修正你的判斷。"
    else:
        expert_input = "系統暫無視覺分析數據。請僅基於上述量價數據，獨立給出最終的交易決策與具體建議。"

    debate_prompt = f"""你是頂級交易員，正在對一份初始分析進行最終裁決。

## 語言要求
- **重要：你必須完全使用繁體中文回覆所有 JSON 欄位。**

## 認知誠實原則（指令約束）
- 你必須僅基於量價數據、視覺分析結論、新聞情緒以及風險因素進行判斷。
- 如果某個操作建議缺乏直接的數據或圖形支撐，必須在 suggestion 中註明「該建議基於綜合經驗，缺乏直接量化指標」。
- 不得編造未在上下文中出現的支撐/壓力位或趨勢。

## 事實錨定要求最終輸出的 suggestion 必須指明其邏輯來源，例如：
- "[基於純量價分析]"
- "[基於視覺分析對假突破的確認]"
- "[基於新聞情緒與量價共振]"

## 新聞情緒參考{sentiment}

## 高級交易計劃要求
你必須輸出一個完整的交易計劃，並在 suggestion 字串中**嚴格使用換行 (\\n) 將以下每個要素獨立成行**：
- 第一行：核心操作建議與方向（包括入場點、止損點、目標位、盈虧比結果，若低於 1:3 標註「博弈性價比低」）
- 第二行：空間止損條件
- 第三行：時間止損條件
- 第四行：（空行）
- 第五行：A路徑（達標）
- 第六行：B路徑（失效）
- 第七行：C路徑（橫盤）

你之前對 {symbol} 的初步判斷是：{json.dumps(initial_judge, ensure_ascii=False)}

{expert_input}

輸出最終結論，嚴格 JSON 格式（確保 Value 為繁體中文）：{{  "action": "BUY/SELL/HOLD",  "confidence": 0-100,  "signal_breakdown": {{    "price_action": "...",    "volume_confirmation": "...",    "visual_pattern": "若無視覺分析請填寫 '無視覺數據，基於純量價分析'"  }},  "risk_factors": ["..."],  "suggestion": "必須用換行分隔的六行交易計劃"}}
"""
    resp = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role":"user","content":debate_prompt}],
        temperature=0.1,
        response_format={"type":"json_object"}
    )
    return json.loads(resp.choices[0].message.content)
# ------- 新聞情緒分析 -------
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

# ------- Telegram 消息發送 -------
def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"})
    if not resp.ok:
        print(f"⚠️ Markdown 发送失败，降级纯文本")
        safe = text.replace("*", "").replace("_", "").replace("`", "")
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": safe})

# ------- Google Sheets 記錄 -------
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

# ------- Telegram 指令處理（/analyze） -------
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

# ------- 強制分析函數 -------
def force_analyze(symbol):
    try:
        hist, vol_ratio, turnover, open_hour_ratio, daily_trend = get_recent_data(symbol)
        if hist is None:
            send_telegram(f"❌ 無法取得 {symbol} 的數據，請檢查代號是否正確。")
            return

        # 計算量化指標
        patterns = detect_patterns(hist)
        key_levels = calculate_key_levels(hist)

        # DeepSeek初判（force=True 跳過硬性過濾，但仍提供量化指標）
        initial = deepseek_judge_alert(
            symbol, hist, vol_ratio, force=True,
            turnover=turnover,
            open_hour_ratio=open_hour_ratio,
            daily_trend=daily_trend,
            patterns=patterns,
            key_levels=key_levels
        )

        # 多時間框架趨勢過濾（僅標註，不攔截）
        is_counter_trend, mtf_reason = mtf_trend_filter(initial.get('action', ''), daily_trend)
        if is_counter_trend:
            initial["reason"] = f"{initial.get('reason', '')} | ⚠️{mtf_reason}"

        # 新聞情緒分析（前置）
        news = search_news(symbol)
        sentiment = deepseek_sentiment(symbol, news)

        # 視覺分析（降低門檻至70，因為是用戶主動請求）
        gemini_vision = None
        if (gemini_client or ZHIPU_API_KEY) and initial.get("confidence", 0) >= 70:
            if not should_skip_gemini(symbol, hist):
                try:
                    img_b64 = generate_chart_b64(symbol, hist)
                    trend_desc = initial.get("trend_summary", "")
                    gemini_vision = call_vision_with_full_fallback(img_b64, symbol, trend_desc)
                    if gemini_vision:
                        LAST_GEMINI_ANALYSIS[symbol] = (time.time(), hist['Close'].iloc[-1])
                except Exception as e:
                    print(f"強制分析視覺異常: {e}")

        # 最終辯論（結合情緒與視覺）
        final = deepseek_debate(symbol, initial, gemini_vision, sentiment)

        # 盈虧比複核
        final = check_risk_reward(final, key_levels)

        # 視覺形態補充：若視覺未提供或置信度不足，顯示純量價說明或視覺錯誤原因
        final.setdefault('signal_breakdown', {})
        vis_pattern = final['signal_breakdown'].get('visual_pattern', '')
        if not vis_pattern or vis_pattern.strip() == '':
            if final.get('confidence', 0) < 80:
                if symbol in LAST_GEMINI_ERROR:
                    final['signal_breakdown']['visual_pattern'] = f"視覺模型失敗：{LAST_GEMINI_ERROR[symbol]}，因此基於純量價分析"
                else:
                    final['signal_breakdown']['visual_pattern'] = "由於置信度不足基於純量價分析"

        conf_val = final.get("confidence", 50)
        if conf_val >= 80:
            conf_tag = "🟢強信號"
        elif conf_val >= 50:
            conf_tag = "🟡弱信號（未經視覺驗證）"
        else:
            conf_tag = "🔴微弱信號"

        action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(final.get("action"), "⚪")
        base_price = hist['Close'].iloc[-1]

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



# ------- 主程序入口 -------
def main():
    # 1. 检查 Telegram 指令
    check_telegram_commands()

    # 2. 初始化 Google Sheets
    sheet = init_gsheet()

    # 3. 根据环境变量决定运行模式
    if MARKET in ("HK", "US"):
        if MARKET == "HK":
            STOCKS = STOCKS_HK
            INDEX_FILE = "current_index_hk.txt"
        else:
            STOCKS = STOCKS_US
            INDEX_FILE = "current_index_us.txt"

        if not STOCKS:
            print(f"⚠️ {MARKET} 股票清單為空，結束運行。")
            return

        # 讀取當前索引（增加容錯）
        current_index = 0
        try:
            if os.path.exists(INDEX_FILE):
                with open(INDEX_FILE, "r") as f:
                    content = f.read().strip()
                    if content:
                        current_index = int(content)
                    if current_index >= len(STOCKS) or current_index < 0:
                        current_index = 0
        except (ValueError, IOError) as e:
            print(f"⚠️ 索引檔案讀取異常，重置為0: {e}")
            current_index = 0

        symbol = STOCKS[current_index]
        print(f"🎯 本次輪詢目標: {MARKET} - {symbol} (索引 {current_index})")

        process_symbol(symbol, sheet)

        # 更新索引（增加容錯）
        next_index = (current_index + 1) % len(STOCKS)
        try:
            with open(INDEX_FILE, "w") as f:
                f.write(str(next_index))
            print(f"📝 索引已更新為 {next_index}")
        except IOError as e:
            print(f"⚠️ 索引檔案寫入失敗: {e}")

    else:
        # 兼容模式：全量扫描（手动触发时）
        print("📋 未指定市場，執行全量掃描...")
        for symbol in STOCKS_HK + STOCKS_US:
            process_symbol(symbol, sheet)

if __name__ == "__main__":
    main()
