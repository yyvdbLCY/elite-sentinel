import os, json, base64, io, requests, feedparser, time
import yfinance as yf
import mplfinance as mpf
from openai import OpenAI
from datetime import datetime, timedelta

# ==================== 配置 ====================
with open("stocks.txt", "r", encoding="utf-8") as f:
    STOCKS = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

DEEPSEEK_KEY = os.environ["DEEPSEEK_API_KEY"]
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")                   # Hugging Face 免费 Token
TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

deepseek = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com/v1")

# Gemini 客户端（最新 SDK）
gemini_client = None
if GEMINI_KEY:
    from google import genai
    from google.genai import types
    gemini_client = genai.Client(api_key=GEMINI_KEY)

# Hugging Face 视觉模型配置
HF_VISION_MODEL = "Qwen/Qwen2-VL-7B-Instruct"          # 中文优秀，完全免费
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_VISION_MODEL}"

# 状态记录
LAST_GEMINI_ANALYSIS = {}
LAST_GEMINI_CALL_TIME = 0
GEMINI_CALL_LOG = []
GEMINI_RPM_LIMIT = 8
GEMINI_PER_RUN_LIMIT = 3
GEMINI_RUN_CALLS = 0

# ==================== 工具函数 ====================
def get_recent_data(symbol):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="5d", interval="1h")
    if hist.empty:
        return None, None
    if len(hist) >= 21:
        avg_vol = hist['Volume'].iloc[-21:-1].mean()
        last_vol = hist['Volume'].iloc[-1]
        vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0
    else:
        vol_ratio = 1.0
    return hist, vol_ratio

def generate_chart_b64(symbol, hist):
    import warnings
    import matplotlib
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*findfont.*')
        warnings.filterwarnings('ignore', message='.*Font.*not found.*')
        buf = io.BytesIO()
        savefig_config = dict(fname=buf, dpi=80, format='png', bbox_inches='tight')
        mpf.plot(hist.tail(50), type='candle', style='charles',
                 volume=True, figsize=(8,4), savefig=savefig_config)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

def should_skip_gemini(symbol, hist):
    if symbol not in LAST_GEMINI_ANALYSIS:
        return False
    last_time, last_price = LAST_GEMINI_ANALYSIS[symbol]
    current_price = hist['Close'].iloc[-1]
    if time.time() - last_time < 1800:
        if abs(current_price - last_price) / last_price < 0.01:
            print(f"{symbol} 近期已分析且波动小，跳过视觉分析")
            return True
    return False

# ---------- 强化版 DeepSeek 初判（硬指标过滤 + 趋势摘要）----------
def deepseek_judge_alert(symbol, hist, vol_ratio, force=False):
    data_text = hist.tail(24)[['Open','High','Low','Close','Volume']].to_string()

    # 根据是否强制指令构建不同的规则头部
    if force:
        hard_filter_block = "(用户主动请求分析，忽略自动静默阈值，但若以下硬指标未通过，请在 risk_factors 中注明，仍给出正常分析)"
    else:
        hard_filter_block = """## 硬性过滤规则（静默阈值）
请先执行以下检查，若触发则直接返回 alert: false，除非发现明确的反转形态（头肩底、双底、楔形突破、早晨之星、镊底等）可将其覆盖：
1. 成交量倍数 < 2.0，且未出现上述底部形态 → alert: false, confidence: 0, reason: "量能不足且无底部反转形态"
2. 价格未突破过去20小时最高价，且未出现破位下跌 → alert: false, confidence: 10, reason: "价格未突破前高，无突破信号"
"""

    prompt = f"""你是顶级交易员，严格遵循下述规则进行分析。

{hard_filter_block}

## 趋势摘要
用一句话总结过去24小时的价格运行轨迹及当前位置，例如："连续缩量阴跌后回踩20周期均线不破，最后一小时出现放量反弹"。

## 分析任务（仅当硬规则未触发或已覆盖时执行）
- 判断是否出现放量突破、断崖下跌、关键反转等非正常逻辑异动
- 推算关键支撑位和压力位
- 输出置信度（0-100%），并提供3个可能误导判断的风险因素
- 当置信度 ≥ 80 时，你必须在 reason 中列举至少3个相互印证的看多/看空信号

以下是 {symbol} 近 24 小时数据：
{data_text}
当前成交量是过去 20 小时均值的 {vol_ratio:.1f} 倍。

严格输出 JSON：
{{"alert": true/false, "reason":"...", "support": 支撑价, "resistance": 压力价, "confidence": 0-100, "risk_factors": ["风险1","风险2","风险3"], "trend_summary": "趋势摘要"}}"""

    resp = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role":"user","content":prompt}],
        temperature=0.1,
        response_format={"type":"json_object"}
    )
    return json.loads(resp.choices[0].message.content)

# ---------- 强化版 Gemini 视觉分析（注入背景趋势）----------
def gemini_vision_analysis(img_b64, symbol, trend_summary="", model='gemini-2.5-flash'):
    base_prompt = "作为首席宏观分析师，严格审视以下视觉信息。"
    if trend_summary:
        base_prompt += f"\n【背景趋势】{trend_summary}"
    prompt = base_prompt + "\n请结合上述趋势背景分析这张 K 线图，重点回答：\n1. 当前形态（如头肩、双底、旗形整理等）及所处阶段。\n2. 是否存在假突破、背离或骗线信号？需与背景趋势交叉验证。\n3. 量价关系是否健康？给出简洁结论。"
    
    image_bytes = base64.b64decode(img_b64)
    resp = gemini_client.models.generate_content(
        model=model,
        contents=[
            prompt,
            types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        ]
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

# ---------- 强化版 Gemini 回退（传递趋势摘要）----------
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
                print("gemini-2.0-flash 回退成功")
                return result
            except Exception as e2:
                print(f"gemini-2.0-flash 也失败: {e2}")
                return None
        else:
            print(f"Gemini 视觉调用失败: {e}")
            return None

def hf_vision_analysis(img_b64, symbol):
    """使用 Hugging Face 免费视觉模型作为次选引擎"""
    if not HF_TOKEN:
        return None
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": f"请分析这张 {symbol} 的 K 线图，观察形态、均线、MACD，指出假突破或骗线信号，给出简洁结论。",
        "parameters": {"max_new_tokens": 200, "temperature": 0.1}
    }
    try:
        image_data = f"data:image/png;base64,{img_b64}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_data},
                    {"type": "text", "text": payload["inputs"]}
                ]
            }
        ]
        resp = requests.post(
            HF_API_URL,
            headers=headers,
            json={
                "inputs": messages,
                "parameters": payload["parameters"]
            },
            timeout=20
        )
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

# ---------- 强化版完整回退链（传递趋势摘要）----------
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

def deepseek_debate(symbol, initial_judge, gemini_vision):
    if gemini_vision:
        expert_input = f"另一位专家（视觉分析）看完 K 线图后指出：\n{gemini_vision}\n请结合视觉分析修正你的判断。"
    else:
        expert_input = "系统暂无视觉分析数据。请仅基于上述量价数据（包含支撑压力与异动原因），独立给出最终的交易决策与具体建议。"

    debate_prompt = f"""你之前对 {symbol} 的判断是：
{json.dumps(initial_judge, ensure_ascii=False)}

{expert_input}

输出最终结论，严格 JSON 格式：
{{
  "action": "BUY/SELL/HOLD",
  "confidence": 0-100,
  "signal_breakdown": {{
    "price_action": "...",
    "volume_confirmation": "...",
    "visual_pattern": "若无视觉分析请填写'无视觉数据，基于纯量价分析'"
  }},
  "risk_factors": ["..."],
  "suggestion": "给交易员的一句明确建议（必须具体，例如：缩量回踩，建议等待突破xx元再介入，或跌破xx元止损）"
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
        return "暂无相关新闻"
    titles = "\n".join([n['title'] for n in news_items])
    prompt = f"关于 {symbol} 的新闻标题：\n{titles}\n判断消息是利好出尽、真正反转或其他，一句话总结。"
    resp = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2
    )
    return resp.choices[0].message.content

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"})

# ==================== 主流程 ====================
def main():
    for symbol in STOCKS:
        try:
            hist, vol_ratio = get_recent_data(symbol)
            if hist is None:
                continue

            # 1. DeepSeek 初判（使用强化版 Prompt，force=False 默认启用硬指标过滤）
            initial = deepseek_judge_alert(symbol, hist, vol_ratio)
            if not initial.get("alert"):
                continue

            # 2. 视觉分析（注入趋势摘要）
            confidence = initial.get("confidence", 50)
            gemini_vision = None

            if (gemini_client or HF_TOKEN) and confidence >= 70:
                if not should_skip_gemini(symbol, hist):
                    try:
                        img_b64 = generate_chart_b64(symbol, hist)
                        # 从初判结果中提取趋势摘要
                        trend_desc = initial.get("trend_summary", "")
                        gemini_vision = call_vision_with_full_fallback(img_b64, symbol, trend_desc)
                        if gemini_vision:
                            LAST_GEMINI_ANALYSIS[symbol] = (time.time(), hist['Close'].iloc[-1])
                    except Exception as e:
                        print(f"视觉分析流程异常: {e}")
                else:
                    print(f"跳过 {symbol} 的视觉分析 (近期已分析且波动小)")

            # 3. 最终决策
            final = deepseek_debate(symbol, initial, gemini_vision)

            # 4. 新闻情绪
            news = search_news(symbol)
            sentiment = deepseek_sentiment(symbol, news)

            # 5. 组装结构化推送（新增动态置信度标签）
            action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(final["action"], "⚪")
            conf_val = final.get("confidence", 50)
            if conf_val >= 80:
                conf_tag = "🟢高置信度"
            elif conf_val >= 50:
                conf_tag = "🟡中置信度"
            else:
                conf_tag = "🔴低置信度"

            message = f"""
🚨 *【{symbol}】异动预警* | 置信度：{conf_val}% {conf_tag}

📊 *信号拆解*
  ▪ 价格行为：{final['signal_breakdown'].get('price_action','')}
  ▪ 量能确认：{final['signal_breakdown'].get('volume_confirmation','')}
  ▪ 图形形态：{final['signal_breakdown'].get('visual_pattern','')}

⚠️ *风险提示*
  {chr(10).join(f'  {i+1}. {r}' for i, r in enumerate(final.get('risk_factors', [])))}

📰 *新闻情绪*：{sentiment}

{action_emoji} *建议*：{final.get('suggestion','')}
"""
            send_telegram(message.strip())
            time.sleep(1)

        except Exception as e:
            print(f"处理 {symbol} 时发生错误: {e}")

if __name__ == "__main__":
    main()
