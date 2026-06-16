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
HF_TOKEN = os.environ.get("HF_TOKEN")
TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

deepseek = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com/v1")

# Gemini 客户端
gemini_client = None
if GEMINI_KEY:
    from google import genai
    from google.genai import types
    gemini_client = genai.Client(api_key=GEMINI_KEY)

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
    """拉取数据，额外计算换手率和开盘第一小时量比"""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="5d", interval="1h")
    if hist.empty:
        return None, None, None, None

    # 成交量倍数
    if len(hist) >= 21:
        avg_vol = hist['Volume'].iloc[-21:-1].mean()
        last_vol = hist['Volume'].iloc[-1]
        vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0
    else:
        vol_ratio = 1.0

    # 换手率
    turnover = None
    try:
        shares = ticker.info.get('sharesOutstanding')
        if shares and shares > 0:
            turnover = (hist['Volume'].iloc[-1] / shares) * 100
    except:
        pass

    # 开盘第一小时量比（基于过去5个交易日同时段均值）
    open_hour_ratio = None
    try:
        # 取最近一天的数据
        today = hist[hist.index.date == hist.index[-1].date()]
        if len(today) > 0:
            first_hour_vol = today['Volume'].iloc[0]
            past_5 = hist[hist.index.date < hist.index[-1].date()]
            if len(past_5) >= 5:
                # 按日期分组取第一天第一小时
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

# ---------- DeepSeek 初判（强化量能 + 硬指标 + 指令约束）----------
def deepseek_judge_alert(symbol, hist, vol_ratio, force=False, turnover=None, open_hour_ratio=None):
    data_text = hist.tail(24)[['Open','High','Low','Close','Volume']].to_string()

    # 构建额外数据文本
    extra_info = ""
    if turnover is not None:
        extra_info += f"\n当前换手率：{turnover:.2f}%"
    if open_hour_ratio is not None:
        extra_info += f"\n开盘第一小时量比（相对过去5日同时段均值）：{open_hour_ratio:.2f}"

    if force:
        hard_filter_block = "(用户主动请求分析，忽略自动静默阈值，但若以下硬指标未通过，请在 risk_factors 中注明，仍给出正常分析)"
    else:
        hard_filter_block = f"""## 硬性过滤规则（静默阈值）
请先执行以下检查，若触发则直接返回 alert: false，除非发现明确的反转形态（头肩底、双底、楔形突破、早晨之星、镊底等）可将其覆盖：
1. 成交量倍数 < 2.0，且未出现上述底部形态 → alert: false, confidence: 0, reason: "量能不足且无底部反转形态"
2. 价格未突破过去20小时最高价，且未出现破位下跌 → alert: false, confidence: 10, reason: "价格未突破前高，无突破信号"
3. 若换手率 > 5% 且价格涨幅 < 1%（滞涨），必须在 reason 中注明“派发风险”，但仍可继续分析（alert 可为 true）"""

    prompt = f"""你是顶级交易员，严格遵循下述规则进行分析。

## 认知诚实原则（指令约束）
- 你必须仅基于下方给出的数据回答。如果某个判断缺乏数据依据，必须在对应的字段中注明“无数据支持”，并将该结论的置信度设置为 0。
- 不允许编造趋势、量价关系或形态，不允许假设数据之外的信息。若强行输出无来源的信息，本次输出将被视为无效。

## 事实锚定要求
在给出支撑位、压力位、突破判断、量价结论时，必须附带信息来源标记，例如：
- "[来自近24小时K线数据]"
- "[来自成交量对比]"
- "[来自趋势摘要]"

{hard_filter_block}

## 趋势摘要
用一句话总结过去24小时的价格运行轨迹及当前位置。

## 分析任务（仅当硬规则未触发或已覆盖时执行）
- 判断是否出现放量突破、断崖下跌、关键反转等非正常逻辑异动
- 推算关键支撑位和压力位（必须标明数据来源）
- 输出置信度（0-100%），并提供3个可能误导判断的风险因素
- 当置信度 ≥ 80 时，你必须在 reason 中列举至少3个相互印证的看多/看空信号

以下是 {symbol} 近 24 小时数据：
{data_text}
当前成交量是过去 20 小时均值的 {vol_ratio:.1f} 倍。{extra_info}

严格输出 JSON：
{{"alert": true/false, "reason":"...", "support": "支撑价 [来源]", "resistance": "压力价 [来源]", "confidence": 0-100, "risk_factors": ["风险1","风险2","风险3"], "trend_summary": "趋势摘要"}}"""

    resp = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role":"user","content":prompt}],
        temperature=0.1,
        response_format={"type":"json_object"}
    )
    return json.loads(resp.choices[0].message.content)

# ---------- Gemini 视觉分析（不变）----------
def gemini_vision_analysis(img_b64, symbol, trend_summary="", model='gemini-2.5-flash'):
    base_prompt = "作为首席宏观分析师，严格审视以下视觉信息。"
    if trend_summary:
        base_prompt += f"\n【背景趋势】{trend_summary}"
    prompt = base_prompt + "\n请结合上述趋势背景分析这张 K 线图，重点回答：\n1. 当前形态及所处阶段。\n2. 是否存在假突破、背离或骗线信号？需与背景趋势交叉验证。\n3. 量价关系是否健康？给出简洁结论。"
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
        "inputs": f"请分析这张 {symbol} 的 K 线图，观察形态、均线、MACD，指出假突破或骗线信号，给出简洁结论。",
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

# ---------- 最终辩论（新增交易计划要求）----------
def deepseek_debate(symbol, initial_judge, gemini_vision):
    if gemini_vision:
        expert_input = f"另一位专家（视觉分析）看完 K 线图后指出：\n{gemini_vision}\n请结合视觉分析修正你的判断。"
    else:
        expert_input = "系统暂无视觉分析数据。请仅基于上述量价数据，独立给出最终的交易决策与具体建议。"

    debate_prompt = f"""你是顶级交易员，正在对一份初始分析进行最终裁决。

## 认知诚实原则（指令约束）
- 你必须仅基于量价数据、视觉分析结论以及风险因素进行判断。
- 如果某个操作建议缺乏直接的数据或图形支撑，必须在 suggestion 中注明“该建议基于综合经验，缺乏直接量化指标”。
- 不得编造未在上下文中出现的支撑/压力位或趋势。

## 事实锚定要求
最终输出的 suggestion 必须指明其逻辑来源，例如：
- "[基于纯量价分析]"
- "[基于视觉分析对假突破的确认]"
- "[基于新闻情绪与量价共振]"

## 高级交易计划要求
你必须输出一个完整的交易计划，包含以下要素：
1. **盈亏比矩阵**：根据建议的入场、止损和目标位，自动计算风险回报比（盈亏比）。若盈亏比低于 1:3，必须在 suggestion 中额外标注“博弈性价比低”。
2. **双重止损机制**：止损必须包含空间止损（价格跌破 X 元）和时间止损（若在 Y 元上方横盘超过 Z 个交易日无法拉回，则失效）。若无法判断时间，可假设“横盘 2 个交易日失效”。
3. **逻辑树预判（A/B/C 路径）**：
   - 路径 A（达标）：若价格站稳 X 元，应如何调仓或加仓。
   - 路径 B（失效）：若价格跌破 Y 元或时间止损触发，该信号作废。
   - 路径 C（横盘）：若在 Z 区间震荡，建议最多持有几天或需等待的突破方向。

你之前对 {symbol} 的初步判断是：
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
  "suggestion": "包含盈亏比计算、双重止损条件、A/B/C 三种情景的完整交易计划（必须包含来源标记）"
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
            hist, vol_ratio, turnover, open_hour_ratio = get_recent_data(symbol)
            if hist is None:
                continue

            # 初判（传入量能细化数据）
            initial = deepseek_judge_alert(symbol, hist, vol_ratio, turnover=turnover, open_hour_ratio=open_hour_ratio)
            if not initial.get("alert"):
                continue

            confidence = initial.get("confidence", 50)
            gemini_vision = None

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

            # 置信度标签
            conf_val = final.get("confidence", 50)
            if conf_val >= 80:
                conf_tag = "🟢高置信度"
            elif conf_val >= 50:
                conf_tag = "🟡中置信度"
            else:
                conf_tag = "🔴低置信度"

            # 构建消息（包含追踪密钥）
            action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(final["action"], "⚪")
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

🔑 *追踪密钥*：`{symbol} | 观察中 | 基准价 {hist['Close'].iloc[-1]:.2f}`
"""
            send_telegram(message.strip())# 写入 Google Sheets（需要 credentials.json 或 GDRIVE_CREDENTIALS 环境变量）
try:
    append_alert(symbol, conf_val, final.get('suggestion',''), hist['Close'].iloc[-1])
except Exception as e:
    print(f"写入 Sheets 失败: {e}")
            time.sleep(1)

        except Exception as e:
            print(f"处理 {symbol} 时发生错误: {e}")

if __name__ == "__main__":
    main()
