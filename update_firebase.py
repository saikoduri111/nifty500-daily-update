# @title Try new Final
import firebase_admin
from firebase_admin import credentials, firestore
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import random
import re
import uuid # For generating news item IDs if yfinance doesn't provide them

# --- Configuration ---
SERVICE_ACCOUNT_KEY_PATH = 'serviceAccountKey.json'

# !!! THIS LIST MUST MATCH NIFTY500_STOCKS_UNIQUE from your constants.ts !!!
# Using the full list of 493 symbols as previously established.
APP_STOCK_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC", "KOTAKBANK",
    "SBIN", "BHARTIARTL", "BAJFINANCE", "LT", "ASIANPAINT", "HCLTECH", "AXISBANK",
    "MARUTI", "WIPRO", "ULTRACEMCO", "SUNPHARMA", "BAJAJFINSV", "ADANIENT", "TITAN",
    "ONGC", "NTPC", "TATAMOTORS", "JSWSTEEL", "POWERGRID", "ADANIPORTS", "COALINDIA",
    "SBILIFE", "GRASIM", "HINDALCO", "EICHERMOT", "CIPLA", "DRREDDY", "TECHM",
    "INDUSINDBK", "BRITANNIA", "NESTLEIND", "BPCL", "SHREECEM", "DIVISLAB",
    "TATACONSUM", "UPL", "HEROMOTOCO", "BAJAJ-AUTO", "APOLLOHOSP", "M&M", "IOC", "VEDL",
    "ADANIGREEN", "TATAPOWER", "SIEMENS", "DLF", "BAJAJHLDNG", "PIDILITIND", "AMBUJACEM",
    "GAIL", "HDFCLIFE", "ICICIPRULI", "LTIM", "TATASTEEL", "INDIGO", "BANKBARODA",
    "HAVELLS", "PNB", "DABUR", "ICICIGI", "MARICO", "ABB", "BOSCHLTD", "LUPIN",
    "COLPAL", "PGHH", "BEL", "CANBK", "UNIONBANK", "NMDC", "AUROPHARMA", "HINDZINC",
    "TORNTPOWER", "PETRONET", "GODREJCP", "CONCOR", "BERGEPAINT", "IDFCFIRSTB",
    "GLAND", "MPHASIS", "IRCTC", "MUTHOOTFIN", "SAIL", "ASHOKLEY", "BANDHANBNK",
    "JSWENERGY", "MRF", "PAGEIND", "ACC", "YESBANK", "BIOCON", "INDUSTOWER",
    "HDFCAMC", "ZYDUSLIFE", "JUBLFOOD", "ASTRAL", "SRF", "ALKEM", "NAUKRI",
    "MCDOWELL-N", "UBL", "VOLTAS", "CUMMINSIND", "ADANITOTAL", "MOTHERSON", "TVSMOTOR",
    "PERSISTENT", "PIIND", "LICHSGFIN", "INDIAMART", "BHEL", "DALBHARAT", "HAL",
    "POLYCAB", "BALKRISIND", "IPCALAB", "LTTS", "SYNGENE", "MAXHEALTH", "OFSS",
    "COROMANDEL", "DIXON", "FACT", "FORTIS", "GUJGASLTD", "HONAUT", "IDBI", "IGL",
    "LODHA", "MANAPPURAM", "MFSL", "NHPC", "OBEROIRLTY", "PATANJALI", "PEL", "PFC",
    "POLICYBZR", "RECLTD", "SBICARD", "SHRIRAMFIN", "SOLARINDS", "TRENT", "CGPOWER",
    "TATATECH", "JINDALSTEL", "THERMAX", "VBL", "ZEEL", "ZOMATO", "AARTIIND",
    "APLAPOLLO", "AUBANK", "BAJAJELEC", "BATAINDIA", "BBTC", "BDL", "BHARATFORG",
    "CASTROLIND", "CENTURYTEX", "CHAMBLFERT", "CHOLAFIN", "COFORGE", "CROMPTON",
    "DEEPAKNTR", "DELHIVERY", "DEVYANI", "EIHOTEL", "EMAMILTD", "ENDURANCE",
    "ESCORTS", "EXIDEIND", "FEDERALBNK", "FINCABLES", "FINPIPE", "FSL", "GET&D",
    "GLAXO", "GLENMARK", "GODREJIND", "GODREJPROP", "GRANULES", "GSFC", "GSPL",
    "HEG", "IDFC", "IIFLWAM", "INDHOTEL", "INDIANB", "IOCL", "IPRUAMC", "IRCON",
    "ISEC", "J&KBANK", "JAICORPLTD", "JBCHEPHARM", "JINDALSAW", "JKCEMENT",
    "JKLAKSHMI", "KAJARIACER", "KALPATPOWR", "KEC", "KNRCON", "KRBL", "KSB",
    "LAURUSLABS", "LXCHEM", "M&MFIN", "MAHABANK", "MAHSEAMLES", "MANKIND",
    "MEDANTA", "METROPOLIS", "MGL", "MOTILALOFS", "NAM-INDIA", "NATIONALUM",
    "NAVINFLUOR", "NBCC", "NCC", "NLCINDIA", "NYKAA", "OIL", "PAYTM", "PCBL",
    "PHOENIXLTD", "PNBHOUSING", "PRESTIGE", "PRSMJOHNSN", "RADICO", "RAILTEL",
    "RAJESHEXPO", "RALLIS", "RAMCOCEM", "RAYMOND", "RCF", "REDINGTON", "RVNL",
    "SCHAEFFLER", "SHOPERSTOP", "SOBHA", "SONACOMS", "STARHEALTH", "STLTECH",
    "SUMICHEM", "SUNDARMFIN", "SUNDRMFAST", "SUNTECK", "SUNTV", "SUPRAJIT",
    "SUVENPHAR", "SWANENERGY", "SYMPHONY", "TANLA", "TATACHEM", "TATACOMM",
    "TATAELXSI", "TATAINVEST", "TEAMLEASE", "TIMKEN", "TITAGARH", "TORNT PHARM",
    "TRIDENT", "TRIVENI", "TTML", "UFLEX", "UNIONASSET", "UTIAMC", "VAIBHAVGBL",
    "VGUARD", "VIPIND", "VMART", "WELCORP", "WELSPUNIND", "WESTLIFE", "WHIRLPOOL",
    "WOCKPHARMA", "ZENSARTECH", "ABFRL", "ADANIPOWER", "AEGISCHEM", "AIAENG",
    "APLLTD", "APTUS", "ASAHIINDIA", "AVANTIFEED", "BAJAJCON", "BALAMINES",
    "BALRAMCHIN", "BANKINDIA", "BASF", "BEML", "BLUEDART", "BRIGADE", "CARBORUNIV",
    "CCL", "CDSL", "CEATLTD", "CENTRALBK", "CESC", "CGCL", "CHENNPETRO", "CIGNITITEC",
    "COCHINSHIP", "CREDITACC", "CSBBANK", "DATAPATTNS", "DCBBANK", "DELTACORP", "DHANI",
    "ECLERX", "EDELWEISS", "EQUITASBNK", "ERIS", "FDC", "FINOLEXIND", "FLUOROCHEM",
    "GABRIEL", "GEPIL", "GESHIP", "GICRE", "GMDCLTD", "GMRINFRA", "GNFC", "GOCOLORS",
    "GPPL", "GRAPHITE", "GRINDWELL", "GUJALKALI", "HFCL", "HIKAL", "IBREALEST",
    "IBULHSGFIN", "ICIL", "IEX", "IFBIND", "IGPL", "IIFL", "INEOSSTYRO",
    "INTELLECT", "IOB", "ITI", "JAMNAAUTO", "JBMA", "JCHAC", "JKPAPER", "JSL",
    "JUSTDIAL", "JYOTHYLAB", "KARURVYSYA", "KIMS", "KIRLOSENG",
    "KPITTECH", "L&TFH", "LATENTVIEW", "LEMONTREE", "LUXIND", "MAHINDCIE",
    "MAHLOG", "MAPMYINDIA", "MASFIN", "MASTEK", "MAZDOCK", "MEDPLUS", "MMTC",
    "MOIL", "MRPL", "MSUMI", "NATCOPHARM", "NAZARA", "NESCO", "NFL", "NIACL",
    "POLYMED", "PRAJIND", "PRICOL", "PSPPROJECT", "PTC", "QUESS",
    "RAIN", "RBLBANK", "RELAXO", "ROUTE", "RTNINDIA", "SANOFI", "SAPPHIRE",
    "SARDAEN", "SCI", "SEQUENT", "SFL", "SHILPAMED", "SJVN", "SKFINDIA",
    "SONATSOFTW", "SOUTHBANK", "SPICEJET", "STAR", "SUZLON", "TASTYBITE",
    "TEJASNET", "TRITURBINE", "TTKPRESTIG", "UCOBANK", "UJJIVANSFB",
    "VAKRANGEE", "VARROC", "VIJAYA", "WELENT", "ZEEMEDIA", "ZFCVINDIA", "3MINDIA",
    "AAVAS", "AFFLE", "AJANTPHARM", "AKZOINDIA", "ALKYLAMINE", "AMARAJABAT",
    "ANGELONE", "ANURAS", "ASTRAZEN", "ATUL", "BLS", "CHALET", "CLEAN",
    "CMSINFO", "DBREALTY", "DCMSHRIRAM", "EASEMYTRIP", "ENGINERSIN", "FINEORG",
    "GODREJAGRO", "GRINFRA", "HAPPSTMNDS", "HUDCO", "INDIGOPNTS", "INOXWIND",
    "JAGRAN", "JUBLINGREA", "KALYANKJIL", "LAXMIMACH", "MANYAVAR", "MCX",
    "MIDHANI", "NETWORK18", "NH", "PVRINOX", "RITES", "SHYAMMETL", "SPANDANA",
    "SUPREMEIND", "TTKHLTCARE", "IRCONISL", "IRCONPB", "IRCONPJ", "JIOFIN",
    "MAZDOCK IN", "SUVEN", "UJJIVAN", "ADSL", "ACE", "ADFFOODS", "ADROITINFO",
    "ADVANIHOTR", "ADVENZYMES", "AGARIND", "AGRITECH", "AHLUCONT", "AHLWEST",
    "ALEMBICLTD", "ALICON", "ALKALI", "ALLCARGO", "ALLSEC", "ALMONDZ", "ALOKINDS",
    "AMBICAAGAR", "AMBER", "AMJLAND", "AMRUTANJAN", "ANANTRAJ", "ANDHRAPAP",
    "ANDHRSUGAR", "APARINDS", "APCOTEXIND", "APEX", "APOLSINHOT", "APTECHT",
    "ARCHIDPLY", "ARIESAGRO", "ARIHANT", "ARMANFIN", "ARROWGREEN", "ARSHIYA",
    "ARTEMISMED", "ARVIND", "ARVINDFASN", "ARVSMART", "ASAL", "ASALCBR", "ASHAPURMIN",
    "ASHIANA", "ASHIMASYN", "ASIANTILES", "ASPINWALL", "ASTRAMICRO", "ATFL",
    "AUTOAXLES", "AUTOLITIND", "AVTNPL", "AXISCADES", "AXITA", "AYMSYNTEX", "BAGFILMS",
    "BAIDFIN", "BALAJITELE", "BALKRISHNA", "BALPHARMA", "BANARISUG", "BANCOINDIA",
    "BANG", "BANKA", "BARBEQUE", "BEARDSELL", "BEDMUTHA", "BEPL", "BHAGERIA",
    "BHAGYANGR", "BHANDARI", "BHARATGEAR", "BHARATRAS", "BHARTIYA", "BIGBLOCK", "BIL",
    "BINANIIND", "BINDALAGRO", "BIOGEN", "BIRLACABLE", "BIRLACORPN", "BIRLAMONEY",
    "BLISSGVS", "BODALCHEM", "BPL", "BROOKS", "BSE", "BSL", "BSOFT", "BURNPUR",
    "BUTTERFLY", "BVCL", "BYKE", "CAMLINFINE", "CAMS", "CANDC", "CAPACITE",
    "CAPLIPOINT", "CAREERP", "CARERATING", "CENTENKA", "CENTEXT", "CENTRUM", "CENTUM",
    "CERA", "CHEMCON", "CHEMFAB", "CHEMPLASTS", "CHORDCHEM"
]
YFINANCE_SYMBOLS = [symbol + ".NS" for symbol in APP_STOCK_SYMBOLS]

FIRESTORE_PRICES_COLLECTION_NAME = 'nifty_prices'
FIRESTORE_ANALYSIS_COLLECTION_NAME = 'analyzed_stocks'
# --- End Configuration ---

db = None

def initialize_firebase():
    global db
    try:
        cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("Firebase Admin SDK initialized successfully.")
        return True
    except Exception as e:
        print(f"Error initializing Firebase Admin SDK: {e}")
        return False

def get_stock_data(symbol_ns):
    ticker = yf.Ticker(symbol_ns)
    info, history, raw_news_data = {}, None, []
    try: info = ticker.info
    except Exception: print(f"  [yf] Could not get .info for {symbol_ns}")
    try: history = ticker.history(period="3mo", interval="1d")
    except Exception: print(f"  [yf] Could not get .history for {symbol_ns}")
    try: raw_news_data = ticker.news
    except Exception: print(f"  [yf] Could not get .news for {symbol_ns}")
    return info, history, raw_news_data

def calculate_sma(series, window):
    if not isinstance(series, pd.Series) or len(series) < window: return None
    try: return series.rolling(window=window, min_periods=1).mean().iloc[-1]
    except Exception: return None

def calculate_rsi(series, window=14):
    if not isinstance(series, pd.Series) or len(series) < window + 1: return None
    try:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0); loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=window, min_periods=1).mean(); avg_loss = loss.rolling(window=window, min_periods=1).mean()
        last_avg_gain = avg_gain.iloc[-1]; last_avg_loss = avg_loss.iloc[-1]
        if last_avg_loss == 0: return 100.0 if last_avg_gain > 0 else 50.0
        rs = last_avg_gain / last_avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
    except Exception: return None

POSITIVE_KEYWORDS = ['positive', 'strong', 'growth', 'profit', 'up', 'bullish', 'good', 'beat', 'buy', 'upgrade', 'optimistic', 'record', 'dividend', 'launch', 'new product', 'expansion', 'partnership', 'successful', 'agreement', 'achieve', 'accelerate', 'innovation', 'outperform', 'surge', 'rally', 'improvement', 'advancement']
NEGATIVE_KEYWORDS = ['negative', 'weak', 'loss', 'decline', 'down', 'bearish', 'bad', 'missed', 'sell', 'downgrade', 'pessimistic', 'scandal', 'investigation', 'poor', 'delay', 'cut', 'halt', 'warning', 'concerns', 'issue', 'drop', 'slump', 'reduce', 'risk', 'volatile', 'pressure', 'disappoint', 'struggle', 'challenge']
COMMON_COMPANY_SUFFIXES = r'\b(ltd|limited|pvt|private|corp|corporation|inc|incorporated|industries|solutions|services|group|ventures|consulting|global|systems|technologies|pharmaceuticals|financial|energy|power|motors|bank|capital|communications|auto|steel|cement|consumer|products|logistics|chemicals|textiles|realty|housing|infra|construction|media|entertainment|digital|online|labs|life|health|care|foods|beverages|agro|paper|shipping|ports|aviation|travel|hotels|retail|fashion|footwear|apparel|jewellers|diagnostics|consultancy|management|research|institute|holding|investment|insurance|finance|credit)\b\.?'

def format_and_score_news(raw_news_list, stock_symbol_original, company_long_name):
    """
    Processes raw news, scores sentiment, and formats relevant news for Firestore.
    Returns: (sentiment_score, sentiment_reasoning_text, list_of_structured_news_items_for_ui)
    """
    if not raw_news_list:
        return 0, "No recent news found by provider for sentiment analysis.", []

    sentiment_score_accumulator = 0
    processed_news_items_for_firestore = []
    headlines_used_for_sentiment_reasoning = []
    news_considered_for_sentiment_count = 0
    max_news_items_to_store_for_ui = 3 # Max news items to format and store for the UI

    name_variants = set()
    if company_long_name and isinstance(company_long_name, str):
        temp_name = company_long_name.lower()
        if temp_name not in ["none", "n/a", "", "company"]: name_variants.add(temp_name)
        cleaned_name_for_match = re.sub(COMMON_COMPANY_SUFFIXES, '', temp_name, flags=re.IGNORECASE)
        cleaned_name_for_match = re.sub(r'[^\w\s-]', '', cleaned_name_for_match).strip()
        if cleaned_name_for_match and cleaned_name_for_match not in name_variants: name_variants.add(cleaned_name_for_match)
        significant_words = [word for word in cleaned_name_for_match.split() if len(word) > 2 and word not in ['the', 'and', 'of']]
        if significant_words: name_variants.update(significant_words)
        if len(significant_words) > 1: name_variants.add(" ".join(significant_words[:2]))
    cleaned_symbol = re.sub(r'[^a-zA-Z0-9]', '', stock_symbol_original.lower())
    if cleaned_symbol: name_variants.add(cleaned_symbol)
    if '-' in stock_symbol_original: name_variants.update(part.lower() for part in stock_symbol_original.split('-') if len(part)>2)
    if len(name_variants) > 1: name_variants = {v for v in name_variants if (len(v) > 2 and v not in ['ltd', 'limited', 'inc']) or v == cleaned_symbol }

    relevant_raw_news_for_processing = []
    for item_data_top_level in raw_news_list[:5]: # Consider top 5 news for relevance
        item_content = item_data_top_level.get('content')
        if not item_content or not isinstance(item_content, dict): continue
        title = item_content.get('title', '').lower()
        title_words = set(re.findall(r'\b\w+\b', title))
        is_relevant = any(variant in title for variant in name_variants if variant) or \
                      any(variant_word in title_words for variant_word in name_variants if variant_word and len(variant_word)>2)
        if "j&kbank" == stock_symbol_original.lower() and ("jammu & kashmir bank" in title or "j&k bank" in title): is_relevant = True
        if is_relevant:
            relevant_raw_news_for_processing.append(item_data_top_level)

    for item_data_top_level in relevant_raw_news_for_processing: # Process relevant news
        item_content = item_data_top_level.get('content')
        news_considered_for_sentiment_count +=1
        title = item_content.get('title', '').lower()

        if item_content.get('title'): headlines_used_for_sentiment_reasoning.append(item_content.get('title', ''))

        headline_sentiment = sum(1 for p_word in POSITIVE_KEYWORDS if p_word in title) - \
                             sum(1 for n_word in NEGATIVE_KEYWORDS if n_word in title)
        sentiment_score_accumulator += headline_sentiment

        # Always try to format news items if we have content, up to the UI limit
        if len(processed_news_items_for_firestore) < max_news_items_to_store_for_ui:
            news_id = item_data_top_level.get('id', str(uuid.uuid4())) # Corrected: use uuid.uuid4()
            provider_info = item_content.get('provider', {})
            clickthrough_info = item_content.get('clickThroughUrl', {})
            pub_date = item_content.get('pubDate', datetime.utcnow().isoformat() + "Z")
            if not isinstance(pub_date, str):
                try: pub_date = pd.Timestamp(pub_date).isoformat() + "Z"
                except: pub_date = datetime.utcnow().isoformat() + "Z"

            processed_news_items_for_firestore.append({
                "id": news_id,
                "content": {
                    "title": item_content.get('title', 'N/A'),
                    "pubDate": pub_date,
                    "provider": {"displayName": provider_info.get('displayName', 'N/A') if isinstance(provider_info, dict) else 'N/A'},
                    "clickThroughUrl": {"url": clickthrough_info.get('url', '#') if isinstance(clickthrough_info, dict) else '#'}
                }
            })

    final_sentiment_score = 0
    if news_considered_for_sentiment_count > 0:
        final_sentiment_score = min(max(int((sentiment_score_accumulator / news_considered_for_sentiment_count) * 2.0), -10), 10)

    if news_considered_for_sentiment_count == 0:
        reasoning_news_sentiment = "No news directly matching company/symbol found for sentiment."
    elif final_sentiment_score > 2:
        reasoning_news_sentiment = f"Sentiment from {news_considered_for_sentiment_count} relevant news item(s) leans positive."
    elif final_sentiment_score < -1:
        reasoning_news_sentiment = f"Sentiment from {news_considered_for_sentiment_count} relevant news item(s) suggests caution."
    else:
        reasoning_news_sentiment = f"Sentiment from {news_considered_for_sentiment_count} relevant news item(s) is neutral/mixed."

    if headlines_used_for_sentiment_reasoning:
        reasoning_news_sentiment += " Key headlines influencing sentiment: \"" + "\"; \"".join(headlines_used_for_sentiment_reasoning[:2]) + "\"." # Show max 2 in reasoning

    return final_sentiment_score, reasoning_news_sentiment, processed_news_items_for_firestore


def detailed_stock_analysis(symbol_ns, stock_name_fallback):
    info, history, raw_news_list_from_yf = get_stock_data(symbol_ns)
    analysis_score_no_news, reasons_no_news_list = 0, []
    original_symbol = symbol_ns.replace(".NS", "")
    display_name = info.get('longName') or info.get('shortName') or stock_name_fallback
    current_price = info.get('regularMarketPrice') or info.get('previousClose')
    if current_price is None and history is not None and not history.empty and 'Close' in history.columns:
        current_price = history['Close'].iloc[-1]
    if current_price is None: return None

    if history is not None and not history.empty and 'Close' in history.columns:
        close_prices = history['Close'].dropna()
        if not close_prices.empty:
            sma_20=calculate_sma(close_prices,20); sma_50=calculate_sma(close_prices,50); rsi=calculate_rsi(close_prices)
            if sma_20 is not None and sma_50 is not None:
                if current_price>sma_50: analysis_score_no_news+=12; reasons_no_news_list.append(f"Price ({current_price:.2f}) > 50D SMA ({sma_50:.2f}).")
                if current_price>sma_20: analysis_score_no_news+=8; reasons_no_news_list.append(f"Price > 20D SMA ({sma_20:.2f}).")
                elif current_price<sma_50: analysis_score_no_news-=(5 if sma_20 and current_price>sma_20 else 10); reasons_no_news_list.append(f"Price < 50D SMA ({sma_50:.2f}).")
            if rsi is not None:
                if rsi<35: analysis_score_no_news+=10; reasons_no_news_list.append(f"RSI {rsi:.2f} (oversold).")
                elif rsi>65: analysis_score_no_news-=7; reasons_no_news_list.append(f"RSI {rsi:.2f} (overbought).")
                else: analysis_score_no_news+=2; reasons_no_news_list.append(f"RSI {rsi:.2f} (neutral).")
    else: reasons_no_news_list.append("Limited history for tech analysis.")
    pe_ratio=info.get('trailingPE'); beta=info.get('beta'); dividend_yield=info.get('dividendYield')
    if pe_ratio is not None:
        if 0<pe_ratio<18: analysis_score_no_news+=7; reasons_no_news_list.append(f"P/E {pe_ratio:.2f} (low).")
        elif pe_ratio>35: analysis_score_no_news-=(3 if pe_ratio<50 else 6); reasons_no_news_list.append(f"P/E {pe_ratio:.2f} (high).")
    if beta is not None:
        if beta>1.2: reasons_no_news_list.append(f"Beta {beta:.2f} (> market vol).")
        elif 0<beta<0.8: reasons_no_news_list.append(f"Beta {beta:.2f} (< market vol).")
    if dividend_yield is not None and dividend_yield>0: analysis_score_no_news+=min(dividend_yield*100,5); reasons_no_news_list.append(f"Yield {dividend_yield*100:.2f}%.")
    growth_chance_no_news=int(max(5,min(95,50+(analysis_score_no_news*0.7))))
    reasoning_no_news_text=generate_reasoning_text(reasons_no_news_list,"core (technical/fundamental)")
    reasoning_no_news_text+=f" Core growth prospect: {growth_chance_no_news}%."

    news_sentiment_score, reasoning_news_sentiment_text, structured_news_items = \
        format_and_score_news(raw_news_list_from_yf, original_symbol, display_name)

    analysis_score_with_news = analysis_score_no_news + news_sentiment_score
    growth_chance_with_news = int(max(5, min(95, 50 + (analysis_score_with_news * 0.65))))
    combined_reasons = list(reasons_no_news_list); combined_reasons.append(reasoning_news_sentiment_text)
    reasoning_with_news_text = generate_reasoning_text(combined_reasons,"overall (incl. news)")
    reasoning_with_news_text += f" Overall growth prospect (incl. news): {growth_chance_with_news}%."

    return {"symbol":original_symbol, "name":display_name,
            "growthChanceNoNews":growth_chance_no_news, "reasoningNoNews":reasoning_no_news_text,
            "growthChanceWithNews":growth_chance_with_news, "reasoningWithNews":reasoning_with_news_text,
            "newsItems":structured_news_items, "id":original_symbol}

def generate_reasoning_text(reasons_list, score_type_name):
    reasoning_intro = f"Reasoning for {score_type_name} analysis: "
    if not reasons_list: return reasoning_intro + "Limited specific data points."
    selected = random.sample(reasons_list, k=min(len(reasons_list), 2)) if len(reasons_list) > 1 else reasons_list
    return reasoning_intro + " ".join(selected)

def update_all_data():
    if not initialize_firebase(): return
    print("Fetching latest EOD prices and performing analysis...")
    price_data_to_store, analysis_results = {}, []
    stock_name_map = {stock['symbol']: stock['name'] for stock in NIFTY500_STOCKS_UNIQUE_PY}
    total_symbols = len(YFINANCE_SYMBOLS)

    for i, symbol_ns in enumerate(YFINANCE_SYMBOLS):
        original_symbol = symbol_ns.replace(".NS", "")
        fallback_name = stock_name_map.get(original_symbol, original_symbol)
        print(f"\nProcessing {original_symbol} ({i + 1}/{total_symbols})...")
        try:
            ticker = yf.Ticker(symbol_ns); info = ticker.info
            current_price = info.get('regularMarketPrice') or info.get('previousClose')
            if current_price is not None: price_data_to_store[original_symbol] = round(float(current_price),2); print(f"  Price: {price_data_to_store[original_symbol]}")
            else:
                hist = ticker.history(period="1d")
                if not hist.empty and 'Close' in hist.columns: price_data_to_store[original_symbol] = round(float(hist['Close'].iloc[-1]),2); print(f"  Price (hist): {price_data_to_store[original_symbol]}")
                else: print(f"  Could not fetch price for {original_symbol}")
        except Exception as e: print(f"  Error fetching price for {original_symbol}: {e}")
        try:
            analysis = detailed_stock_analysis(symbol_ns, fallback_name)
            if analysis: analysis_results.append(analysis); print(f"  Analyzed: {analysis['name']} | Core: {analysis['growthChanceNoNews']}% | With News: {analysis['growthChanceWithNews']}% | News Items: {len(analysis.get('newsItems',[]))}")
            else: print(f"  Could not analyze {original_symbol}.")
        except Exception as e: print(f"  Error during analysis of {original_symbol}: {e}")
        if (i + 1) % 15 == 0: print(f"Pausing for 5 seconds after processing {i + 1} symbols..."); time.sleep(5)

    if price_data_to_store:
        today_str = datetime.now().strftime('%Y-%m-%d')
        prices_doc_ref = db.collection(FIRESTORE_PRICES_COLLECTION_NAME).document(today_str)
        try: prices_doc_ref.set(price_data_to_store, merge=True); print(f"\nStored EOD prices for {len(price_data_to_store)} stocks for {today_str}.")
        except Exception as e: print(f"Error storing EOD prices: {e}")
    else: print("\nNo EOD prices fetched.")
    if analysis_results:
        today_str = datetime.now().strftime('%Y-%m-%d')

        sorted_analysis = sorted(analysis_results, key=lambda x: x['growthChanceWithNews'], reverse=True)
        analysis_doc_ref = db.collection(FIRESTORE_ANALYSIS_COLLECTION_NAME).document(today_str)
        try: analysis_doc_ref.set({"stocks": sorted_analysis}, merge=True); print(f"\nStored analysis for {len(sorted_analysis)} stocks for {today_str}.")
        except Exception as e: print(f"Error storing analysis: {e}")
    else: print("\nNo analysis results.")

NIFTY500_STOCKS_UNIQUE_PY = []
if __name__ == "__main__":
    NIFTY500_STOCKS_UNIQUE_PY = [{"symbol": s, "name": s} for s in APP_STOCK_SYMBOLS]
    update_all_data()