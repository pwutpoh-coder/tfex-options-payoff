import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import os
from datetime import datetime, date, timezone, timedelta
from scipy.stats import norm

# ตั้งค่าหน้าเว็บ: ใช้ชื่อ Futures & Options Payoff และไอคอน 📈
st.set_page_config(
    page_title="Futures & Options Payoff", 
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto"
)

# กำหนดโซนเวลา Bangkok (UTC+7)
BANGKOK_TZ = timezone(timedelta(hours=7))

# --- CSS พิเศษช่วยปรับแต่งการแสดงผลบนมือถือ, สัญลักษณ์สี, และตารางแบบเลื่อนบาร์แนวนอนบรรทัดเดียว ---
st.markdown("""
    <style>
    .stMetric {
        background-color: #f8f9fa;
        padding: 8px;
        border-radius: 6px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .badge-long-fu { background-color: #e2f0d9; color: #385723; padding: 2px 6px; border-radius: 4px; font-weight: 500; font-size: 11px; }
    .badge-short-fu { background-color: #fce4d6; color: #c65911; padding: 2px 6px; border-radius: 4px; font-weight: 500; font-size: 11px; }
    .badge-long-call { background-color: #d9e1f2; color: #1f3864; padding: 2px 6px; border-radius: 4px; font-weight: 500; font-size: 11px; }
    .badge-short-call { background-color: #f2d9d9; color: #c00000; padding: 2px 6px; border-radius: 4px; font-weight: 500; font-size: 11px; }
    .badge-long-put { background-color: #fff2cc; color: #7f6000; padding: 2px 6px; border-radius: 4px; font-weight: 500; font-size: 11px; }
    .badge-short-put { background-color: #e1d5e7; color: #351c75; padding: 2px 6px; border-radius: 4px; font-weight: 500; font-size: 11px; }
    
    .scrollable-table-container {
        width: 100%;
        overflow-x: auto;
        white-space: nowrap;
        border: 1px solid #e9ecef;
        border-radius: 6px;
        padding: 4px;
        background-color: #ffffff;
    }
    .scrollable-table-container table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }
    .scrollable-table-container th, .scrollable-table-container td {
        padding: 6px 10px;
        text-align: left;
        border-bottom: 1px solid #dee2e6;
    }
    .scrollable-table-container th {
        background-color: #f8f9fa;
        font-weight: bold;
    }

    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- ฟังก์ชันจัดการไฟล์พอร์ตและระบบสำรองข้อมูล ---
ARCHIVE_FILE = "trade_history_archive.csv"
BACKUP_DIR = "backup_portfolios"

if not os.path.exists(BACKUP_DIR):
    os.makedirs(BACKUP_DIR)

def init_archive_file():
    expected_cols = [
        "ID", "Market", "Strategy", "Series", "Type", "Status", 
        "Strike", "Premium", "Contracts", "Commission", "TradeDate", "ExpiryDate", "EntrySpot", "RefID", "PortfolioSource", "SavedAt"
    ]
    if not os.path.exists(ARCHIVE_FILE):
        df_arch = pd.DataFrame(columns=expected_cols)
        df_arch.to_csv(ARCHIVE_FILE, index=False)

init_archive_file()

def get_available_portfolios():
    files = [f for f in os.listdir('.') if f.startswith('portfolio_') and f.endswith('.csv')]
    if not files:
        default_files = ["portfolio_1.csv", "portfolio_2.csv", "portfolio_3.csv"]
        for df_file in default_files:
            if not os.path.exists(df_file):
                empty_df = pd.DataFrame(columns=[
                    "ID", "Market", "Strategy", "Series", "Type", "Status", 
                    "Strike", "Premium", "Contracts", "Commission", "TradeDate", "ExpiryDate", "EntrySpot", "RefID"
                ])
                empty_df.to_csv(df_file, index=False)
        files = default_files
    return sorted(files)

def get_portfolio_summary():
    files = get_available_portfolios()
    summary_data = []
    for f in files:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f)
                num_trades = len(df)
            except:
                num_trades = 0
            file_size = os.path.getsize(f)
            mod_time = datetime.fromtimestamp(os.path.getmtime(f), BANGKOK_TZ).strftime("%Y-%m-%d %H:%M")
        else:
            num_trades = 0
            file_size = 0
            mod_time = "N/A"
        summary_data.append({
            "พอร์ต": f,
            "รายการ": num_trades,
            "ขนาด": f"{file_size} B",
            "แก้ไข": mod_time
        })
    return pd.DataFrame(summary_data)

def load_trades_from_file(file_name):
    expected_cols = [
        "ID", "Market", "Strategy", "Series", "Type", "Status", 
        "Strike", "Premium", "Contracts", "Commission", "TradeDate", "ExpiryDate", "EntrySpot", "RefID"
    ]
    if os.path.exists(file_name):
        try:
            df = pd.read_csv(file_name)
        except:
            df = pd.DataFrame(columns=expected_cols)
            
        for col in expected_cols:
            if col not in df.columns:
                if col == "Status":
                    df[col] = "Open"
                elif col in ["TradeDate", "ExpiryDate"]:
                    df[col] = str(date.today())
                elif col in ["EntrySpot", "Strike", "Premium", "Commission"]:
                    df[col] = 0.0
                elif col in ["Contracts", "ID", "RefID"]:
                    df[col] = 0
                else:
                    df[col] = "N/A"
        return df
    return pd.DataFrame(columns=expected_cols)

def save_trades_to_file(df, file_name):
    df.to_csv(file_name, index=False)
    backup_path = os.path.join(BACKUP_DIR, f"{file_name}.bak")
    df.to_csv(backup_path, index=False)
    
    if os.path.exists(ARCHIVE_FILE):
        df_arch = pd.read_csv(ARCHIVE_FILE)
    else:
        df_arch = pd.DataFrame()
    
    df_copy = df.copy()
    df_copy["PortfolioSource"] = file_name
    df_copy["SavedAt"] = datetime.now(BANGKOK_TZ).strftime("%Y-%m-%d %H:%M:%S")
    
    updated_arch = pd.concat([df_arch, df_copy], ignore_index=True).drop_duplicates(subset=["PortfolioSource", "ID", "TradeDate", "Strike", "Type"], keep="last")
    updated_arch.to_csv(ARCHIVE_FILE, index=False)

# --- สมุดบันทึกประจำพอร์ต ---
def get_journal_filename(portfolio_name):
    return f"journal_{portfolio_name}.csv"

def load_journal(portfolio_name):
    j_file = get_journal_filename(portfolio_name)
    expected_cols = ["Timestamp", "Title", "SpotPrice", "Volatility", "TotalPnL", "Notes"]
    if os.path.exists(j_file):
        try:
            df_j = pd.read_csv(j_file)
        except:
            df_j = pd.DataFrame(columns=expected_cols)
        for col in expected_cols:
            if col not in df_j.columns:
                df_j[col] = ""
        return df_j
    return pd.DataFrame(columns=expected_cols)

def save_journal_entry(portfolio_name, title, spot, vol, pnl, notes):
    j_file = get_journal_filename(portfolio_name)
    df_j = load_journal(portfolio_name)
    new_entry = pd.DataFrame([{
        "Timestamp": datetime.now(BANGKOK_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "Title": title,
        "SpotPrice": spot,
        "Volatility": vol,
        "TotalPnL": pnl,
        "Notes": notes
    }])
    df_j = pd.concat([df_j, new_entry], ignore_index=True)
    df_j.to_csv(j_file, index=False)

# --- ดึงข้อมูลตลาด ---
@st.cache_data(ttl=10)
def get_market_data(market_type):
    try:
        if market_type == "TFEX USD/THB":
            ticker_str, default_p, default_v = "THB=X", 35.0, 0.08
        else:
            ticker_str, default_p, default_v = "^SET50.BK", 950.0, 0.18
            
        ticker = yf.Ticker(ticker_str)
        price = None
        if hasattr(ticker, "fast_info") and "last_price" in ticker.fast_info:
            price = float(ticker.fast_info["last_price"])
        
        hist = ticker.history(period="6mo")
        if not hist.empty and len(hist) > 1:
            hist['Log_Return'] = np.log(hist['Close'] / hist['Close'].shift(1))
            calc_vol = float(hist['Log_Return'].std() * np.sqrt(252))
            if np.isnan(calc_vol) or calc_vol <= 0:
                calc_vol = default_v
        else:
            calc_vol = default_v

        if not price:
            price = float(hist['Close'].iloc[-1]) if not hist.empty else default_p
                
        update_time = datetime.now(BANGKOK_TZ).strftime("%Y-%m-%d %H:%M:%S")
        return price, calc_vol, update_time
    except:
        default_p = 35.0 if market_type == "TFEX USD/THB" else 950.0
        default_v = 0.08 if market_type == "TFEX USD/THB" else 0.18
        update_time = datetime.now(BANGKOK_TZ).strftime("%Y-%m-%d %H:%M:%S")
        return default_p, default_v, update_time

# --- Black-Scholes ---
def bs_option_price(S, K, T, r, sigma, option_type):
    if T <= 0:
        return np.maximum(0, S - K) if option_type == "Call" else np.maximum(0, K - S)
    vol = sigma if sigma > 0 else 0.15
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    if option_type == "Call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-1 * d1)

def bs_greeks(S, K, T, r, sigma, option_type):
    if T <= 0:
        delta = (1.0 if S > K else 0.0) if option_type == "Call" else (-1.0 if S < K else 0.0)
        return {"Delta": delta, "Gamma": 0.0, "Theta": 0.0, "Vega": 0.0}
    vol = sigma if sigma > 0 else 0.15
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    if option_type == "Call":
        delta = norm.cdf(d1)
        theta = (- (S * norm.pdf(d1) * vol) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
    else:
        delta = norm.cdf(d1) - 1
        theta = (- (S * norm.pdf(d1) * vol) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0
    gamma = norm.pdf(d1) / (S * vol * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100.0
    return {"Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega}

# --- UI หลัก: Futures & Options Payoff ---
st.title("📈 Futures & Options Payoff")

# --- Sidebar ---
st.sidebar.header("📁 จัดการพอร์ต")
available_portfolios = get_available_portfolios()
active_portfolio = st.sidebar.selectbox("เลือกพอร์ต", available_portfolios)

st.sidebar.markdown("---")
st.sidebar.subheader("💾 สำรอง / นำเข้า")
current_df_for_download = load_trades_from_file(active_portfolio)
csv_bytes = current_df_for_download.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(label=f"📥 ดาวน์โหลด `{active_portfolio}`", data=csv_bytes, file_name=active_portfolio, mime="text/csv")

uploaded_file = st.sidebar.file_uploader("📤 อัปโหลดพอร์ต", type=["csv"])
if uploaded_file is not None:
    import_target_name = st.sidebar.text_input("ชื่อไฟล์ปลายทาง", value=uploaded_file.name)
    col_imp1, col_imp2 = st.sidebar.columns(2)
    with col_imp1:
        if st.button("✅ ยืนยัน"):
            if import_target_name:
                if not import_target_name.endswith(".csv"): import_target_name += ".csv"
                try:
                    uploaded_file.seek(0)
                    save_trades_to_file(pd.read_csv(uploaded_file), import_target_name)
                    st.sidebar.success("สำเร็จ!")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"ผิดพลาด: {e}")
    with col_imp2:
        if st.button("❌ ยกเลิก"): st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("➕ สร้างพอร์ตใหม่")
new_port_name = st.sidebar.text_input("ชื่อไฟล์พอร์ต (เช่น port_new.csv)")
if st.sidebar.button("สร้างพอร์ต"):
    if new_port_name:
        if not new_port_name.endswith(".csv"): new_port_name += ".csv"
        if not os.path.exists(new_port_name):
            save_trades_to_file(pd.DataFrame(columns=["ID", "Market", "Strategy", "Series", "Type", "Status", "Strike", "Premium", "Contracts", "Commission", "TradeDate", "ExpiryDate", "EntrySpot", "RefID"]), new_port_name)
            st.sidebar.success("สร้างสำเร็จ!")
            st.rerun()
        else:
            st.sidebar.warning("ชื่อซ้ำ")

st.sidebar.markdown("---")
st.sidebar.subheader("📂 สรุปพอร์ต")
st.sidebar.dataframe(get_portfolio_summary(), use_container_width=True, hide_index=True)

st.sidebar.markdown("---")
st.sidebar.header("⚙ ตั้งค่าตลาด")
selected_market = st.sidebar.selectbox("ตลาด", ["TFEX USD/THB", "TFEX SET50"])

current_spot, auto_volatility, last_update_time = get_market_data(selected_market)

col_head1, col_head2 = st.columns([1, 2])
with col_head1:
    if st.button("🔄 รีเฟรช"):
        st.cache_data.clear()
        st.rerun()
with col_head2:
    st.metric(label=f"Spot ({selected_market})", value=f"{current_spot:,.4f}")

st.markdown(f"*อัปเดตล่าสุด:* `{last_update_time}`")
st.markdown("---")

spot_price = st.sidebar.number_input("ราคาอ้างอิง", value=float(current_spot), format="%.4f")
volatility_input = st.sidebar.slider("Volatility (%)", 1.0, 100.0, float(auto_volatility * 100)) / 100.0

risk_free_rate = 0.025
contract_multiplier = 1000 if selected_market == "TFEX USD/THB" else 200

trades_df = load_trades_from_file(active_portfolio)
if not trades_df.empty:
    trades_df["ID"] = pd.to_numeric(trades_df["ID"], errors='coerce').fillna(0).astype(int)
    trades_df["RefID"] = pd.to_numeric(trades_df["RefID"], errors='coerce').fillna(0).astype(int)

current_year = date.today().year
short_year = str(current_year)[-2:]
next_short_year = str(current_year + 1)[-2:]
prefix_code = "USD" if selected_market == "TFEX USD/THB" else "S50"
month_codes = ["H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]
generated_series_options = [f"{prefix_code}{m}{yr}" for yr in [short_year, next_short_year] for m in month_codes]

# --- ฟอร์มเพิ่มสัญญา ---
st.sidebar.subheader("➕ เพิ่มสัญญา")

position_status = st.sidebar.selectbox("สถานะ", options=["Open", "Close"], index=0)

ref_id_input = 0
if position_status == "Close":
    open_list = trades_df[trades_df["Status"] == "Open"]
    if not open_list.empty:
        open_ids = open_list["ID"].tolist()
        ref_id_input = st.sidebar.selectbox("จับคู่ปิดกับ Open ID บรรทัดที่", options=open_ids)
    else:
        st.sidebar.warning("ยังไม่มี Open ในพอร์ตสำหรับจับคู่ปิด")

if "form_key" not in st.session_state: st.session_state["form_key"] = 0

with st.sidebar.form(key=f"trade_form_{st.session_state['form_key']}"):
    strategy_name = st.text_input("กลยุทธ์", value="")
    series_name = st.selectbox("ซีรีส์", options=generated_series_options, index=0)
    position_type = st.selectbox("ประเภท", options=["Long Futures", "Short Futures", "Long Call Option", "Short Call Option", "Long Put Option", "Short Put Option"], index=0)
    
    strike_input_str = st.text_input("Strike Price", value="")
    premium = st.number_input("พรีเมี่ยม / ราคาปิด", value=0.0, format="%.2f")
    contracts = st.number_input("สัญญา", value=1, min_value=1, step=1)
    commission = st.number_input("ค่าคอมฯ/สัญญา", value=0.0, format="%.2f")
    trade_date = st.date_input("วันที่เทรด", value=date.today())
    expiry_date = st.date_input("วันหมดอายุ", value=date.today() + timedelta(days=30))
    
    confirm_add = st.checkbox("☑ ยืนยัน", value=False)
    submitted = st.form_submit_button("บันทึก")
    
    if submitted:
        if not confirm_add:
            st.sidebar.error("โปรดติ๊กยืนยัน")
        elif not strike_input_str.strip():
            st.sidebar.error("โปรดระบุ Strike")
        else:
            try:
                strike_val = float(strike_input_str.strip())
            except ValueError:
                strike_val = None
                st.sidebar.error("Strike ไม่ถูกต้อง")
                
            if strike_val is not None:
                new_id = int(trades_df["ID"].max() + 1) if not trades_df.empty and "ID" in trades_df.columns and not pd.isna(trades_df["ID"].max()) else 1
                new_row = pd.DataFrame([{
                    "ID": int(new_id), "Market": selected_market, "Strategy": strategy_name if strategy_name else "Strategy #1", 
                    "Series": series_name, "Type": position_type, "Status": position_status, 
                    "Strike": float(strike_val), "Premium": premium, "Contracts": contracts, "Commission": commission, 
                    "TradeDate": str(trade_date), "ExpiryDate": str(expiry_date), "EntrySpot": spot_price, "RefID": int(ref_id_input) if position_status == "Close" else 0
                }])
                trades_df = pd.concat([trades_df, new_row], ignore_index=True)
                save_trades_to_file(trades_df, active_portfolio)
                st.sidebar.success("บันทึกสำเร็จ!")
                st.session_state["form_key"] += 1
                st.rerun()

# --- ตารางจัดการข้อมูล ---
st.subheader("📋 จัดการพอร์ตข้อมูลการเทรด")
active_df_to_use = trades_df.copy()

if not trades_df.empty:
    edited_df = st.data_editor(trades_df, num_rows="dynamic", use_container_width=True, key="editor")
    active_df_to_use = edited_df
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("💾 บันทึกตาราง"):
            save_trades_to_file(edited_df, active_portfolio)
            st.success("บันทึกสำเร็จ!")
            st.rerun()
    st.markdown("---")
    del_id = st.selectbox("เลือก ID ที่ต้องการลบ", options=edited_df["ID"].tolist() if not edited_df.empty else [])
    confirm_del = st.checkbox("ยืนยันการลบ", value=False)
    if st.button("🗑️ ลบรายการ"):
        if confirm_del:
            updated_trades = edited_df[edited_df["ID"] != del_id].reset_index(drop=True)
            save_trades_to_file(updated_trades, active_portfolio)
            st.success("ลบเรียบร้อย")
            st.rerun()
        else:
            st.warning("โปรดติ๊กยืนยัน")
else:
    st.info("พอร์ตว่าง")

# --- รวมพอร์ต & Archive ---
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("🔀 รวมพอร์ตและประวัติ")
tab_m1, tab_m2 = st.tabs(["เลือกพอร์ตวิเคราะห์", "ประวัติ (Archive)"])

with tab_m1:
    selected_ports = st.multiselect("เลือกพอร์ต", options=available_portfolios, default=[active_portfolio])
    combined_trades_df = pd.DataFrame()
    if selected_ports:
        dfs = []
        for p in selected_ports:
            p_df = active_df_to_use.copy() if p == active_portfolio else load_trades_from_file(p)
            if not p_df.empty:
                p_df['Source_Portfolio'] = p
                dfs.append(p_df)
        if dfs:
            combined_trades_df = pd.concat(dfs, ignore_index=True)

with tab_m2:
    if os.path.exists(ARCHIVE_FILE):
        arch_df = pd.read_csv(ARCHIVE_FILE)
        if not arch_df.empty:
            arch_ports = arch_df["PortfolioSource"].unique().tolist() if "PortfolioSource" in arch_df.columns else []
            sel_arch = st.selectbox("เลือกจาก Archive", options=arch_ports)
            filt_arch = arch_df[arch_df["PortfolioSource"] == sel_arch] if sel_arch else arch_df
            st.dataframe(filt_arch, use_container_width=True)
            if st.button("📥 กู้คืนชุดนี้"):
                save_trades_to_file(filt_arch.drop(columns=["PortfolioSource", "SavedAt"], errors="ignore"), active_portfolio)
                st.success("กู้คืนสำเร็จ!")
                st.rerun()

market_trades = pd.DataFrame()
if not combined_trades_df.empty and "Market" in combined_trades_df.columns:
    market_trades = combined_trades_df[combined_trades_df["Market"] == selected_market]
    if not market_trades.empty:
        market_trades["ID"] = pd.to_numeric(market_trades["ID"], errors='coerce').fillna(0).astype(int)
        market_trades["RefID"] = pd.to_numeric(market_trades["RefID"], errors='coerce').fillna(0).astype(int)

st.markdown("<br>", unsafe_allow_html=True)

# --- สรุปสัญญาแยกตามประเภท ---
st.subheader("📊 สรุปสัญญา")
if not market_trades.empty:
    types_list = ["Long Futures", "Short Futures", "Long Call Option", "Short Call Option", "Long Put Option", "Short Put Option"]
    contract_summary_data = []
    
    for t_item in types_list:
        sub_df = market_trades[market_trades["Type"] == t_item]
        total_cnt = sub_df["Contracts"].sum() if not sub_df.empty else 0
        open_rows = sub_df[sub_df["Status"] == "Open"]
        close_rows = sub_df[sub_df["Status"] == "Close"]
        open_cnt = open_rows["Contracts"].sum() if not open_rows.empty else 0
        close_cnt = close_rows["Contracts"].sum() if not close_rows.empty else 0
        net_open_cnt = max(0, open_cnt - close_cnt)
        
        badge_class = "badge-long-fu"
        if "Short Futures" in t_item: badge_class = "badge-short-fu"
        elif "Long Call" in t_item: badge_class = "badge-long-call"
        elif "Short Call" in t_item: badge_class = "badge-short-call"
        elif "Long Put" in t_item: badge_class = "badge-long-put"
        elif "Short Put" in t_item: badge_class = "badge-short-put"
        
        contract_summary_data.append({
            "ประเภท": f'<span class="{badge_class}">{t_item}</span>',
            "ทั้งหมด": int(total_cnt),
            "Open": int(open_cnt),
            "Close": int(close_cnt),
            "คงเหลือ": int(net_open_cnt)
        })
    st.markdown(pd.DataFrame(contract_summary_data).to_html(escape=False, index=False), unsafe_allow_html=True)
else:
    st.info("ไม่มีสัญญาในตลาดนี้")

st.markdown("<br>", unsafe_allow_html=True)

# --- ตารางสรุปสถานะและ P&L ปัจจุบัน (บรรทัดเดียว เลื่อนบาร์ได้) ---
st.subheader("📊 สถานะและ P&L ปัจจุบัน")
total_pnl_sum = 0.0
if not market_trades.empty:
    summary_list = []
    open_trades_df = market_trades[market_trades["Status"] == "Open"]
    close_trades_df = market_trades[market_trades["Status"] == "Close"]
    
    matched_close_ids = set()
    row_seq = 1
    
    for _, open_row in open_trades_df.iterrows():
        p_src = open_row.get("Source_Portfolio", active_portfolio)
        open_id = int(open_row["ID"])
        ser, stk, p_type = open_row["Series"], float(open_row["Strike"]), open_row["Type"]
        open_qty, open_prem = int(open_row["Contracts"]), float(open_row["Premium"])
        comm_unit_open = float(open_row["Commission"])
        
        matching_closes = close_trades_df[
            (close_trades_df["RefID"] == open_id) | 
            ((close_trades_df["RefID"] == 0) & 
             (close_trades_df["Series"] == ser) & 
             (close_trades_df["Type"] == p_type) & 
             (('Futures' in p_type) | (close_trades_df["Strike"] == stk)))
        ]
        
        for _, c_item in matching_closes.iterrows():
            matched_close_ids.add(int(c_item["ID"]))
        
        closed_qty = matching_closes["Contracts"].sum() if not matching_closes.empty else 0
        rem_qty = max(0, open_qty - closed_qty)
        
        badge_class = "badge-long-fu"
        if "Short Futures" in p_type: badge_class = "badge-short-fu"
        elif "Long Call" in p_type: badge_class = "badge-long-call"
        elif "Short Call" in p_type: badge_class = "badge-short-call"
        elif "Long Put" in p_type: badge_class = "badge-long-put"
        elif "Short Put" in p_type: badge_class = "badge-short-put"
        type_badge_html = f'<span class="{badge_class}">{p_type}</span>'
        
        rem_pnl = 0.0
        if rem_qty > 0:
            mult_qty = rem_qty * contract_multiplier
            comm_rem = comm_unit_open * rem_qty
            if p_type == "Long Futures": rem_pnl = (spot_price - stk) * mult_qty - comm_rem
            elif p_type == "Short Futures": rem_pnl = (stk - spot_price) * mult_qty - comm_rem
            elif p_type == "Long Call Option": rem_pnl = (np.maximum(0, spot_price - stk) - open_prem) * mult_qty - comm_rem
            elif p_type == "Short Call Option": rem_pnl = (open_prem - np.maximum(0, spot_price - stk)) * mult_qty - comm_rem
            elif p_type == "Long Put Option": rem_pnl = (np.maximum(0, stk - spot_price) - open_prem) * mult_qty - comm_rem
            elif p_type == "Short Put Option": rem_pnl = (open_prem - np.maximum(0, stk - spot_price)) * mult_qty - comm_rem
        
        if matching_closes.empty:
            summary_list.append({
                "ลำดับ": row_seq, "พอร์ต": p_src, "ซีรีส์": ser, "Strike": stk, "ประเภท": type_badge_html,
                "Open": f"ID {open_id} ({open_qty} สัญญา)", "Close": "", "P&L สุทธิ (บาท)": f"{round(rem_pnl, 2):,}"
            })
            row_seq += 1
        else:
            for _, c_row in matching_closes.iterrows():
                c_id = int(c_row["ID"])
                c_qty = int(c_row["Contracts"])
                c_prem = float(c_row["Premium"])
                c_strike_close = float(c_row["Strike"])
                comm_unit_close = float(c_row["Commission"])
                
                c_comm = (comm_unit_open * c_qty) + (comm_unit_close * c_qty)
                mult_c = c_qty * contract_multiplier
                
                if "Futures" in p_type:
                    c_pnl = ((c_strike_close - stk) * mult_c) - c_comm if "Long" in p_type else ((stk - c_strike_close) * mult_c) - c_comm
                else:
                    c_pnl = ((c_prem - open_prem) * mult_c) - c_comm if "Long" in p_type else ((open_prem - c_prem) * mult_c) - c_comm
                
                total_paired_pnl = rem_pnl + c_pnl
                summary_list.append({
                    "ลำดับ": row_seq, "พอร์ต": p_src, "ซีรีส์": ser, "Strike": stk, "ประเภท": type_badge_html,
                    "Open": f"ID {open_id} (เหลือ {rem_qty})" if rem_qty > 0 else f"ID {open_id} (ปิดหมด)", 
                    "Close": f"ID {c_id} ({c_qty} @ {c_strike_close if 'Futures' in p_type else c_prem})", 
                    "P&L สุทธิ (บาท)": f"{round(total_paired_pnl, 2):,}"
                })
                row_seq += 1

    unmatched_closes = close_trades_df[~close_trades_df["ID"].isin(matched_close_ids)]
    for _, c_row in unmatched_closes.iterrows():
        p_src = c_row.get("Source_Portfolio", active_portfolio)
        c_id = int(c_row["ID"])
        ser, stk, p_type = c_row["Series"], float(c_row["Strike"]), c_row["Type"]
        c_qty = int(c_row["Contracts"])
        c_prem = float(c_row["Premium"])
        c_comm = float(c_row["Commission"]) * c_qty
        mult_c = c_qty * contract_multiplier
        
        badge_class = "badge-long-fu"
        if "Short Futures" in p_type: badge_class = "badge-short-fu"
        elif "Long Call" in p_type: badge_class = "badge-long-call"
        elif "Short Call" in p_type: badge_class = "badge-short-call"
        elif "Long Put" in p_type: badge_class = "badge-long-put"
        elif "Short Put" in p_type: badge_class = "badge-short-put"
        type_badge_html = f'<span class="{badge_class}">{p_type}</span>'
        
        standalone_pnl = (c_prem * mult_c) - c_comm
        summary_list.append({
            "ลำดับ": row_seq, "พอร์ต": p_src, "ซีรีส์": ser, "Strike": stk, "ประเภท": type_badge_html,
            "Open": "-", "Close": f"ID {c_id} (Unmatched, {c_qty} สัญญา)", "P&L สุทธิ (บาท)": f"{round(standalone_pnl, 2):,}"
        })
        row_seq += 1

    summary_df = pd.DataFrame(summary_list)
    if not summary_df.empty:
        raw_pnl_list = [float(str(item["P&L สุทธิ (บาท)"]).replace(",", "")) for item in summary_list]
        total_pnl_sum = sum(raw_pnl_list)
        total_sum_row = pd.DataFrame([{
            "ลำดับ": "", "พอร์ต": "รวมทั้งหมด", "ซีรีส์": "-", "Strike": "-", "ประเภท": "-",
            "Open": "", "Close": "", "P&L สุทธิ (บาท)": f"{round(total_pnl_sum, 2):,}"
        }])
        summary_df = pd.concat([summary_df, total_sum_row], ignore_index=True)
    
    st.markdown(f'<div class="scrollable-table-container">{summary_df.to_html(escape=False, index=False)}</div>', unsafe_allow_html=True)
else:
    st.info("ไม่มีสัญญาในตลาดนี้")

st.markdown("<br>", unsafe_allow_html=True)

# --- สมุดบันทึกพอร์ต ---
st.subheader(f"📝 บันทึก: `{active_portfolio}`")
with st.form("journal_form"):
    j_title = st.text_input("หัวข้อ", "บันทึกประจำวัน")
    j_notes = st.text_area("รายละเอียด")
    if st.form_submit_button("💾 บันทึก"):
        save_journal_entry(active_portfolio, j_title, spot_price, volatility_input, total_pnl_sum, j_notes)
        st.success("บันทึกสำเร็จ!")
        st.rerun()

journal_df = load_journal(active_portfolio)
if not journal_df.empty:
    for idx, row in journal_df.iloc[::-1].iterrows():
        with st.expander(f"📌 [{row['Timestamp']}] {row['Title']} (P&L: {row['TotalPnL']:,.2f} ฿)"):
            st.markdown(f"Spot: `{row['SpotPrice']}` | Vol: `{float(row['Volatility'])*100:.2f}%`")
            st.markdown(f"> {row['Notes']}")

st.markdown("<br><br>", unsafe_allow_html=True)

# ==========================================
# กราฟที่ 1: Net Expiry Payoff (ปรับแยกสีพื้นที่ใต้กราฟ ฟ้า/แดง)
# ==========================================
st.subheader(f"📈 1. Payoff วันหมดอายุ ({selected_market})")
payoff_mode = st.selectbox("หน่วย", ["บาทรวม (THB)", "จุด (Points)"])

price_range = np.linspace(spot_price * 0.85, spot_price * 1.15, 300)
total_payoff = np.zeros_like(price_range)
total_greeks = {"Delta": 0.0, "Gamma": 0.0, "Theta": 0.0, "Vega": 0.0}

today = date.today()
fig = go.Figure()

if not market_trades.empty:
    open_trades_df = market_trades[market_trades["Status"] == "Open"]
    close_trades_df = market_trades[market_trades["Status"] == "Close"]
    
    for _, open_row in open_trades_df.iterrows():
        open_id = int(open_row["ID"])
        p_type, stk, prem = open_row["Type"], float(open_row["Strike"]), float(open_row["Premium"])
        open_qty = int(open_row["Contracts"])
        comm_unit_open = float(open_row["Commission"])
        
        try:
            T = max((datetime.strptime(str(open_row["ExpiryDate"]), "%Y-%m-%d").date() - today).days, 0) / 365.0
        except:
            T = 30 / 365.0
            
        matching_closes = close_trades_df[
            (close_trades_df["RefID"] == open_id) | 
            ((close_trades_df["RefID"] == 0) & 
             (close_trades_df["Series"] == open_row["Series"]) & 
             (close_trades_df["Type"] == p_type) & 
             (('Futures' in p_type) | (close_trades_df["Strike"] == stk)))
        ]
        
        closed_qty = matching_closes["Contracts"].sum() if not matching_closes.empty else 0
        rem_qty = max(0, open_qty - closed_qty)
        
        if rem_qty > 0:
            mult_qty = (rem_qty * contract_multiplier if payoff_mode == "บาทรวม (THB)" else rem_qty)
            comm_rem = (comm_unit_open * rem_qty) if payoff_mode == "บาทรวม (THB)" else ((comm_unit_open * rem_qty) / contract_multiplier)
            payoff_rem = np.zeros_like(price_range)
            
            if p_type == "Long Futures": payoff_rem = (price_range - stk) * mult_qty - comm_rem
            elif p_type == "Short Futures": payoff_rem = (stk - price_range) * mult_qty - comm_rem
            elif p_type == "Long Call Option": payoff_rem = (np.maximum(0, price_range - stk) - prem) * mult_qty - comm_rem
            elif p_type == "Short Call Option": payoff_rem = (prem - np.maximum(0, price_range - stk)) * mult_qty - comm_rem
            elif p_type == "Long Put Option": payoff_rem = (np.maximum(0, stk - price_range) - prem) * mult_qty - comm_rem
            elif p_type == "Short Put Option": payoff_rem = (prem - np.maximum(0, stk - price_range)) * mult_qty - comm_rem
                
            total_payoff += payoff_rem
            opt_flag = "Call" if "Call" in p_type else ("Put" if "Put" in p_type else None)
            if opt_flag:
                g = bs_greeks(spot_price, stk, T, risk_free_rate, volatility_input, opt_flag)
                dir_s = 1 if "Long" in p_type else -1
                total_greeks["Delta"] += g["Delta"] * rem_qty * contract_multiplier * dir_s
                total_greeks["Gamma"] += g["Gamma"] * rem_qty * contract_multiplier * dir_s
                total_greeks["Theta"] += g["Theta"] * rem_qty * dir_s
                total_greeks["Vega"] += g["Vega"] * rem_qty * contract_multiplier * dir_s
            else:
                total_greeks["Delta"] += 1.0 * rem_qty * contract_multiplier * (1 if "Long" in p_type else -1)
            
        if not matching_closes.empty:
            for _, c_row in matching_closes.iterrows():
                c_qty = int(c_row["Contracts"])
                c_prem = float(c_row["Premium"])
                c_strike_close = float(c_row["Strike"])
                comm_unit_close = float(c_row["Commission"])
                
                c_comm = (comm_unit_open * c_qty) + (comm_unit_close * c_qty)
                c_comm_scaled = c_comm if payoff_mode == "บาทรวม (THB)" else (c_comm / contract_multiplier)
                mult_c = (c_qty * contract_multiplier if payoff_mode == "บาทรวม (THB)" else c_qty)
                
                if "Futures" in p_type:
                    locked_val = ((c_strike_close - stk) * mult_c) - c_comm_scaled if "Long" in p_type else ((stk - c_strike_close) * mult_c) - c_comm_scaled
                else:
                    locked_val = ((c_prem - prem) * mult_c) - c_comm_scaled if "Long" in p_type else ((prem - c_prem) * mult_c) - c_comm_scaled
                    
                total_payoff += np.full_like(price_range, locked_val)

be_points = []
for j in range(len(price_range) - 1):
    if total_payoff[j] * total_payoff[j+1] < 0:
        x1, x2 = price_range[j], price_range[j+1]
        y1, y2 = total_payoff[j], total_payoff[j+1]
        if y2 - y1 != 0: be_points.append(x1 - y1 * (x2 - x1) / (y2 - y1))

max_pnl_val, price_at_max = total_payoff[np.argmax(total_payoff)], price_range[np.argmax(total_payoff)]
min_pnl_val, price_at_min = total_payoff[np.argmin(total_payoff)], price_range[np.argmin(total_payoff)]
unit_label = "บาท" if payoff_mode == "บาทรวม (THB)" else "จุด"

st.markdown("#### 🎯 สรุปความเสี่ยง")
sc1, sc2, sc3, sc4 = st.columns(4)
with sc1: st.markdown(f"**กำไรสูงสุด:** `{max_pnl_val:,.2f} {unit_label}` (ที่ {price_at_max:,.2f})")
with sc2: st.markdown(f"**ขาดทุนสูงสุด:** `{min_pnl_val:,.2f} {unit_label}` (ที่ {price_at_min:,.2f})")
with sc3: st.markdown(f"**ซ้ายสุด:** `{total_payoff[0]:,.2f} {unit_label}`")
with sc4: st.markdown(f"**ขวาสุด:** `{total_payoff[-1]:,.2f} {unit_label}`")

be_str = ", ".join([f"`{bp:,.2f}`" for bp in be_points]) if be_points else "ไม่มีจุดคุ้มทุน"
st.markdown(f"💡 **จุดคุ้มทุน (BE):** {be_str}")
st.markdown("<br>", unsafe_allow_html=True)

# --- วาดกราฟ Net Expiry Payoff พร้อมแยกสีพื้นที่กำไร (ฟ้า) และขาดทุน (แดงอ่อน) ---
# พื้นที่กำไร (y >= 0)
fig.add_trace(go.Scatter(
    x=price_range, 
    y=np.where(total_payoff >= 0, total_payoff, 0), 
    mode='lines', 
    name='Profit Zone', 
    line=dict(width=0), 
    fill='tozeroy', 
    fillcolor='rgba(0, 123, 255, 0.15)',
    showlegend=False
))

# พื้นที่ขาดทุน (y < 0)
fig.add_trace(go.Scatter(
    x=price_range, 
    y=np.where(total_payoff < 0, total_payoff, 0), 
    mode='lines', 
    name='Loss Zone', 
    line=dict(width=0), 
    fill='tozeroy', 
    fillcolor='rgba(255, 0, 0, 0.12)',
    showlegend=False
))

# เส้นกราฟหลัก Net Payoff
fig.add_trace(go.Scatter(
    x=price_range, 
    y=total_payoff, 
    mode='lines', 
    name='Net Payoff', 
    line=dict(color='blue', width=3)
))

if be_points:
    fig.add_trace(go.Scatter(x=be_points, y=[0]*len(be_points), mode='markers+text', name='Break-even', marker=dict(color='orange', size=10, symbol='diamond'), text=[f"BE: {bp:.2f}" for bp in be_points], textposition="bottom center"))

fig.add_hline(y=0, line_color="black", line_width=1)
fig.add_vline(x=spot_price, line_dash="dot", line_color="red", line_width=2)
spot_pnl_expiry = np.interp(spot_price, price_range, total_payoff)
fig.add_annotation(x=spot_price, y=1.03, yref="paper", text=f"Spot: {spot_price:,.2f} ({spot_pnl_expiry:+,.2f})", showarrow=False, font=dict(color="red", size=13))

fig.update_layout(
    title=dict(text=f"Net Expiry Payoff ({payoff_mode})", y=0.98, x=0.0, xanchor='left', yanchor='top'),
    xaxis_title="Underlying Price", yaxis_title=f"P&L ({payoff_mode})",
    hovermode="x unified", template="plotly_white", height=500, margin=dict(t=60, b=40, l=40, r=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<br><br>", unsafe_allow_html=True)

# ==========================================
# กราฟที่ 2 & 3: Time Decay & Greeks
# ==========================================
st.subheader("⏱ 2. จำลอง Payoff รายวัน")
sim_date = st.date_input("วันที่จำลอง", value=date.today())
sim_date_str = str(sim_date)

fig_daily_single = go.Figure()
fig_daily_cumulative = go.Figure()

if not market_trades.empty:
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    unique_dates = sorted(list(set(market_trades["TradeDate"].astype(str))))
    if sim_date_str not in unique_dates:
        unique_dates.append(sim_date_str)
        unique_dates = sorted(unique_dates)

    for i, t_str in enumerate(unique_dates):
        t_d = datetime.strptime(t_str, "%Y-%m-%d").date() if t_str else date.today()
        day_val = np.zeros_like(price_range)
        has_t = False
        
        day_trades = market_trades[market_trades["TradeDate"].astype(str) == t_str]
        if not day_trades.empty:
            has_t = True
            for _, row in day_trades.iterrows():
                p_type, stk, prem, qty, status = row["Type"], float(row["Strike"]), float(row["Premium"]), int(row["Contracts"]), row["Status"]
                comm_val = (float(row["Commission"]) * qty) if payoff_mode == "บาทรวม (THB)" else ((float(row["Commission"]) * qty) / contract_multiplier)
                
                try:
                    T_sim = max((datetime.strptime(str(row["ExpiryDate"]), "%Y-%m-%d").date() - t_d).days, 0) / 365.0
                except:
                    T_sim = 30 / 365.0
                    
                if status == "Open":
                    mult_qty = (qty * contract_multiplier if payoff_mode == "บาทรวม (THB)" else qty)
                    if p_type == "Long Futures": val = (price_range - stk) * mult_qty - comm_val
                    elif p_type == "Short Futures": val = (stk - price_range) * mult_qty - comm_val
                    elif "Call" in p_type:
                        opt_v = np.array([bs_option_price(p, stk, T_sim, risk_free_rate, volatility_input, "Call") for p in price_range])
                        val = (opt_v - prem) * mult_qty - comm_val if "Long" in p_type else (prem - opt_v) * mult_qty - comm_val
                    elif "Put" in p_type:
                        opt_v = np.array([bs_option_price(p, stk, T_sim, risk_free_rate, volatility_input, "Put") for p in price_range])
                        val = (opt_v - prem) * mult_qty - comm_val if "Long" in p_type else (prem - opt_v) * mult_qty - comm_val
                    else:
                        val = -comm_val
                else:
                    open_id_ref = int(row["RefID"])
                    matching_open = market_trades[market_trades["ID"] == open_id_ref]
                    entry_price = float(matching_open["Strike"].values[0]) if not matching_open.empty and "Futures" in p_type else float(matching_open["Premium"].values[0]) if not matching_open.empty else prem
                    mult_qty = (qty * contract_multiplier if payoff_mode == "บาทรวม (THB)" else qty)
                    if "Futures" in p_type:
                        val = np.full_like(price_range, ((prem - entry_price) * mult_qty) - comm_val) if "Long" in p_type else np.full_like(price_range, ((entry_price - prem) * mult_qty) - comm_val)
                    else:
                        val = np.full_like(price_range, ((prem - entry_price) * mult_qty) - comm_val) if "Long" in p_type else np.full_like(price_range, ((entry_price - prem) * mult_qty) - comm_val)
                day_val += val
                
        if has_t:
            is_sel = (t_str == sim_date_str)
            fig_daily_single.add_trace(go.Scatter(x=price_range, y=day_val, mode='lines', name=t_str, line=dict(color=colors[i % len(colors)], width=3 if is_sel else 1.5, dash='solid' if is_sel else 'dash')))

    for i, t_str in enumerate(unique_dates):
        if t_str > sim_date_str: continue
        t_d = datetime.strptime(t_str, "%Y-%m-%d").date() if t_str else date.today()
        cum_val = np.zeros_like(price_range)
        cum_trades = market_trades[market_trades["TradeDate"].astype(str) <= t_str]
        
        if not cum_trades.empty:
            for _, row in cum_trades.iterrows():
                p_type, stk, prem, qty, status = row["Type"], float(row["Strike"]), float(row["Premium"]), int(row["Contracts"]), row["Status"]
                comm_val = (float(row["Commission"]) * qty) if payoff_mode == "บาทรวม (THB)" else ((float(row["Commission"]) * qty) / contract_multiplier)
                
                try:
                    T_sim = max((datetime.strptime(str(row["ExpiryDate"]), "%Y-%m-%d").date() - t_d).days, 0) / 365.0
                except:
                    T_sim = 30 / 365.0
                    
                if status == "Open":
                    mult_qty = (qty * contract_multiplier if payoff_mode == "บาทรวม (THB)" else qty)
                    if p_type == "Long Futures": val = (price_range - stk) * mult_qty - comm_val
                    elif p_type == "Short Futures": val = (stk - price_range) * mult_qty - comm_val
                    elif "Call" in p_type:
                        opt_v = np.array([bs_option_price(p, stk, T_sim, risk_free_rate, volatility_input, "Call") for p in price_range])
                        val = (opt_v - prem) * mult_qty - comm_val if "Long" in p_type else (prem - opt_v) * mult_qty - comm_val
                    elif "Put" in p_type:
                        opt_v = np.array([bs_option_price(p, stk, T_sim, risk_free_rate, volatility_input, "Put") for p in price_range])
                        val = (opt_v - prem) * mult_qty - comm_val if "Long" in p_type else (prem - opt_v) * mult_qty - comm_val
                    else:
                        val = -comm_val
                else:
                    open_id_ref = int(row["RefID"])
                    matching_open = market_trades[market_trades["ID"] == open_id_ref]
                    entry_price = float(matching_open["Strike"].values[0]) if not matching_open.empty and "Futures" in p_type else float(matching_open["Premium"].values[0]) if not matching_open.empty else prem
                    mult_qty = (qty * contract_multiplier if payoff_mode == "บาทรวม (THB)" else qty)
                    if "Futures" in p_type:
                        val = np.full_like(price_range, ((prem - entry_price) * mult_qty) - comm_val) if "Long" in p_type else np.full_like(price_range, ((entry_price - prem) * mult_qty) - comm_val)
                    else:
                        val = np.full_like(price_range, ((prem - entry_price) * mult_qty) - comm_val) if "Long" in p_type else np.full_like(price_range, ((entry_price - prem) * mult_qty) - comm_val)
                cum_val += val
                
        is_sel = (t_str == sim_date_str)
        fig_daily_cumulative.add_trace(go.Scatter(x=price_range, y=cum_val, mode='lines', name=t_str, line=dict(color=colors[i % len(colors)], width=3.5 if is_sel else 1.5, dash='solid' if is_sel else 'dash')))

fig_daily_single.add_hline(y=0, line_color="black", line_width=1)
fig_daily_single.add_vline(x=spot_price, line_dash="dot", line_color="red", line_width=2)
fig_daily_single.update_layout(title="Payoff เฉพาะวันที่เทรด", template="plotly_white", height=380, margin=dict(t=40, b=30, l=30, r=30), legend=dict(orientation="h", y=1.1))

fig_daily_cumulative.add_hline(y=0, line_color="black", line_width=1)
fig_daily_cumulative.add_vline(x=spot_price, line_dash="dot", line_color="red", line_width=2)
fig_daily_cumulative.update_layout(title="Payoff สะสมถึงวันที่เลือก", template="plotly_white", height=380, margin=dict(t=40, b=30, l=30, r=30), legend=dict(orientation="h", y=1.1))

st.plotly_chart(fig_daily_single, use_container_width=True)
st.plotly_chart(fig_daily_cumulative, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"### 📋 รายการเทรด (ณ วันที่ `{sim_date_str}`)")
if not market_trades.empty:
    filt_sim = market_trades[market_trades["TradeDate"].astype(str) <= sim_date_str]
    if not filt_sim.empty:
        st.dataframe(filt_sim[["ID", "Strategy", "Series", "Type", "Status", "Strike", "Premium", "Contracts", "Commission", "RefID", "TradeDate"]], use_container_width=True)
    else:
        st.info("ไม่มีรายการ")

st.markdown("<br><br>", unsafe_allow_html=True)

# ==========================================
# สรุป Greeks
# ==========================================
st.subheader("📐 สรุป Greeks พอร์ต")
g1, g2, g3, g4 = st.columns(4)
g1.metric("Delta", f"{total_greeks['Delta']:,.2f}")
g2.metric("Gamma", f"{total_greeks['Gamma']:,.4f}")
g3.metric("Theta (Daily)", f"{total_greeks['Theta']:,.2f} ฿")
g4.metric("Vega", f"{total_greeks['Vega']:,.2f}")
