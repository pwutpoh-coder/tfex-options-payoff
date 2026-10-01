import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import os
from datetime import datetime, date, timezone, timedelta
from scipy.stats import norm

# ตั้งค่าหน้าเว็บให้รองรับการใช้งานบนมือถือ (Mobile Responsive)
st.set_page_config(
    page_title="TFEX Multi-Asset Payoff & Greeks Dashboard Pro", 
    layout="wide",
    initial_sidebar_state="auto"
)

# กำหนดโซนเวลา Bangkok (UTC+7)
BANGKOK_TZ = timezone(timedelta(hours=7))

# --- CSS พิเศษช่วยปรับแต่งการแสดงผลบนมือถือให้สวยงามยิ่งขึ้น ---
st.markdown("""
    <style>
    .stMetric {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- ฟังก์ชันจัดการไฟล์พอร์ตหลายพอร์ต และระบบประวัติถาวร (Archive & Backup) ---
ARCHIVE_FILE = "trade_history_archive.csv"
BACKUP_DIR = "backup_portfolios"

if not os.path.exists(BACKUP_DIR):
    os.makedirs(BACKUP_DIR)

def init_archive_file():
    expected_cols = [
        "ID", "Market", "Strategy", "Series", "Type", "Status", 
        "Strike", "Premium", "Contracts", "Commission", "TradeDate", "ExpiryDate", "EntrySpot", "PortfolioSource", "SavedAt"
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
                    "Strike", "Premium", "Contracts", "Commission", "TradeDate", "ExpiryDate", "EntrySpot"
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
            mod_time = datetime.fromtimestamp(os.path.getmtime(f), BANGKOK_TZ).strftime("%Y-%m-%d %H:%M:%S")
        else:
            num_trades = 0
            file_size = 0
            mod_time = "N/A"
        summary_data.append({
            "พอร์ต": f,
            "รายการ": num_trades,
            "ขนาด (B)": file_size,
            "แก้ไขล่าสุด": mod_time
        })
    return pd.DataFrame(summary_data)

def load_trades_from_file(file_name):
    expected_cols = [
        "ID", "Market", "Strategy", "Series", "Type", "Status", 
        "Strike", "Premium", "Contracts", "Commission", "TradeDate", "ExpiryDate", "EntrySpot"
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
                elif col == "EntrySpot":
                    df[col] = 0.0
                elif col in ["Strike", "Premium", "Contracts", "Commission", "ID"]:
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

# --- ฟังก์ชันจัดการสมุดบันทึกประจำพอร์ต ---
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

# --- ดึงข้อมูลราคาและคำนวณ Volatility จาก Yahoo Finance ---
@st.cache_data(ttl=10)
def get_market_data(market_type):
    try:
        if market_type == "TFEX USD/THB":
            ticker_str = "THB=X"
            default_p = 35.0
            default_v = 0.08
        else:
            ticker_str = "^SET50.BK"
            default_p = 950.0
            default_v = 0.18
            
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
                
        update_time = datetime.now(BANGKOK_TZ).strftime("%Y-%m-%d %H:%M:%S (ICT)")
        return price, calc_vol, update_time
    except:
        default_p = 35.0 if market_type == "TFEX USD/THB" else 950.0
        default_v = 0.08 if market_type == "TFEX USD/THB" else 0.18
        update_time = datetime.now(BANGKOK_TZ).strftime("%Y-%m-%d %H:%M:%S (ICT)")
        return default_p, default_v, update_time

# --- Black-Scholes Option Pricing ---
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

# --- Greeks Calculation ---
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

# --- UI หลัก ---
st.title("📈 TFEX Multi-Asset Options & Futures Pro")
st.markdown("ระบบวิเคราะห์ Payoff Chart และความเสี่ยงพอร์ต TFEX (บันทึกข้อมูลอัตโนมัติ & สำรองไฟล์ถาวร)")

# --- Sidebar ---
st.sidebar.header("📁 จัดการพอร์ต")
available_portfolios = get_available_portfolios()
selected_portfolio_tab = st.sidebar.selectbox("เลือกพอร์ต", available_portfolios)
active_portfolio = selected_portfolio_tab

# --- ระบบจัดการไฟล์อัปโหลดพร้อมปุ่มยืนยันและตั้งชื่อพอร์ตปลายทาง ---
st.sidebar.markdown("---")
st.sidebar.subheader("💾 สำรอง / กู้คืน / นำเข้าไฟล์พอร์ต")

current_df_for_download = load_trades_from_file(active_portfolio)
csv_bytes = current_df_for_download.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label=f"📥 ดาวน์โหลดไฟล์ `{active_portfolio}`",
    data=csv_bytes,
    file_name=active_portfolio,
    mime="text/csv",
    help="ดาวน์โหลดไฟล์ CSV นี้เก็บไว้ในเครื่องคอมพิวเตอร์ของคุณ"
)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📤 อัปโหลดไฟล์พอร์ตจากเครื่อง", type=["csv"], help="เลือกไฟล์ CSV ที่เคยดาวน์โหลดไว้เพื่อนำเข้าสู่ระบบ")

if uploaded_file is not None:
    st.sidebar.info("📂 ตรวจพบไฟล์ที่อัปโหลด กรุณาระบุชื่อพอร์ตปลายทางด้านล่างเพื่อยืนยันการนำเข้า")
    import_target_name = st.sidebar.text_input("ตั้งชื่อไฟล์พอร์ตปลายทาง (เช่น my_imported_port.csv)", value=uploaded_file.name)
    
    col_imp1, col_imp2 = st.sidebar.columns(2)
    with col_imp1:
        if st.button("✅ ยืนยันนำเข้าไฟล์นี้"):
            if import_target_name:
                if not import_target_name.endswith(".csv"):
                    import_target_name += ".csv"
                try:
                    uploaded_file.seek(0)
                    imported_df = pd.read_csv(uploaded_file)
                    save_trades_to_file(imported_df, import_target_name)
                    st.sidebar.success(f"นำเข้าและบันทึกข้อมูลลงพอร์ต `{import_target_name}` สำเร็จ!")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")
            else:
                st.sidebar.warning("กรุณาตั้งชื่อไฟล์ปลายทางก่อน")
    with col_imp2:
        if st.button("❌ ยกเลิก"):
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("➕ สร้างพอร์ตใหม่")
new_port_name = st.sidebar.text_input("ชื่อไฟล์พอร์ตใหม่ (เช่น portfolio_4.csv หรือ my_strategy.csv)")
if st.sidebar.button("สร้างพอร์ต"):
    if new_port_name:
        if not new_port_name.endswith(".csv"):
            new_port_name += ".csv"
        if not os.path.exists(new_port_name):
            empty_df = pd.DataFrame(columns=[
                "ID", "Market", "Strategy", "Series", "Type", "Status", 
                "Strike", "Premium", "Contracts", "Commission", "TradeDate", "ExpiryDate", "EntrySpot"
            ])
            save_trades_to_file(empty_df, new_port_name)
            st.sidebar.success("สร้างสำเร็จ!")
            active_portfolio = new_port_name
            st.rerun()
        else:
            st.sidebar.warning("ชื่อซ้ำในระบบ")

st.sidebar.markdown("---")
st.sidebar.subheader("📂 สรุปพอร์ตทั้งหมด")
st.sidebar.dataframe(get_portfolio_summary(), use_container_width=True, hide_index=True)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ ตั้งค่าตลาด")
selected_market = st.sidebar.selectbox("เลือกตลาด", ["TFEX USD/THB", "TFEX SET50"])

current_spot, auto_volatility, last_update_time = get_market_data(selected_market)

col_head1, col_head2 = st.columns([1, 2])
with col_head1:
    if st.button("🔄 รีเฟรชราคา"):
        st.cache_data.clear()
        st.rerun()
with col_head2:
    st.metric(label=f"Spot ({selected_market})", value=f"{current_spot:,.4f}")

st.markdown(f"*อัปเดต:* `{last_update_time}`")
st.markdown("---")

spot_price = st.sidebar.number_input("ราคาอ้างอิง", value=float(current_spot), format="%.4f")
volatility_input = st.sidebar.slider("Volatility (%)", 1.0, 100.0, float(auto_volatility * 100)) / 100.0

risk_free_rate = 0.025
contract_multiplier = 1000 if selected_market == "TFEX USD/THB" else 200

trades_df = load_trades_from_file(active_portfolio)

# --- Strike และ Series ---
if selected_market == "TFEX USD/THB":
    base_center = round(spot_price * 4) / 4.0
    step = 0.25
    base_strikes = [f"{round(base_center + i * step, 2):.2f}" for i in range(-25, 26)]
else:
    base_center = round(spot_price / 10.0) * 10.0
    step = 10.0
    base_strikes = [int(round(base_center + i * step)) for i in range(-20, 21)]

current_year = date.today().year
short_year = str(current_year)[-2:]
next_short_year = str(current_year + 1)[-2:]
prefix_code = "USD" if selected_market == "TFEX USD/THB" else "S50"
month_codes = ["H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]

generated_series_options = [f"{prefix_code}{m}{yr}" for yr in [short_year, next_short_year] for m in month_codes]

# --- ฟอร์มเพิ่มสัญญา ---
st.sidebar.subheader("➕ เพิ่มสัญญาใหม่")
if "form_key" not in st.session_state:
    st.session_state["form_key"] = 0

with st.sidebar.form(key=f"trade_form_{st.session_state['form_key']}"):
    strategy_name = st.text_input("ชื่อกลยุทธ์", "Strategy #1")
    series_name = st.selectbox("ซีรีส์", options=generated_series_options)
    position_status = st.selectbox("สถานะ", ["Open", "Close"])
    position_type = st.selectbox("ประเภท", ["Long Futures", "Short Futures", "Long Call Option", "Short Call Option", "Long Put Option", "Short Put Option"])
    
    strike = st.number_input("Strike Price", value=float(round(spot_price, 2)), format="%.2f")
    premium = st.number_input("ราคาพรีเมี่ยม", value=float(0.25 if selected_market == "TFEX USD/THB" else 15.0), format="%.2f")
    contracts = st.number_input("จำนวนสัญญา", value=1, min_value=1, step=1)
    commission = st.number_input("ค่าคอมฯ รวม/สัญญา", value=float(30.0 if selected_market == "TFEX USD/THB" else 60.0), format="%.2f")
    
    trade_date = st.date_input("วันที่ซื้อ", value=date.today())
    expiry_date = st.date_input("วันหมดอายุ", value=date.today() + timedelta(days=30))
    
    confirm_add = st.checkbox("☑️️ ยืนยันข้อมูล", value=False)
    submitted = st.form_submit_button("บันทึกสัญญา")
    
    if submitted:
        if not confirm_add:
            st.sidebar.error("โปรดติ๊กยืนยันข้อมูล")
        else:
            new_id = int(trades_df["ID"].max() + 1) if not trades_df.empty and "ID" in trades_df.columns and not pd.isna(trades_df["ID"].max()) else 1
            new_row = pd.DataFrame([{
                "ID": new_id, "Market": selected_market, "Strategy": strategy_name, "Series": series_name,
                "Type": position_type, "Status": position_status, "Strike": strike, "Premium": premium,
                "Contracts": contracts, "Commission": commission, "TradeDate": str(trade_date),
                "ExpiryDate": str(expiry_date), "EntrySpot": spot_price
            }])
            trades_df = pd.concat([trades_df, new_row], ignore_index=True)
            save_trades_to_file(trades_df, active_portfolio)
            st.sidebar.success("บันทึกสำเร็จ!")
            st.session_state["form_key"] += 1
            st.rerun()

# --- ตารางแก้ไขข้อมูล ---
st.subheader(f"📋 จัดการข้อมูลพอร์ต: `{active_portfolio}`")
active_df_to_use = trades_df.copy()

if not trades_df.empty:
    edited_df = st.data_editor(trades_df, num_rows="dynamic", use_container_width=True, key="editor")
    active_df_to_use = edited_df
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("💾 บันทึกการแก้ไขตาราง"):
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
            st.warning("โปรดติ๊กยืนยันการลบ")
else:
    st.info("พอร์ตว่าง ไม่มีรายการเทรด")

# --- รวมพอร์ต & Archive ---
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("🔀 รวมพอร์ตและประวัติถาวร")
tab_m1, tab_m2 = st.tabs(["เลือกพอร์ตวิเคราะห์", "ประวัติถาวร (Archive)"])

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
            sel_arch = st.selectbox("เลือกพอร์ตจาก Archive", options=arch_ports)
            filt_arch = arch_df[arch_df["PortfolioSource"] == sel_arch] if sel_arch else arch_df
            st.dataframe(filt_arch, use_container_width=True)
            if st.button("📥 กู้คืนข้อมูลชุดนี้"):
                restore_clean = filt_arch.drop(columns=["PortfolioSource", "SavedAt"], errors="ignore")
                save_trades_to_file(restore_clean, active_portfolio)
                st.success("กู้คืนสำเร็จ!")
                st.rerun()

market_trades = pd.DataFrame()
if not combined_trades_df.empty and "Market" in combined_trades_df.columns:
    market_trades = combined_trades_df[combined_trades_df["Market"] == selected_market]

st.markdown("<br>", unsafe_allow_html=True)

# --- ตารางสรุป P&L ---
st.subheader("📊 ตารางสรุปสถานะและ P&L ปัจจุบัน")
total_pnl_sum = 0.0
if not market_trades.empty:
    summary_list = []
    for idx, row in market_trades.iterrows():
        p_src = row.get("Source_Portfolio", active_portfolio)
        ser, stk, p_type, status = row["Series"], float(row["Strike"]), row["Type"], row["Status"]
        qty, prem, comm = int(row["Contracts"]), float(row["Premium"]), float(row["Commission"]) * int(row["Contracts"])
        
        sign_m = 1 if status == "Open" else -1
        mult_qty = qty * contract_multiplier * sign_m
        
        if p_type == "Long Futures":
            item_pnl = (spot_price - stk) * mult_qty - comm
        elif p_type == "Short Futures":
            item_pnl = (stk - spot_price) * mult_qty - comm
        elif p_type == "Long Call Option":
            item_pnl = (np.maximum(0, spot_price - stk) - prem) * mult_qty - comm
        elif p_type == "Short Call Option":
            item_pnl = (prem - np.maximum(0, spot_price - stk)) * mult_qty - comm
        elif p_type == "Long Put Option":
            item_pnl = (np.maximum(0, stk - spot_price) - prem) * mult_qty - comm
        elif p_type == "Short Put Option":
            item_pnl = (prem - np.maximum(0, stk - spot_price)) * mult_qty - comm
        else:
            item_pnl = 0

        summary_list.append({
            "พอร์ต": p_src, "ซีรีส์": ser, "Strike": stk, "ประเภท": p_type,
            "สถานะ": status, "สัญญา": qty * sign_m, "P&L (บาท)": round(item_pnl, 2)
        })
    
    summary_df = pd.DataFrame(summary_list)
    if not summary_df.empty:
        total_pnl_sum = summary_df["P&L (บาท)"].sum()
        total_sum_row = pd.DataFrame([{
            "พอร์ต": "รวมทั้งหมด", "ซีรีส์": "-", "Strike": "-", "ประเภท": "-",
            "สถานะ": "-", "สัญญา": summary_df["สัญญา"].sum(), "P&L (บาท)": round(total_pnl_sum, 2)
        }])
        summary_df = pd.concat([summary_df, total_sum_row], ignore_index=True)
    st.dataframe(summary_df, use_container_width=True)
else:
    st.info("ไม่มีสัญญาในตลาดนี้")

st.markdown("<br>", unsafe_allow_html=True)

# --- สมุดบันทึกประจำพอร์ต ---
st.subheader(f"📝 สมุดบันทึกพอร์ต: `{active_portfolio}`")
with st.form("journal_form"):
    j_title = st.text_input("หัวข้อ", "บันทึกประจำวัน")
    j_notes = st.text_area("รายละเอียด / แผนการเทรด")
    if st.form_submit_button("💾 บันทึกสมุดบันทึก"):
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
# กราฟที่ 1: Net Expiry Payoff
# ==========================================
st.subheader(f"📈 1. Payoff รวมวันหมดอายุ ({selected_market})")
payoff_mode = st.selectbox("หน่วยแสดงผล", ["บาทรวม (THB)", "จุด (Points)"])

price_range = np.linspace(spot_price * 0.85, spot_price * 1.15, 300)
total_payoff = np.zeros_like(price_range)
total_greeks = {"Delta": 0.0, "Gamma": 0.0, "Theta": 0.0, "Vega": 0.0}

today = date.today()
fig = go.Figure()

if not market_trades.empty:
    for idx, row in market_trades.iterrows():
        p_type, stk, prem, qty, status = row["Type"], float(row["Strike"]), float(row["Premium"]), int(row["Contracts"]), row["Status"]
        sign_m = 1 if status == "Open" else -1
        comm_val = (float(row["Commission"]) * qty) if payoff_mode == "บาทรวม (THB)" else ((float(row["Commission"]) * qty) / contract_multiplier)
        
        try:
            T = max((datetime.strptime(str(row["ExpiryDate"]), "%Y-%m-%d").date() - today).days, 0) / 365.0
        except:
            T = 30 / 365.0
            
        mult_qty = (qty * contract_multiplier if payoff_mode == "บาทรวม (THB)" else qty) * sign_m
        payoff = np.zeros_like(price_range)
        
        if p_type == "Long Futures":
            payoff = (price_range - stk) * mult_qty - comm_val
        elif p_type == "Short Futures":
            payoff = (stk - price_range) * mult_qty - comm_val
        elif p_type == "Long Call Option":
            payoff = (np.maximum(0, price_range - stk) - prem) * mult_qty - comm_val
        elif p_type == "Short Call Option":
            payoff = (prem - np.maximum(0, price_range - stk)) * mult_qty - comm_val
        elif p_type == "Long Put Option":
            payoff = (np.maximum(0, stk - price_range) - prem) * mult_qty - comm_val
        elif p_type == "Short Put Option":
            payoff = (prem - np.maximum(0, stk - price_range)) * mult_qty - comm_val
            
        total_payoff += payoff
        
        if status == "Open":
            opt_flag = "Call" if "Call" in p_type else ("Put" if "Put" in p_type else None)
            if opt_flag:
                g = bs_greeks(spot_price, stk, T, risk_free_rate, volatility_input, opt_flag)
                dir_s = 1 if "Long" in p_type else -1
                total_greeks["Delta"] += g["Delta"] * qty * contract_multiplier * dir_s
                total_greeks["Gamma"] += g["Gamma"] * qty * contract_multiplier * dir_s
                total_greeks["Theta"] += g["Theta"] * qty * dir_s
                total_greeks["Vega"] += g["Vega"] * qty * contract_multiplier * dir_s
            else:
                total_greeks["Delta"] += 1.0 * qty * contract_multiplier * (1 if "Long" in p_type else -1)

        fig.add_trace(go.Scatter(
            x=price_range, y=payoff, mode='lines', name=f"{row['Strategy']} ({p_type})",
            line=dict(dash='dash', width=1), opacity=0.4,
            hovertemplate=f"<b>{row['Strategy']}</b><br>Price: %{{x:.2f}}<br>P&L: %{{y:,.2f}}<extra></extra>"
        ))

# Break-even
be_points = []
for j in range(len(price_range) - 1):
    if total_payoff[j] * total_payoff[j+1] < 0:
        x1, x2 = price_range[j], price_range[j+1]
        y1, y2 = total_payoff[j], total_payoff[j+1]
        if y2 - y1 != 0:
            be_points.append(x1 - y1 * (x2 - x1) / (y2 - y1))

max_pnl_val, price_at_max = total_payoff[np.argmax(total_payoff)], price_range[np.argmax(total_payoff)]
min_pnl_val, price_at_min = total_payoff[np.argmin(total_payoff)], price_range[np.argmin(total_payoff)]
unit_label = "บาท" if payoff_mode == "บาทรวม (THB)" else "จุด"

# --- สรุปสถิติความเสี่ยง ---
st.markdown("#### 🎯 สรุปความเสี่ยง (Max / Min & Boundaries)")
sc1, sc2, sc3, sc4 = st.columns(4)

with sc1:
    st.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 8px; border-radius: 6px; border: 1px solid #e9ecef;">
        <div style="font-size: 11px; color: #6c757d;">🟢 กำไรสูงสุด</div>
        <div style="font-size: 14px; font-weight: bold; color: #28a745;">{max_pnl_val:,.2f} {unit_label}</div>
        <div style="font-size: 10px;">ที่ราคา: {price_at_max:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with sc2:
    st.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 8px; border-radius: 6px; border: 1px solid #e9ecef;">
        <div style="font-size: 11px; color: #6c757d;">🔴 ขาดทุนสูงสุด</div>
        <div style="font-size: 14px; font-weight: bold; color: #dc3545;">{min_pnl_val:,.2f} {unit_label}</div>
        <div style="font-size: 10px;">ที่ราคา: {price_at_min:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with sc3:
    st.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 8px; border-radius: 6px; border: 1px solid #e9ecef;">
        <div style="font-size: 11px; color: #6c757d;">📉 ฝั่งซ้ายสุด</div>
        <div style="font-size: 14px; font-weight: bold; color: #343a40;">{total_payoff[0]:,.2f} {unit_label}</div>
        <div style="font-size: 10px;">ที่ราคา: {price_range[0]:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with sc4:
    st.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 8px; border-radius: 6px; border: 1px solid #e9ecef;">
        <div style="font-size: 11px; color: #6c757d;">📈 ฝั่งขวาสุด</div>
        <div style="font-size: 14px; font-weight: bold; color: #343a40;">{total_payoff[-1]:,.2f} {unit_label}</div>
        <div style="font-size: 10px;">ที่ราคา: {price_range[-1]:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

be_str = ", ".join([f"`{bp:,.2f}`" for bp in be_points]) if be_points else "ไม่มีจุดคุ้มทุนในช่วงนี้"
st.markdown(f"<br>💡 **จุดคุ้มทุน (Break-even):** {be_str}", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

fig.add_trace(go.Scatter(
    x=price_range, y=total_payoff, mode='lines', name='Net Portfolio Payoff',
    line=dict(color='blue', width=3), fill='tozeroy', fillcolor='rgba(0, 123, 255, 0.1)'
))

fig.add_trace(go.Scatter(
    x=price_range, y=np.where(total_payoff < 0, total_payoff, 0), mode='lines',
    line=dict(width=0), fill='tozeroy', fillcolor='rgba(220, 53, 69, 0.15)', showlegend=False, hoverinfo='skip'
))

if be_points:
    fig.add_trace(go.Scatter(
        x=be_points, y=[0]*len(be_points), mode='markers+text', name='Break-even',
        marker=dict(color='orange', size=10, symbol='diamond'),
        text=[f"BE: {bp:.2f}" for bp in be_points], textposition="bottom center"
    ))

fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
fig.add_vline(x=spot_price, line_dash="dot", line_color="red", line_width=2)

spot_pnl_expiry = np.interp(spot_price, price_range, total_payoff)
pnl_status = f"+{spot_pnl_expiry:,.2f}" if spot_pnl_expiry >= 0 else f"{spot_pnl_expiry:,.2f}"

fig.add_annotation(
    x=spot_price, y=1.03, yref="paper", text=f"Spot: {spot_price:,.2f} ({pnl_status})",
    showarrow=False, font=dict(color="red", size=13), xanchor="center", yanchor="bottom"
)

fig.update_layout(
    title=dict(text=f"Net Expiry Payoff ({payoff_mode})", y=0.98, x=0.0, xanchor='left', yanchor='top'),
    xaxis_title="Underlying Price", yaxis_title=f"P&L ({payoff_mode})",
    hovermode="x unified", template="plotly_white", height=560,
    margin=dict(t=80, b=50, l=50, r=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# ==========================================
# กราฟที่ 2 & 3: Time Decay Simulation & Greeks
# ==========================================
st.subheader("⏱️️ 2. จำลอง Payoff รายวัน")
sim_date = st.date_input("เลือกวันที่จำลอง", value=date.today())
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
                sign_m = 1 if status == "Open" else -1
                mult_qty = (qty * contract_multiplier if payoff_mode == "บาทรวม (THB)" else qty) * sign_m
                comm_val = (float(row["Commission"]) * qty) if payoff_mode == "บาทรวม (THB)" else ((float(row["Commission"]) * qty) / contract_multiplier)
                
                try:
                    T_sim = max((datetime.strptime(str(row["ExpiryDate"]), "%Y-%m-%d").date() - t_d).days, 0) / 365.0
                except:
                    T_sim = 30 / 365.0
                    
                if p_type == "Long Futures":
                    val = (price_range - stk) * mult_qty - comm_val
                elif p_type == "Short Futures":
                    val = (stk - price_range) * mult_qty - comm_val
                elif "Call" in p_type:
                    opt_v = np.array([bs_option_price(p, stk, T_sim, risk_free_rate, volatility_input, "Call") for p in price_range])
                    val = (opt_v - prem) * mult_qty - comm_val if "Long" in p_type else (prem - opt_v) * mult_qty - comm_val
                elif "Put" in p_type:
                    opt_v = np.array([bs_option_price(p, stk, T_sim, risk_free_rate, volatility_input, "Put") for p in price_range])
                    val = (opt_v - prem) * mult_qty - comm_val if "Long" in p_type else (prem - opt_v) * mult_qty - comm_val
                else:
                    val = -comm_val
                day_val += val
                
        if has_t:
            is_sel = (t_str == sim_date_str)
            fig_daily_single.add_trace(go.Scatter(
                x=price_range, y=day_val, mode='lines', name=f"Date {t_str}" + (" (Selected)" if is_sel else ""),
                line=dict(color=colors[i % len(colors)], width=3 if is_sel else 1.5, dash='solid' if is_sel else 'dash')
            ))

    for i, t_str in enumerate(unique_dates):
        if t_str > sim_date_str:
            continue
        t_d = datetime.strptime(t_str, "%Y-%m-%d").date() if t_str else date.today()
        cum_val = np.zeros_like(price_range)
        cum_trades = market_trades[market_trades["TradeDate"].astype(str) <= t_str]
        
        if not cum_trades.empty:
            for _, row in cum_trades.iterrows():
                p_type, stk, prem, qty, status = row["Type"], float(row["Strike"]), float(row["Premium"]), int(row["Contracts"]), row["Status"]
                sign_m = 1 if status == "Open" else -1
                mult_qty = (qty * contract_multiplier if payoff_mode == "บาทรวม (THB)" else qty) * sign_m
                comm_val = (float(row["Commission"]) * qty) if payoff_mode == "บาทรวม (THB)" else ((float(row["Commission"]) * qty) / contract_multiplier)
                
                try:
                    T_sim = max((datetime.strptime(str(row["ExpiryDate"]), "%Y-%m-%d").date() - t_d).days, 0) / 365.0
                except:
                    T_sim = 30 / 365.0
                    
                if p_type == "Long Futures":
                    val = (price_range - stk) * mult_qty - comm_val
                elif p_type == "Short Futures":
                    val = (stk - price_range) * mult_qty - comm_val
                elif "Call" in p_type:
                    opt_v = np.array([bs_option_price(p, stk, T_sim, risk_free_rate, volatility_input, "Call") for p in price_range])
                    val = (opt_v - prem) * mult_qty - comm_val if "Long" in p_type else (prem - opt_v) * mult_qty - comm_val
                elif "Put" in p_type:
                    opt_v = np.array([bs_option_price(p, stk, T_sim, risk_free_rate, volatility_input, "Put") for p in price_range])
                    val = (opt_v - prem) * mult_qty - comm_val if "Long" in p_type else (prem - opt_v) * mult_qty - comm_val
                else:
                    val = -comm_val
                cum_val += val
                
        is_sel = (t_str == sim_date_str)
        fig_daily_cumulative.add_trace(go.Scatter(
            x=price_range, y=cum_val, mode='lines', name=f"Cum. to {t_str}" + (" (Selected)" if is_sel else ""),
            line=dict(color=colors[i % len(colors)], width=3.5 if is_sel else 1.5, dash='solid' if is_sel else 'dash')
        ))

fig_daily_single.add_hline(y=0, line_color="black", line_width=1)
fig_daily_single.add_vline(x=spot_price, line_dash="dot", line_color="red", line_width=2)
fig_daily_single.update_layout(title="Payoff เฉพาะวันที่เทรด", template="plotly_white", height=400, legend=dict(orientation="h", y=1.1))

fig_daily_cumulative.add_hline(y=0, line_color="black", line_width=1)
fig_daily_cumulative.add_vline(x=spot_price, line_dash="dot", line_color="red", line_width=2)
fig_daily_cumulative.update_layout(title="Payoff สะสมถึงวันที่เลือก", template="plotly_white", height=400, legend=dict(orientation="h", y=1.1))

st.plotly_chart(fig_daily_single, use_container_width=True)
st.plotly_chart(fig_daily_cumulative, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"### 📋 รายการเทรด ณ หรือก่อนวันที่ `{sim_date_str}`")
if not market_trades.empty:
    filt_sim = market_trades[market_trades["TradeDate"].astype(str) <= sim_date_str]
    if not filt_sim.empty:
        st.dataframe(filt_sim[["ID", "Strategy", "Series", "Type", "Status", "Strike", "Premium", "Contracts", "Commission", "TradeDate"]], use_container_width=True)
    else:
        st.info("ไม่มีรายการ")

st.markdown("<br><br>", unsafe_allow_html=True)

# ==========================================
# สรุป Greeks
# ==========================================
st.subheader("📐 สรุป Greeks รวมพอร์ต")
g1, g2, g3, g4 = st.columns(4)
g1.metric("Delta", f"{total_greeks['Delta']:,.2f}")
g2.metric("Gamma", f"{total_greeks['Gamma']:,.4f}")
g3.metric("Theta (Daily)", f"{total_greeks['Theta']:,.2f} ฿")
g4.metric("Vega", f"{total_greeks['Vega']:,.2f}")
