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

# --- ฟังก์ชันจัดการไฟล์พอร์ตหลายพอร์ต ---
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

def load_trades_from_file(file_name):
    expected_cols = [
        "ID", "Market", "Strategy", "Series", "Type", "Status", 
        "Strike", "Premium", "Contracts", "Commission", "TradeDate", "ExpiryDate", "EntrySpot"
    ]
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
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

# --- ดึงข้อมูลราคาและคำนวณ Volatility อัตโนมัติจาก Yahoo Finance ---
@st.cache_data(ttl=10)
def get_market_data(market_type):
    try:
        if market_type == "TFEX USD/THB":
            ticker_str = "THB=X"
            default_p = 35.0
            default_v = 0.08
        else:  # TFEX SET50
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
            daily_vol = hist['Log_Return'].std()
            calc_vol = float(daily_vol * np.sqrt(252))
            if np.isnan(calc_vol) or calc_vol <= 0:
                calc_vol = default_v
        else:
            calc_vol = default_v

        if not price:
            if not hist.empty:
                price = float(hist['Close'].iloc[-1])
            else:
                price = default_p
                
        update_time = datetime.now(BANGKOK_TZ).strftime("%Y-%m-%d %H:%M:%S (ICT)")
        return price, calc_vol, update_time
    except:
        default_p = 35.0 if market_type == "TFEX USD/THB" else 950.0
        default_v = 0.08 if market_type == "TFEX USD/THB" else 0.18
        update_time = datetime.now(BANGKOK_TZ).strftime("%Y-%m-%d %H:%M:%S (ICT)")
        return default_p, default_v, update_time

# --- ฟังก์ชัน Black-Scholes สำหรับ Option Pricing ---
def bs_option_price(S, K, T, r, sigma, option_type):
    if T <= 0:
        if option_type == "Call":
            return np.maximum(0, S - K)
        else:
            return np.maximum(0, K - S)
    
    vol = sigma if sigma > 0 else 0.15
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    
    if option_type == "Call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(d1)
    return price

# --- ฟังก์ชันคำนวณ Greeks ---
def bs_greeks(S, K, T, r, sigma, option_type):
    if T <= 0:
        if option_type == "Call":
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
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
st.markdown("ระบบวิเคราะห์ Payoff Chart รองรับ Multi-Portfolio, ซีรีส์มาตรฐาน, บันทึกวันที่ซื้อขาย และเหมาะสำหรับมือถือ")

# --- แถบ Sidebar จัดการพอร์ตและตลาด ---
st.sidebar.header("📁 จัดการพอร์ตเทรด")
available_portfolios = get_available_portfolios()

active_portfolio = st.sidebar.selectbox("เลือกพอร์ตหลัก", available_portfolios)

new_port_name = st.sidebar.text_input("ชื่อพอร์ตใหม่ (เช่น portfolio_4.csv)")
if st.sidebar.button("➕ สร้างพอร์ตใหม่"):
    if new_port_name:
        if not new_port_name.endswith(".csv"):
            new_port_name += ".csv"
        if not os.path.exists(new_port_name):
            empty_df = pd.DataFrame(columns=[
                "ID", "Market", "Strategy", "Series", "Type", "Status", 
                "Strike", "Premium", "Contracts", "Commission", "TradeDate", "ExpiryDate", "EntrySpot"
            ])
            empty_df.to_csv(new_port_name, index=False)
            st.sidebar.success(f"สร้างพอร์ต {new_port_name} สำเร็จ!")
            st.rerun()
        else:
            st.sidebar.warning("ชื่อพอร์ตนี้มีอยู่แล้ว")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ ตั้งค่าตลาด")
selected_market = st.sidebar.selectbox("เลือกตลาด TFEX", ["TFEX USD/THB", "TFEX SET50"])

current_spot, auto_volatility, last_update_time = get_market_data(selected_market)

# --- แถบแสดงข้อมูลราคาอ้างอิง ---
col_head1, col_head2 = st.columns([1, 2])
with col_head1:
    if st.button("🔄 รีเฟรชตลาด"):
        st.cache_data.clear()
        st.rerun()
with col_head2:
    st.metric(label=f"🟢 Spot ({selected_market})", value=f"{current_spot:,.4f}")

st.markdown(f"*อัปเดตล่าสุด:* `{last_update_time}`")
st.markdown("---")

spot_price = st.sidebar.number_input(f"ราคาอ้างอิง ({selected_market})", value=float(current_spot), format="%.4f")
volatility_input = st.sidebar.slider(
    "Volatility (%) [ตลาด]", 
    1.0, 100.0, 
    float(auto_volatility * 100)
) / 100.0

risk_free_rate = 0.025
contract_multiplier = 1000 if selected_market == "TFEX USD/THB" else 200

trades_df = load_trades_from_file(active_portfolio)

# --- สร้างรายการ Strike และ Series อัตโนมัติ ---
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

generated_series_options = []
for yr in [short_year, next_short_year]:
    for m in month_codes:
        generated_series_options.append(f"{prefix_code}{m}{yr}")

# --- ฟอร์มเพิ่มรายการเทรด ---
st.sidebar.subheader(f"➕ เพิ่มสัญญาใหม่ ({active_portfolio})")
with st.sidebar.form("trade_form"):
    strategy_name = st.text_input("ชื่อกลยุทธ์ / Note", "Strategy #1")
    
    series_mode = st.radio("เลือกซีรีส์", ["Dropdown มาตรฐาน", "พิมพ์เอง"])
    if series_mode == "Dropdown มาตรฐาน":
        default_series_idx = 6 if "U" + short_year in generated_series_options else 0
        series_name = st.selectbox("ซีรีส์ TFEX", options=generated_series_options, index=default_series_idx)
    else:
        series_name = st.text_input("รหัสซีรีส์", value=f"{prefix_code}U{short_year}")
    
    position_status = st.selectbox("สถานะคำสั่ง", ["Open", "Close"])
    position_type = st.selectbox("ประเภทสัญญา", [
        "Long Futures", "Short Futures", 
        "Long Call Option", "Short Call Option", 
        "Long Put Option", "Short Put Option"
    ])
    
    strike_mode = st.radio("ราคาใช้สิทธิ (Strike)", ["Dropdown อัตโนมัติ", "พิมพ์เอง"])
    if strike_mode == "Dropdown อัตโนมัติ":
        if selected_market == "TFEX USD/THB":
            closest_val = round(spot_price * 4) / 4.0
            closest_str = f"{closest_val:.2f}"
            default_idx = base_strikes.index(closest_str) if closest_str in base_strikes else len(base_strikes) // 2
        else:
            closest_val = int(round(spot_price / 10.0) * 10.0)
            default_idx = base_strikes.index(closest_val) if closest_val in base_strikes else len(base_strikes) // 2
        selected_strike_item = st.selectbox("Strike Price", options=base_strikes, index=default_idx)
        strike = float(selected_strike_item)
    else:
        strike = st.number_input("Strike Price เอง", value=float(round(spot_price, 2)), format="%.2f")

    default_prem = 0.25 if selected_market == "TFEX USD/THB" else 15.0
    premium = st.number_input("ราคาพรีเมี่ยม / ต้นทุน", value=float(default_prem), format="%.2f")
    contracts = st.number_input("จำนวนสัญญา (Contracts)", value=1, min_value=1, step=1)
    
    default_comm = 30.0 if selected_market == "TFEX USD/THB" else 60.0
    commission = st.number_input("ค่าคอมฯ รวมต่อสัญญา (บาท)", value=float(default_comm), format="%.2f")
    
    trade_date = st.date_input("วันที่ซื้อขาย (Trade Date)", value=date.today())
    expiry_date = st.date_input("วันหมดอายุ (Expiry Date)", value=date.today() + timedelta(days=30))
    
    submitted = st.form_submit_button("บันทึกเพิ่มเข้าพอร์ต")
    if submitted:
        new_id = int(trades_df["ID"].max() + 1) if not trades_df.empty and "ID" in trades_df.columns else 1
        new_row = pd.DataFrame([{
            "ID": new_id,
            "Market": selected_market,
            "Strategy": strategy_name,
            "Series": series_name,
            "Type": position_type,
            "Status": position_status,
            "Strike": strike,
            "Premium": premium,
            "Contracts": contracts,
            "Commission": commission,
            "TradeDate": str(trade_date),
            "ExpiryDate": str(expiry_date),
            "EntrySpot": spot_price
        }])
        trades_df = pd.concat([trades_df, new_row], ignore_index=True)
        save_trades_to_file(trades_df, active_portfolio)
        st.sidebar.success("บันทึกสำเร็จ!")
        st.rerun()

# ==========================================
# ตารางจัดการและแก้ไขข้อมูล
# ==========================================
st.subheader(f"📋 ตารางข้อมูลพอร์ต: `{active_portfolio}`")
active_df_to_use = trades_df.copy()

if not trades_df.empty:
    edited_df = st.data_editor(trades_df, num_rows="dynamic", use_container_width=True, key="trade_editor")
    active_df_to_use = edited_df
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("💾 บันทึกการแก้ไข"):
            save_trades_to_file(edited_df, active_portfolio)
            st.success("บันทึกสำเร็จ!")
            st.rerun()
            
    st.markdown("---")
    st.subheader("🗑️ ลบรายการเทรด")
    del_col1, del_col2 = st.columns(2)
    with del_col1:
        trade_ids_to_delete = st.selectbox("เลือก ID ที่ต้องการลบ", options=edited_df["ID"].tolist() if not edited_df.empty else [])
        confirm_delete = st.checkbox("⚠️ ยืนยันการลบ", value=False)
    with del_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ ลบรายการ"):
            if confirm_delete:
                updated_trades = edited_df[edited_df["ID"] != trade_ids_to_delete].reset_index(drop=True)
                save_trades_to_file(updated_trades, active_portfolio)
                st.success(f"ลบ ID {trade_ids_to_delete} เรียบร้อย")
                st.rerun()
            else:
                st.warning("โปรดติ๊กเครื่องหมายยืนยันการลบ")
else:
    st.info(f"พอร์ต `{active_portfolio}` ยังว่างอยู่")

# --- ฟังก์ชันรวมพอร์ต (Combine Portfolios) ---
st.markdown("---")
st.subheader("🔀 รวมพอร์ตเพื่อวิเคราะห์ Payoff ร่วมกัน")
selected_portfolios_for_merge = st.multiselect(
    "เลือกพอร์ตที่ต้องการนำมารวมกัน",
    options=available_portfolios,
    default=[active_portfolio]
)

combined_trades_df = pd.DataFrame()
if selected_portfolios_for_merge:
    dfs = []
    for p_file in selected_portfolios_for_merge:
        if p_file == active_portfolio:
            p_df = active_df_to_use.copy()
        else:
            p_df = load_trades_from_file(p_file)
        if not p_df.empty:
            p_df['Source_Portfolio'] = p_file
            dfs.append(p_df)
    if dfs:
        combined_trades_df = pd.concat(dfs, ignore_index=True)

market_trades = pd.DataFrame()
if not combined_trades_df.empty and "Market" in combined_trades_df.columns:
    market_trades = combined_trades_df[combined_trades_df["Market"] == selected_market]

# ==========================================
# ตารางสรุป P&L
# ==========================================
st.subheader("📊 ตารางสรุปสถานะและ P&L ปัจจุบัน")
if not market_trades.empty:
    summary_list = []
    grouped = market_trades.groupby(["Source_Portfolio", "Series", "Strike", "Type"])
    for key, group in grouped:
        p_src, ser, stk, p_type = key
        open_qty = group[group["Status"] == "Open"]["Contracts"].sum()
        close_qty = group[group["Status"] == "Close"]["Contracts"].sum()
        net_qty = open_qty - close_qty
        
        item_pnl = 0
        mult_qty = contract_multiplier
        for _, r_row in group.iterrows():
            r_stk = float(r_row["Strike"])
            r_prem = float(r_row["Premium"])
            r_qty = int(r_row["Contracts"])
            r_comm = float(r_row["Commission"]) * r_qty
            r_stat = r_row["Status"]
            dir_factor = 1 if r_stat == "Open" else -1
            
            if r_row["Type"] == "Long Futures":
                item_pnl += (spot_price - r_stk) * r_qty * mult_qty * dir_factor - r_comm
            elif r_row["Type"] == "Short Futures":
                item_pnl += (r_stk - spot_price) * r_qty * mult_qty * dir_factor - r_comm
            elif "Call" in r_row["Type"]:
                curr_opt_val = bs_option_price(spot_price, r_stk, 30/365.0, risk_free_rate, volatility_input, "Call")
                if "Long" in r_row["Type"]:
                    item_pnl += (curr_opt_val - r_prem) * r_qty * mult_qty * dir_factor - r_comm
                else:
                    item_pnl += (r_prem - curr_opt_val) * r_qty * mult_qty * dir_factor - r_comm
            elif "Put" in r_row["Type"]:
                curr_opt_val = bs_option_price(spot_price, r_stk, 30/365.0, risk_free_rate, volatility_input, "Put")
                if "Long" in r_row["Type"]:
                    item_pnl += (curr_opt_val - r_prem) * r_qty * mult_qty * dir_factor - r_comm
                else:
                    item_pnl += (r_prem - curr_opt_val) * r_qty * mult_qty * dir_factor - r_comm

        summary_list.append({
            "Portfolio": p_src,
            "Series": ser,
            "Strike": stk,
            "Type": p_type,
            "Open Qty": open_qty,
            "Net Qty": net_qty,
            "Current P&L (THB)": round(item_pnl, 2)
        })
    st.dataframe(pd.DataFrame(summary_list), use_container_width=True)
else:
    st.info("ไม่มีข้อมูลสัญญาในตลาดนี้")

# ==========================================
# กราฟที่ 1: Payoff ณ วันหมดอายุ (พร้อม Spot ป้ายด้านบน และจุดคุ้มทุน BEP)
# ==========================================
st.subheader(f"📈 1. Payoff รวม ณ วันหมดอายุ ({selected_market})")
payoff_mode = st.selectbox("หน่วยแสดงผล", ["บาทรวม (THB)", "จุด (Points)"])

price_range = np.linspace(spot_price * 0.85, spot_price * 1.15, 300)
total_payoff = np.zeros_like(price_range)
total_greeks = {"Delta": 0.0, "Gamma": 0.0, "Theta": 0.0, "Vega": 0.0}

today = date.today()
fig = go.Figure()

if not market_trades.empty:
    for idx, row in market_trades.iterrows():
        p_type = row["Type"]
        stk = float(row["Strike"])
        prem = float(row["Premium"])
        qty = int(row["Contracts"])
        status = row["Status"]
        sign_multiplier = 1 if status == "Open" else -1
        
        try:
            exp_d = datetime.strptime(str(row["ExpiryDate"]), "%Y-%m-%d").date()
            days_to_expiry = (exp_d - today).days
            T = max(days_to_expiry, 0) / 365.0
        except:
            T = 30 / 365.0
            
        mult_qty = (qty * contract_multiplier if payoff_mode == "บาทรวม (THB)" else qty) * sign_multiplier
        payoff = np.zeros_like(price_range)
        
        if p_type == "Long Futures":
            payoff = (price_range - stk) * mult_qty
        elif p_type == "Short Futures":
            payoff = (stk - price_range) * mult_qty
        elif p_type == "Long Call Option":
            payoff = (np.maximum(0, price_range - stk) - prem) * mult_qty
        elif p_type == "Short Call Option":
            payoff = (prem - np.maximum(0, price_range - stk)) * mult_qty
        elif p_type == "Long Put Option":
            payoff = (np.maximum(0, stk - price_range) - prem) * mult_qty
        elif p_type == "Short Put Option":
            payoff = (prem - np.maximum(0, stk - price_range)) * mult_qty
            
        total_payoff += payoff
        
        if status == "Open":
            opt_flag = "Call" if "Call" in p_type else ("Put" if "Put" in p_type else None)
            if opt_flag:
                g = bs_greeks(spot_price, stk, T, risk_free_rate, volatility_input, opt_flag)
                dir_sign = 1 if "Long" in p_type else -1
                total_greeks["Delta"] += g["Delta"] * qty * contract_multiplier * dir_sign
                total_greeks["Gamma"] += g["Gamma"] * qty * contract_multiplier * dir_sign
                total_greeks["Theta"] += g["Theta"] * qty * dir_sign
                total_greeks["Vega"] += g["Vega"] * qty * contract_multiplier * dir_sign
            else:
                dir_sign = 1 if "Long" in p_type else -1
                total_greeks["Delta"] += 1.0 * qty * contract_multiplier * dir_sign

        fig.add_trace(go.Scatter(
            x=price_range, y=payoff,
            mode='lines',
            name=f"{row['Strategy']} ({p_type})",
            line=dict(dash='dash', width=1),
            opacity=0.4,
            hovertemplate=f"<b>{row['Strategy']}</b><br>Price: %{{x:.2f}}<br>P&L: %{{y:,.2f}}<extra></extra>"
        ))

# เพิ่มกราฟ Net Payoff (กำไรฟ้าอ่อน)
fig.add_trace(go.Scatter(
    x=price_range, y=total_payoff,
    mode='lines',
    name='Net Portfolio Payoff',
    line=dict(color='blue', width=3),
    fill='tozeroy',
    fillcolor='rgba(0, 123, 255, 0.1)',
    hovertemplate="<b>Net Portfolio</b><br>Price: %{x:.2f}<br>Total P&L: %{y:,.2f}<extra></extra>"
))

# สร้างโซนสีแดงใต้เส้น 0 (ขาดทุน)
fig.add_trace(go.Scatter(
    x=price_range,
    y=np.where(total_payoff < 0, total_payoff, 0),
    mode='lines',
    line=dict(width=0),
    fill='tozeroy',
    fillcolor='rgba(220, 53, 69, 0.15)',
    showlegend=False,
    hoverinfo='skip'
))

# --- คำนวณหาจุดคุ้มทุน (Break-even Points - BEP) ---
sign_changes = np.where(np.diff(np.signbit(total_payoff)))[0]
bep_prices = []
for idx_sc in sign_changes:
    x1, x2 = price_range[idx_sc], price_range[idx_sc + 1]
    y1, y2 = total_payoff[idx_sc], total_payoff[idx_sc + 1]
    if y2 - y1 != 0:
        bep = x1 - y1 * (x2 - x1) / (y2 - y1)
        bep_prices.append(bep)

if bep_prices:
    fig.add_trace(go.Scatter(
        x=bep_prices,
        y=[0] * len(bep_prices),
        mode='markers+text',
        name='Break-even (BEP)',
        marker=dict(color='orange', size=10, symbol='diamond'),
        text=[f"BEP: {bep:.2f}" for bep in bep_prices],
        textposition="top center",
        hoverinfo='text'
    ))

fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
fig.add_vline(x=spot_price, line_dash="dot", line_color="red", line_width=2)

# เพิ่ม Annotations แสดงราคาอ้างอิงไว้ที่ด้านบนสุดของเส้น Spot (ไม่ทับตัวกราฟหลัก)
fig.add_annotation(
    x=spot_price,
    y=1.0,
    yref="paper",
    text=f"Spot: {spot_price:,.2f}",
    showarrow=False,
    font=dict(color="red", size=12, family="sans-serif"),
    bgcolor="rgba(255, 255, 255, 0.8)",
    bordercolor="red",
    borderwidth=1,
    borderpad=4
)

fig.update_layout(
    title=f"Net Expiry Payoff ({payoff_mode})",
    xaxis_title="Underlying Price",
    yaxis_title=f"P&L ({payoff_mode})",
    hovermode="x unified",
    template="plotly_white",
    height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# กราฟที่ 2 & 3: จำลองรายวัน (พร้อม Spot ป้ายด้านบน) และแสดง Greeks
# ==========================================
st.subheader("⏱️ 2. จำลองกราฟ Payoff รายวัน (Time Decay Simulation)")
sim_date = st.date_input("เลือกวันที่ต้องการจำลองสถานะ", value=date.today())
sim_date_str = str(sim_date)

fig_daily_single = go.Figure()
fig_daily_cumulative = go.Figure()

if not market_trades.empty:
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    unique_trade_dates = sorted(list(set(market_trades["TradeDate"].astype(str))))
    
    if sim_date_str not in unique_trade_dates:
        unique_trade_dates.append(sim_date_str)
        unique_trade_dates = sorted(unique_trade_dates)

    # กราฟที่ 2.1: เฉพาะสัญญาที่เทรดในวันนั้นๆ (Single Day Trades Only)
    for i, t_date_str in enumerate(unique_trade_dates):
        try:
            t_d = datetime.strptime(t_date_str, "%Y-%m-%d").date()
        except:
            t_d = date.today()
            
        day_specific_value = np.zeros_like(price_range)
        has_trade_on_day = False
        
        for idx, row in market_trades.iterrows():
            if str(row["TradeDate"]) == t_date_str:
                has_trade_on_day = True
                p_type = row["Type"]
                stk = float(row["Strike"])
                prem = float(row["Premium"])
                qty = int(row["Contracts"])
                status = row["Status"]
                sign_multiplier = 1 if status == "Open" else -1
                mult_qty = (qty * contract_multiplier if payoff_mode == "บาทรวม (THB)" else qty) * sign_multiplier
                
                try:
                    exp_d = datetime.strptime(str(row["ExpiryDate"]), "%Y-%m-%d").date()
                    rem_days = (exp_d - t_d).days
                    T_sim = max(rem_days, 0) / 365.0
                except:
                    T_sim = 30 / 365.0
                    
                if p_type == "Long Futures":
                    val = (price_range - stk) * mult_qty
                elif p_type == "Short Futures":
                    val = (stk - price_range) * mult_qty
                elif "Call" in p_type:
                    opt_values = np.array([bs_option_price(p, stk, T_sim, risk_free_rate, volatility_input, "Call") for p in price_range])
                    val = (opt_values - prem) * mult_qty if "Long" in p_type else (prem - opt_values) * mult_qty
                elif "Put" in p_type:
                    opt_values = np.array([bs_option_price(p, stk, T_sim, risk_free_rate, volatility_input, "Put") for p in price_range])
                    val = (opt_values - prem) * mult_qty if "Long" in p_type else (prem - opt_values) * mult_qty
                else:
                    val = 0
                day_specific_value += val
                
        if has_trade_on_day:
            line_color = colors[i % len(colors)]
            is_selected = (t_date_str == sim_date_str)
            fig_daily_single.add_trace(go.Scatter(
                x=price_range, y=day_specific_value,
                mode='lines',
                name=f"Trades on {t_date_str}" + (" ⭐ (Selected)" if is_selected else ""),
                line=dict(color=line_color, width=3 if is_selected else 1.5, dash='solid' if is_selected else 'dash'),
                fill='tozeroy' if is_selected else None,
                fillcolor='rgba(0, 123, 255, 0.08)' if is_selected else None,
                hovertemplate=f"<b>Date: {t_date_str}</b><br>Price: %{{x:.2f}}<br>Value: %{{y:,.2f}}<extra></extra>"
            ))
            if is_selected:
                fig_daily_single.add_trace(go.Scatter(
                    x=price_range,
                    y=np.where(day_specific_value < 0, day_specific_value, 0),
                    mode='lines',
                    line=dict(width=0),
                    fill='tozeroy',
                    fillcolor='rgba(220, 53, 69, 0.15)',
                    showlegend=False,
                    hoverinfo='skip'
                ))

    # กราฟที่ 2.2: แบบสะสมยอดรวมถึงวันที่เลือก (Cumulative Portfolio Value up to Date)
    for i, t_date_str in enumerate(unique_trade_dates):
        if t_date_str > sim_date_str:
            continue
        try:
            t_d = datetime.strptime(t_date_str, "%Y-%m-%d").date()
        except:
            t_d = date.today()
            
        cumulative_portfolio_value = np.zeros_like(price_range)
        
        for idx, row in market_trades.iterrows():
            if str(row["TradeDate"]) <= t_date_str:
                p_type = row["Type"]
                stk = float(row["Strike"])
                prem = float(row["Premium"])
                qty = int(row["Contracts"])
                status = row["Status"]
                sign_multiplier = 1 if status == "Open" else -1
                mult_qty = (qty * contract_multiplier if payoff_mode == "บาทรวม (THB)" else qty) * sign_multiplier
                
                try:
                    exp_d = datetime.strptime(str(row["ExpiryDate"]), "%Y-%m-%d").date()
                    rem_days = (exp_d - t_d).days
                    T_sim = max(rem_days, 0) / 365.0
                except:
                    T_sim = 30 / 365.0
                    
                if p_type == "Long Futures":
                    val = (price_range - stk) * mult_qty
                elif p_type == "Short Futures":
                    val = (stk - price_range) * mult_qty
                elif "Call" in p_type:
                    opt_values = np.array([bs_option_price(p, stk, T_sim, risk_free_rate, volatility_input, "Call") for p in price_range])
                    val = (opt_values - prem) * mult_qty if "Long" in p_type else (prem - opt_values) * mult_qty
                elif "Put" in p_type:
                    opt_values = np.array([bs_option_price(p, stk, T_sim, risk_free_rate, volatility_input, "Put") for p in price_range])
                    val = (opt_values - prem) * mult_qty if "Long" in p_type else (prem - opt_values) * mult_qty
                else:
                    val = 0
                cumulative_portfolio_value += val
                
        line_color = colors[i % len(colors)]
        is_selected = (t_date_str == sim_date_str)
        fig_daily_cumulative.add_trace(go.Scatter(
            x=price_range, y=cumulative_portfolio_value,
            mode='lines',
            name=f"Cumulative up to {t_date_str}" + (" ⭐ (Selected)" if is_selected else ""),
            line=dict(color=line_color, width=3.5 if is_selected else 1.5, dash='solid' if is_selected else 'dash'),
            fill='tozeroy' if is_selected else None,
            fillcolor='rgba(0, 123, 255, 0.08)' if is_selected else None,
            hovertemplate=f"<b>Cumulative to {t_date_str}</b><br>Price: %{{x:.2f}}<br>Value: %{{y:,.2f}}<extra></extra>"
        ))
        if is_selected:
            fig_daily_cumulative.add_trace(go.Scatter(
                x=price_range,
                y=np.where(cumulative_portfolio_value < 0, cumulative_portfolio_value, 0),
                mode='lines',
                line=dict(width=0),
                fill='tozeroy',
                fillcolor='rgba(220, 53, 69, 0.15)',
                showlegend=False,
                hoverinfo='skip'
            ))

# เพิ่มเส้นอ้างอิงและป้าย Spot ด้านบนให้กับกราฟจำลองรายวันทั้งสอง
fig_daily_single.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
fig_daily_single.add_vline(x=spot_price, line_dash="dot", line_color="red", line_width=2)
fig_daily_single.add_annotation(
    x=spot_price, y=1.0, yref="paper", text=f"Spot: {spot_price:,.2f}",
    showarrow=False, font=dict(color="red", size=12), bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="red", borderwidth=1, borderpad=4
)
fig_daily_single.update_layout(
    title="กราฟเฉพาะวันที่เทรด (Trades on Specific Date)",
    xaxis_title="Underlying Price", yaxis_title=f"Value ({payoff_mode})",
    hovermode="x unified", template="plotly_white", height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

fig_daily_cumulative.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
fig_daily_cumulative.add_vline(x=spot_price, line_dash="dot", line_color="red", line_width=2)
fig_daily_cumulative.add_annotation(
    x=spot_price, y=1.0, yref="paper", text=f"Spot: {spot_price:,.2f}",
    showarrow=False, font=dict(color="red", size=12), bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="red", borderwidth=1, borderpad=4
)
fig_daily_cumulative.update_layout(
    title="กราฟสะสมยอดรวมถึงวันที่เลือก (Cumulative Portfolio Value)",
    xaxis_title="Underlying Price", yaxis_title=f"Value ({payoff_mode})",
    hovermode="x unified", template="plotly_white", height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_daily_single, use_container_width=True)
st.plotly_chart(fig_daily_cumulative, use_container_width=True)

# ==========================================
# สรุปค่า Greeks รวมพอร์ต
# ==========================================
st.subheader("📐 สรุปค่า Greeks รวมพอร์ต")
g1, g2, g3, g4 = st.columns(4)
g1.metric("Delta", f"{total_greeks['Delta']:,.2f}")
g2.metric("Gamma", f"{total_greeks['Gamma']:,.4f}")
g3.metric("Theta (Daily)", f"{total_greeks['Theta']:,.2f} THB")
g4.metric("Vega", f"{total_greeks['Vega']:,.2f}")
