import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import os
from datetime import datetime, date, timezone, timedelta
from scipy.stats import norm

st.set_page_config(page_title="TFEX Multi-Asset Payoff & Greeks Dashboard Pro", layout="wide")

TRADE_FILE = "trades_multi_asset_pro.csv"

# กำหนดโซนเวลา Bangkok (UTC+7)
BANGKOK_TZ = timezone(timedelta(hours=7))

# --- ฟังก์ชันจัดการข้อมูล (CRUD) ---
def load_trades():
    if os.path.exists(TRADE_FILE):
        df = pd.read_csv(TRADE_FILE)
        expected_cols = ["ID", "Market", "Strategy", "Series", "Type", "Status", "Strike", "Premium", "Contracts", "Commission", "TradeDate", "ExpiryDate", "EntrySpot"]
        for col in expected_cols:
            if col not in df.columns:
                if col == "Status":
                    df[col] = "Open"
                elif col == "TradeDate":
                    df[col] = str(date.today())
                elif col == "EntrySpot":
                    df[col] = 0.0
                elif col in ["Strike", "Premium", "Contracts", "Commission", "ID"]:
                    df[col] = 0
                else:
                    df[col] = "N/A"
        return df
    return pd.DataFrame(columns=["ID", "Market", "Strategy", "Series", "Type", "Status", "Strike", "Premium", "Contracts", "Commission", "TradeDate", "ExpiryDate", "EntrySpot"])

def save_trades(df):
    df.to_csv(TRADE_FILE, index=False)

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
                
        update_time = datetime.now(BANGKOK_TZ).strftime("%Y-%m-%d %H:%M:%S (ICT / UTC+7)")
        return price, calc_vol, update_time
    except:
        default_p = 35.0 if market_type == "TFEX USD/THB" else 950.0
        default_v = 0.08 if market_type == "TFEX USD/THB" else 0.18
        update_time = datetime.now(BANGKOK_TZ).strftime("%Y-%m-%d %H:%M:%S (ICT / UTC+7)")
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
st.title("📈 TFEX Multi-Asset Options & Futures Pro Dashboard")
st.markdown("ระบบวิเคราะห์ Payoff Chart รองรับ Gross Position (Open/Close), Time Decay รายวันตามวันที่บันทึก และ Strike Dropdown อัตโนมัติ")

# เลือกตลาดหลักใน Sidebar
st.sidebar.header("⚙️ เลือกตลาดและตั้งค่า")
selected_market = st.sidebar.selectbox("เลือกตลาด TFEX", ["TFEX USD/THB", "TFEX SET50"])

current_spot, auto_volatility, last_update_time = get_market_data(selected_market)

# --- แถบแสดงข้อมูลราคาอ้างอิงสดๆ ฝั่งขวาของหน้าจอหลัก ---
col_head1, col_head2, col_head3 = st.columns([1, 2, 4])
with col_head1:
    if st.button("🔄 รีเฟรชตลาด"):
        st.cache_data.clear()
        st.rerun()
with col_head2:
    st.metric(label=f"🟢 ราคา Spot ปัจจุบัน ({selected_market})", value=f"{current_spot:,.4f}")
with col_head3:
    st.markdown(f"**อัปเดตล่าสุด:** `{last_update_time}`")

st.markdown("---")

spot_price = st.sidebar.number_input(f"ราคาอ้างอิงปัจจุบัน ({selected_market})", value=float(current_spot), format="%.4f")
volatility_input = st.sidebar.slider(
    "Volatility (%) [คำนวณอัตโนมัติจากตลาด]", 
    1.0, 100.0, 
    float(auto_volatility * 100)
) / 100.0

risk_free_rate = 0.025
contract_multiplier = 1000 if selected_market == "TFEX USD/THB" else 200

trades_df = load_trades()

# --- สร้างรายการ Strike แบบ Dropdown จัดรูปแบบทศนิยม (USDTHB = 2 ตำแหน่ง, SET50 = เลขจำนวนเต็ม) ---
if selected_market == "TFEX USD/THB":
    base_center = round(spot_price * 4) / 4.0
    step = 0.25
    # สร้างรายการเป็น float แล้วจัดฟอร์แมต string ทศนิยม 2 ตำแหน่ง
    base_strikes = [f"{round(base_center + i * step, 2):.2f}" for i in range(-25, 26)]
else:
    base_center = round(spot_price / 10.0) * 10.0
    step = 10.0
    # สร้างรายการเป็น int (ไม่มีทศนิยม)
    base_strikes = [int(round(base_center + i * step)) for i in range(-20, 21)]

# --- ฟอร์มเพิ่ม / แก้ไขรายการเทรด ---
st.sidebar.subheader("➕ เพิ่มสถานะการเทรด (Gross Position)")
with st.sidebar.form("trade_form"):
    strategy_name = st.text_input("ชื่อกลยุทธ์ / Note", "Strategy #1")
    default_series = "USDU26" if selected_market == "TFEX USD/THB" else "S50U26"
    series_name = st.text_input("ซีรีส์ TFEX", default_series)
    
    position_status = st.selectbox("สถานะคำสั่ง (Status)", ["Open", "Close"])
    position_type = st.selectbox("ประเภทสัญญา", [
        "Long Futures", "Short Futures", 
        "Long Call Option", "Short Call Option", 
        "Long Put Option", "Short Put Option"
    ])
    
    strike_mode = st.radio("เลือกรูปแบบราคาใช้สิทธิ (Strike)", ["เลือกจาก Dropdown (สเตปอัตโนมัติ)", "พิมพ์ระบุเอง"])
    if strike_mode == "เลือกจาก Dropdown (สเตปอัตโนมัติ)":
        # แปลงค่าใน Dropdown กลับเป็นตัวเลข float เพื่อใช้คำนวณต่อได้ทันที
        if selected_market == "TFEX USD/THB":
            closest_val = round(spot_price * 4) / 4.0
            closest_str = f"{closest_val:.2f}"
            if closest_str in base_strikes:
                default_idx = base_strikes.index(closest_str)
            else:
                default_idx = len(base_strikes) // 2
        else:
            closest_val = int(round(spot_price / 10.0) * 10.0)
            if closest_val in base_strikes:
                default_idx = base_strikes.index(closest_val)
            else:
                default_idx = len(base_strikes) // 2
                
        selected_strike_item = st.selectbox("ราคาใช้สิทธิ (Strike)", options=base_strikes, index=default_idx)
        strike = float(selected_strike_item)
    else:
        strike_default_val = float(round(spot_price, 2)) if selected_market == "TFEX USD/THB" else float(round(spot_price))
        strike = st.number_input("ระบุราคาใช้สิทธิเอง", value=strike_default_val, format="%.2f" if selected_market == "TFEX USD/THB" else "%.0f")

    default_prem = 0.25 if selected_market == "TFEX USD/THB" else 15.0
    premium = st.number_input("ราคาพรีเมี่ยม / ต้นทุนต่อหน่วย", value=float(default_prem), format="%.2f")
    contracts = st.number_input("จำนวนสัญญา (Contracts)", value=1, min_value=1, step=1)
    
    default_comm = 30.0 if selected_market == "TFEX USD/THB" else 60.0
    commission = st.number_input("ค่าคอมมิชชั่นรวมต่อสัญญา (บาท)", value=float(default_comm), format="%.2f")
    
    trade_date = st.date_input("วันที่บันทึกรายการ (Trade Date)", value=date.today())
    expiry_date = st.date_input("วันหมดอายุสัญญา (Expiry Date)", value=date.today() + timedelta(days=30))
    
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
        save_trades(trades_df)
        st.sidebar.success("เพิ่มข้อมูลสำเร็จ!")
        st.rerun()

# --- แสดงตารางจัดการข้อมูลพอร์ต (แก้ไข / ลบได้) ---
st.subheader("📋 รายการเทรดในพอร์ตทั้งหมด (รองรับ Gross Position & ราคาอ้างอิงตอนเข้า)")
if not trades_df.empty:
    edited_df = st.data_editor(trades_df, num_rows="dynamic", use_container_width=True, key="trade_editor")
    
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("💾 บันทึกการแก้ไขตาราง"):
            save_trades(edited_df)
            st.success("บันทึกการเปลี่ยนแปลงเรียบร้อย!")
            st.rerun()
    with col_btn2:
        if st.button("🗑️ ลบข้อมูลทั้งหมดในพอร์ต"):
            if os.path.exists(TRADE_FILE):
                os.remove(TRADE_FILE)
            st.rerun()
    trades_df = edited_df
else:
    st.info("ยังไม่มีข้อมูลในพอร์ต กรุณาเพิ่มสัญญาจากเมนูด้านซ้ายเพื่อเริ่มต้นบันทึกข้อมูล")

# --- กรองข้อมูลเฉพาะตลาดที่เลือก ---
market_trades = pd.DataFrame()
if not trades_df.empty and "Market" in trades_df.columns:
    market_trades = trades_df[trades_df["Market"] == selected_market]

# ==========================================
# ตารางสรุปสถานะ Open/Close และกำไรขาดทุนเทียบกับราคาอ้างอิง
# ==========================================
st.subheader("📊 ตารางสรุปจำนวนสัญญา (Open vs Close) และผลกำไร/ขาดทุนปัจจุบัน")
if not market_trades.empty:
    summary_list = []
    grouped = market_trades.groupby(["Series", "Strike", "Type"])
    for key, group in grouped:
        ser, stk, p_type = key
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
            "Series": ser,
            "Strike": stk,
            "Type": p_type,
            "Open Contracts": open_qty,
            "Close Contracts": close_qty,
            "Net Outstanding": net_qty,
            "Current P&L (THB)": round(item_pnl, 2)
        })
    
    summary_df = pd.DataFrame(summary_list)
    st.dataframe(summary_df, use_container_width=True)
else:
    st.info("ยังไม่มีข้อมูลสรุปสำหรับตลาดนี้")

# ==========================================
# กราฟที่ 1: Payoff ณ วันหมดอายุ (Expiry Payoff)
# ==========================================
st.subheader(f"📈 1. วิเคราะห์ Payoff ณ วันหมดอายุ ({selected_market})")

c_opt1, c_opt2, c_opt3, c_opt4 = st.columns(4)
with c_opt1:
    payoff_mode = st.selectbox("หน่วยแสดงผลกำไร/ขาดทุน", ["บาทรวม (THB)", "จุด (Points)"])
with c_opt2:
    include_comm = st.checkbox("รวมหักค่าคอมมิชชั่น", value=True)
with c_opt3:
    show_greeks_lines = st.checkbox("แสดงรายละเอียดวันหมดอายุ", value=True)
with c_opt4:
    show_breakeven = st.checkbox("แสดงจุดคุ้มทุน (Break-even)", value=True)

price_range = np.linspace(spot_price * 0.85, spot_price * 1.15, 400)
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
        comm_total = float(row["Commission"]) * qty if include_comm else 0.0
        
        sign_multiplier = 1 if status == "Open" else -1
        
        try:
            exp_d = datetime.strptime(str(row["ExpiryDate"]), "%Y-%m-%d").date()
            days_to_expiry = (exp_d - today).days
            T = max(days_to_expiry, 0) / 365.0
        except:
            days_to_expiry = 30
            T = 30 / 365.0
            
        mult_qty = (qty * contract_multiplier if payoff_mode == "บาทรวม (THB)" else qty) * sign_multiplier
        payoff = np.zeros_like(price_range)
        
        if p_type == "Long Futures":
            payoff = (price_range - stk) * mult_qty - (comm_total if status=="Open" else -comm_total)
        elif p_type == "Short Futures":
            payoff = (stk - price_range) * mult_qty - (comm_total if status=="Open" else -comm_total)
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
            name=f"[{status}] {row['Strategy']} ({row['Series']} {p_type})",
            line=dict(dash='dash', width=1.5),
            opacity=0.5,
            hovertemplate=f"<b>[{status}] {row['Strategy']} ({p_type})</b><br>Price: %{{x:.2f}}<br>P&L: %{{y:,.2f}}<extra></extra>"
        ))

fig.add_trace(go.Scatter(
    x=price_range, y=total_payoff,
    mode='lines',
    name='Total Net Portfolio Payoff',
    line=dict(color='blue', width=3),
    hovertemplate="<b>Total Net Portfolio</b><br>Price at Expiry: %{x:.2f}<br>Total P&L: %{y:,.2f}<extra></extra>"
))

fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)

current_portfolio_pnl = 0
if not market_trades.empty:
    for idx, row in market_trades.iterrows():
        p_type = row["Type"]
        stk = float(row["Strike"])
        prem = float(row["Premium"])
        qty = int(row["Contracts"])
        status = row["Status"]
        sign_multiplier = 1 if status == "Open" else -1
        mult_qty = (qty * contract_multiplier if payoff_mode == "บาทรวม (THB)" else qty) * sign_multiplier
        
        if p_type == "Long Futures":
            current_portfolio_pnl += (spot_price - stk) * mult_qty
        elif p_type == "Short Futures":
            current_portfolio_pnl += (stk - spot_price) * mult_qty
        elif "Call" in p_type:
            c_val = bs_option_price(spot_price, stk, 30/365.0, risk_free_rate, volatility_input, "Call")
            if "Long" in p_type:
                current_portfolio_pnl += (c_val - prem) * mult_qty
            else:
                current_portfolio_pnl += (prem - c_val) * mult_qty
        elif "Put" in p_type:
            p_val = bs_option_price(spot_price, stk, 30/365.0, risk_free_rate, volatility_input, "Put")
            if "Long" in p_type:
                current_portfolio_pnl += (p_val - prem) * mult_qty
            else:
                current_portfolio_pnl += (prem - p_val) * mult_qty

fig.add_vline(x=spot_price, line_dash="dot", line_color="red", line_width=2)
fig.add_trace(go.Scatter(
    x=[spot_price], y=[current_portfolio_pnl],
    mode='markers+text',
    name='Current Spot',
    marker=dict(color='red', size=12),
    text=[f"Spot: {spot_price:.2f}<br>P&L: {current_portfolio_pnl:,.2f} THB"],
    textposition="top center",
    hovertemplate="<b>Current Status</b><br>Spot: %{x:.2f}<br>P&L: %{y:,.2f}<extra></extra>"
))

fig.update_layout(
    title=f"{selected_market} Net Portfolio Payoff ({payoff_mode})",
    xaxis_title="Underlying Price at Expiry",
    yaxis_title=f"Profit / Loss ({payoff_mode})",
    hovermode="x unified",
    template="plotly_white",
    height=550,
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
)

st.plotly_chart(fig, use_container_width=True)


# ==========================================
# กราฟที่ 2: จำลอง Payoff รายวันตามวันที่บันทึกจริง & เลือกวันจำลองได้
# ==========================================
st.subheader(f"⏱️ 2. จำลองกราฟ Payoff รายวัน (Interactive Date Simulation)")
st.markdown("ระบบจะจำลองมูลค่าพอร์ตย้อนหลังหรือล่วงหน้าอิงตามวันที่บันทึกจริง (`Trade Date`) และวันหมดอายุ (`Expiry Date`) โดยคุณสามารถเลือกวันที่ต้องการจำลองได้อิสระ")

sim_col1, sim_col2 = st.columns(2)
with sim_col1:
    sim_date = st.date_input("เลือกวันที่ต้องการจำลองสถานะพอร์ต (Simulation Date)", value=date.today())

fig_daily = go.Figure()

if not market_trades.empty:
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    unique_trade_dates = sorted(list(set(market_trades["TradeDate"].astype(str))))
    
    sim_date_str = str(sim_date)
    if sim_date_str not in unique_trade_dates:
        unique_trade_dates.append(sim_date_str)
    
    for i, t_date_str in enumerate(unique_trade_dates):
        try:
            t_d = datetime.strptime(t_date_str, "%Y-%m-%d").date()
        except:
            t_d = date.today()
            
        daily_portfolio_value = np.zeros_like(price_range)
        
        for idx, row in market_trades.iterrows():
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
                if "Long" in p_type:
                    val = (opt_values - prem) * mult_qty
                else:
                    val = (prem - opt_values) * mult_qty
            elif "Put" in p_type:
                opt_values = np.array([bs_option_price(p, stk, T_sim, risk_free_rate, volatility_input, "Put") for p in price_range])
                if "Long" in p_type:
                    val = (opt_values - prem) * mult_qty
                else:
                    val = (prem - opt_values) * mult_qty
            else:
                val = 0
                
            daily_portfolio_value += val
            
        line_color = colors[i % len(colors)]
        is_selected_sim = (t_date_str == sim_date_str)
        line_width = 3.5 if is_selected_sim else 1.5
        line_dash = 'solid' if is_selected_sim else 'dash'
        label_text = f"จำลอง ณ วันที่: {t_date_str}" + (" ⭐ (วันที่เลือก)" if is_selected_sim else "")

        fig_daily.add_trace(go.Scatter(
            x=price_range, y=daily_portfolio_value,
            mode='lines',
            name=label_text,
            line=dict(color=line_color, width=line_width, dash=line_dash),
            hovertemplate=f"<b>{t_date_str}</b><br>Price: %{{x:.2f}}<br>Value: %{{y:,.2f}}<extra></extra>"
        ))

fig_daily.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
fig_daily.add_vline(x=spot_price, line_dash="dot", line_color="red", line_width=2)

fig_daily.update_layout(
    title=f"Time Decay Daily Simulation by Trade Dates ({selected_market})",
    xaxis_title="Underlying Price",
    yaxis_title=f"Portfolio Value ({payoff_mode})",
    hovermode="x unified",
    template="plotly_white",
    height=500,
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
)

st.plotly_chart(fig_daily, use_container_width=True)

if market_trades.empty:
    st.info(f"💡 ขณะนี้ยังไม่มีรายการเทรดในตลาด `{selected_market}` กรุณาเพิ่มรายการเทรดเพื่อแสดงกราฟจำลองรายวัน")

st.subheader("📐 สรุปค่า Greeks และสถานะพอร์ต")
g1, g2, g3, g4 = st.columns(4)
g1.metric("Portfolio Delta", f"{total_greeks['Delta']:,.2f}")
g2.metric("Portfolio Gamma", f"{total_greeks['Gamma']:,.4f}")
g3.metric("Portfolio Theta (Daily)", f"{total_greeks['Theta']:,.2f} THB")
g4.metric("Portfolio Vega", f"{total_greeks['Vega']:,.2f}")
