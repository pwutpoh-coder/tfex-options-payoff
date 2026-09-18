import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import os
from datetime import datetime, date, timezone, timedelta
from scipy.stats import norm

st.set_page_config(page_title="TFEX Multi-Asset Payoff & Greeks Dashboard", layout="wide")

TRADE_FILE = "trades_multi_asset.csv"

# กำหนดโซนเวลา Bangkok (UTC+7)
BANGKOK_TZ = timezone(timedelta(hours=7))

# --- ฟังก์ชันจัดการข้อมูล (CRUD) ---
def load_trades():
    if os.path.exists(TRADE_FILE):
        df = pd.read_csv(TRADE_FILE)
        expected_cols = ["ID", "Market", "Strategy", "Series", "Type", "Strike", "Premium", "Contracts", "Commission", "ExpiryDate"]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0 if col != "Strategy" and col != "Series" and col != "Type" and col != "ExpiryDate" and col != "Market" else "N/A"
        return df
    return pd.DataFrame(columns=["ID", "Market", "Strategy", "Series", "Type", "Strike", "Premium", "Contracts", "Commission", "ExpiryDate"])

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

# --- ฟังก์ชัน Black-Scholes สำหรับ Option Pricing (ใช้วาดกราฟรายวัน) ---
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
st.markdown("ระบบวิเคราะห์ Payoff Chart (USD/THB และ SET50), Interactive Hover, Greeks, และการจำลอง Payoff รายวัน")

# เลือกตลาดหลักใน Sidebar
st.sidebar.header("⚙️ เลือกตลาดและตั้งค่า")
selected_market = st.sidebar.selectbox("เลือกตลาด TFEX", ["TFEX USD/THB", "TFEX SET50"])

# ดึงราคาและ Volatility อัตโนมัติจากตลาดที่เลือก
current_spot, auto_volatility, last_update_time = get_market_data(selected_market)

col_r1, col_r2, col_r3 = st.columns([2, 4, 2])
with col_r1:
    if st.button("🔄 รีเฟรชข้อมูลตลาดทันที"):
        st.cache_data.clear()
        st.rerun()

with col_r2:
    st.markdown(f"**ตลาด:** `{selected_market}` | **อัปเดตล่าสุด:** `{last_update_time}`")

spot_price = st.sidebar.number_input(f"ราคาอ้างอิงปัจจุบัน ({selected_market})", value=float(current_spot), format="%.4f")
volatility_input = st.sidebar.slider(
    "Volatility (%) [คำนวณอัตโนมัติจากตลาด]", 
    1.0, 100.0, 
    float(auto_volatility * 100)
) / 100.0

risk_free_rate = 0.025
contract_multiplier = 1000 if selected_market == "TFEX USD/THB" else 200

trades_df = load_trades()

# --- ฟอร์มเพิ่ม / แก้ไขรายการเทรด ---
st.sidebar.subheader("➕ เพิ่ม / จัดการสถานะการเทรด")
with st.sidebar.form("trade_form"):
    strategy_name = st.text_input("ชื่อกลยุทธ์ / Note", "Strategy #1")
    default_series = "USDU26" if selected_market == "TFEX USD/THB" else "S50U26"
    series_name = st.text_input("ซีรีส์ TFEX", default_series)
    position_type = st.selectbox("ประเภทสัญญา", [
        "Long Futures", "Short Futures", 
        "Long Call Option", "Short Call Option", 
        "Long Put Option", "Short Put Option"
    ])
    default_strike = round(spot_price, 2)
    strike = st.number_input("ราคาใช้สิทธิ (Strike / Entry)", value=float(default_strike), format="%.2f")
    default_prem = 0.25 if selected_market == "TFEX USD/THB" else 15.0
    premium = st.number_input("ราคาพรีเมี่ยม / ต้นทุนต่อหน่วย", value=float(default_prem), format="%.2f")
    contracts = st.number_input("จำนวนสัญญา (Contracts)", value=1, min_value=1, step=1)
    default_comm = 30.0 if selected_market == "TFEX USD/THB" else 60.0
    commission = st.number_input("ค่าคอมมิชชั่นรวมต่อสัญญา (บาท)", value=float(default_comm), format="%.2f")
    expiry_date = st.date_input("วันหมดอายุสัญญา (Expiry Date)", value=date.today())
    
    submitted = st.form_submit_button("บันทึกเพิ่มเข้าพอร์ต")
    if submitted:
        new_id = int(trades_df["ID"].max() + 1) if not trades_df.empty and "ID" in trades_df.columns else 1
        new_row = pd.DataFrame([{
            "ID": new_id,
            "Market": selected_market,
            "Strategy": strategy_name,
            "Series": series_name,
            "Type": position_type,
            "Strike": strike,
            "Premium": premium,
            "Contracts": contracts,
            "Commission": commission,
            "ExpiryDate": str(expiry_date)
        }])
        trades_df = pd.concat([trades_df, new_row], ignore_index=True)
        save_trades(trades_df)
        st.sidebar.success("เพิ่มข้อมูลสำเร็จ!")
        st.rerun()

# --- แสดงตารางจัดการข้อมูลพอร์ต (แก้ไข / ลบได้) ---
st.subheader("📋 รายการเทรดในพอร์ตทั้งหมด (ทุกตลาด)")
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
# กราฟที่ 1: Payoff ณ วันหมดอายุ (Expiry Payoff)
# ==========================================
st.subheader(f"📊 1. วิเคราะห์ Payoff ณ วันหมดอายุ ({selected_market})")

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
        comm_total = float(row["Commission"]) * qty if include_comm else 0.0
        
        try:
            exp_d = datetime.strptime(str(row["ExpiryDate"]), "%Y-%m-%d").date()
            days_to_expiry = (exp_d - today).days
            T = max(days_to_expiry, 0) / 365.0
        except:
            days_to_expiry = 30
            T = 30 / 365.0
            
        mult_qty = qty * contract_multiplier if payoff_mode == "บาทรวม (THB)" else qty
        payoff = np.zeros_like(price_range)
        
        if p_type == "Long Futures":
            payoff = (price_range - stk) * mult_qty - comm_total
        elif p_type == "Short Futures":
            payoff = (stk - price_range) * mult_qty - comm_total
        elif p_type == "Long Call Option":
            payoff = (np.maximum(0, price_range - stk) - prem) * mult_qty - comm_total
        elif p_type == "Short Call Option":
            payoff = (prem - np.maximum(0, price_range - stk)) * mult_qty - comm_total
        elif p_type == "Long Put Option":
            payoff = (np.maximum(0, stk - price_range) - prem) * mult_qty - comm_total
        elif p_type == "Short Put Option":
            payoff = (prem - np.maximum(0, stk - price_range)) * mult_qty - comm_total
            
        total_payoff += payoff
        
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
            name=f"{row['Strategy']} ({row['Series']} {p_type})",
            line=dict(dash='dash', width=1.5),
            opacity=0.5,
            hovertemplate=f"<b>{row['Strategy']} ({p_type})</b><br>Price: %{{x:.2f}}<br>P&L: %{{y:,.2f}}<extra></extra>"
        ))

fig.add_trace(go.Scatter(
    x=price_range, y=total_payoff,
    mode='lines',
    name='Total Portfolio Payoff',
    line=dict(color='blue', width=3),
    hovertemplate="<b>Total Portfolio</b><br>Price at Expiry: %{x:.2f}<br>Total P&L: %{y:,.2f}<extra></extra>"
))

fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)

current_portfolio_pnl = 0
if not market_trades.empty:
    for idx, row in market_trades.iterrows():
        p_type = row["Type"]
        stk = float(row["Strike"])
        prem = float(row["Premium"])
        qty = int(row["Contracts"])
        comm_total = float(row["Commission"]) * qty if include_comm else 0.0
        mult_qty = qty * contract_multiplier if payoff_mode == "บาทรวม (THB)" else qty
        
        if p_type == "Long Futures":
            current_portfolio_pnl += (spot_price - stk) * mult_qty - comm_total
        elif p_type == "Short Futures":
            current_portfolio_pnl += (stk - spot_price) * mult_qty - comm_total
        elif p_type == "Long Call Option":
            current_portfolio_pnl += (np.maximum(0, spot_price - stk) - prem) * mult_qty - comm_total
        elif p_type == "Short Call Option":
            current_portfolio_pnl += (prem - np.maximum(0, spot_price - stk)) * mult_qty - comm_total
        elif p_type == "Long Put Option":
            current_portfolio_pnl += (np.maximum(0, stk - spot_price) - prem) * mult_qty - comm_total
        elif p_type == "Short Put Option":
            current_portfolio_pnl += (prem - np.maximum(0, stk - spot_price)) * mult_qty - comm_total

fig.add_vline(x=spot_price, line_dash="dot", line_color="red", line_width=2)
fig.add_trace(go.Scatter(
    x=[spot_price], y=[current_portfolio_pnl],
    mode='markers+text',
    name='Current Spot',
    marker=dict(color='red', size=12),
    text=[f"Current Spot: {spot_price:.2f}<br>P&L: {current_portfolio_pnl:,.2f} THB"],
    textposition="top center",
    hovertemplate="<b>Current Status</b><br>Spot: %{x:.2f}<br>Current P&L: %{y:,.2f}<extra></extra>"
))

if show_breakeven and not market_trades.empty:
    sign_change = np.where(np.diff(np.sign(total_payoff)))[0]
    be_x = []
    be_y = []
    be_text = []
    for idx_be in sign_change:
        p_be = price_range[idx_be]
        be_x.append(p_be)
        be_y.append(0)
        be_text.append(f"BE: {p_be:.2f}")
    
    if be_x:
        fig.add_trace(go.Scatter(
            x=be_x, y=be_y,
            mode='markers+text',
            name='Break-even Points',
            marker=dict(color='green', symbol='x', size=12),
            text=be_text,
            textposition="bottom center",
            hovertemplate="<b>Break-even</b><br>Price: %{x:.2f}<extra></extra>"
        ))

fig.update_layout(
    title=f"{selected_market} Strategy Payoff Chart ({payoff_mode})",
    xaxis_title="Underlying Price at Expiry",
    yaxis_title=f"Profit / Loss ({payoff_mode})",
    hovermode="x unified",
    template="plotly_white",
    height=550,
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
)

st.plotly_chart(fig, use_container_width=True)


# ==========================================
# กราฟที่ 2: จำลอง Payoff แบบรายวัน (Daily Time Decay Simulation)
# ==========================================
st.subheader(f"⏱️ 2. จำลองกราฟ Payoff รายวัน (Time Decay Simulation)")
st.markdown("แสดงเส้นมูลค่าพอร์ตจำลองล่วงหน้าตามจำนวนวันที่เหลืออยู่ก่อนหมดอายุ (สเตปละ 0, 7, 15, 30 วัน หรือตามวันหมดอายุจริง)")

fig_daily = go.Figure()

if not market_trades.empty:
    # หาค่าวันหมดอายุสูงสุดในพอร์ตเพื่อทำ Simulation ช่วงเวลา
    max_days = 30
    for idx, row in market_trades.iterrows():
        try:
            exp_d = datetime.strptime(str(row["ExpiryDate"]), "%Y-%m-%d").date()
            d_rem = (exp_d - today).days
            if d_rem > max_days:
                max_days = d_rem
        except:
            pass

    # กำหนดจุดเวลาที่จะนำมาวาดจำลอง (เช่น วันนี้, อีก 7 วัน, อีก 15 วัน, อีก 30 วัน และวันหมดอายุ 0)
    step_days = sorted(list(set([0, min(max_days, 7), min(max_days, 15), min(max_days, 30), max_days])))
    
    # สีสำหรับแต่ละเส้นเวลา (ไล่เฉด)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, d_rem in enumerate(step_days):
        T_sim = max(d_rem, 0) / 365.0
        daily_portfolio_value = np.zeros_like(price_range)

        for idx, row in market_trades.iterrows():
            p_type = row["Type"]
            stk = float(row["Strike"])
            prem = float(row["Premium"])
            qty = int(row["Contracts"])
            comm_total = float(row["Commission"]) * qty if include_comm else 0.0
            mult_qty = qty * contract_multiplier if payoff_mode == "บาทรวม (THB)" else qty

            if p_type == "Long Futures":
                val = (price_range - stk) * mult_qty - comm_total
            elif p_type == "Short Futures":
                val = (stk - price_range) * mult_qty - comm_total
            elif "Call" in p_type:
                # คำนวณมูลค่าทางทฤษฎี Black-Scholes ตาม T_sim
                opt_values = np.array([bs_option_price(p, stk, T_sim, risk_free_rate, volatility_input, "Call") for p in price_range])
                if "Long" in p_type:
                    val = (opt_values - prem) * mult_qty - comm_total
                else:
                    val = (prem - opt_values) * mult_qty - comm_total
            elif "Put" in p_type:
                opt_values = np.array([bs_option_price(p, stk, T_sim, risk_free_rate, volatility_input, "Put") for p in price_range])
                if "Long" in p_type:
                    val = (opt_values - prem) * mult_qty - comm_total
                else:
                    val = (prem - opt_values) * mult_qty - comm_total
            else:
                val = 0

            daily_portfolio_value += val

        line_color = colors[i % len(colors)]
        line_width = 3 if d_rem == 0 else 1.5
        line_dash = 'solid' if d_rem == 0 else 'dash'

        fig_daily.add_trace(go.Scatter(
            x=price_range, y=daily_portfolio_value,
            mode='lines',
            name=f"เหลือเวลา {d_rem} วัน (T = {d_rem}d)",
            line=dict(color=line_color, width=line_width, dash=line_dash),
            hovertemplate=f"<b>เหลือ {d_rem} วัน</b><br>Price: %{{x:.2f}}<br>Value: %{{y:,.2f}}<extra></extra>"
        ))

fig_daily.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
fig_daily.add_vline(x=spot_price, line_dash="dot", line_color="red", line_width=2)

fig_daily.update_layout(
    title=f"Time Decay Payoff Simulation ({selected_market})",
    xaxis_title="Underlying Price",
    yaxis_title=f"Portfolio Value ({payoff_mode})",
    hovermode="x unified",
    template="plotly_white",
    height=500,
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
)

st.plotly_chart(fig_daily, use_container_width=True)

if market_trades.empty:
    st.info(f"💡 ขณะนี้ยังไม่มีรายการเทรดในตลาด `{selected_market}` กรุณาเพิ่มรายการเทรดจากเมนูด้านซ้ายเพื่อแสดงกราฟจำลองรายวัน")

st.subheader("📐 สรุปค่า Greeks และสถานะพอร์ต")
g1, g2, g3, g4 = st.columns(4)
g1.metric("Portfolio Delta", f"{total_greeks['Delta']:,.2f}")
g2.metric("Portfolio Gamma", f"{total_greeks['Gamma']:,.4f}")
g3.metric("Portfolio Theta (Daily)", f"{total_greeks['Theta']:,.2f} THB")
g4.metric("Portfolio Vega", f"{total_greeks['Vega']:,.2f}")
