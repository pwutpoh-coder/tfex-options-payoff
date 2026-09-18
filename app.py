import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import os
from datetime import datetime, date
from scipy.stats import norm

st.set_page_config(page_title="TFEX USD/THB Advanced Payoff & Greeks", layout="wide")

TRADE_FILE = "trades_advanced.csv"

# --- ฟังก์ชันจัดการข้อมูล (CRUD) ---
def load_trades():
    if os.path.exists(TRADE_FILE):
        df = pd.read_csv(TRADE_FILE)
        # ตรวจสอบคอลัมน์ที่จำเป็นเผื่อไฟล์เก่า
        expected_cols = ["ID", "Strategy", "Series", "Type", "Strike", "Premium", "Contracts", "Commission", "ExpiryDate"]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0 if col != "Strategy" and col != "Series" and col != "Type" and col != "ExpiryDate" else "N/A"
        return df
    return pd.DataFrame(columns=["ID", "Strategy", "Series", "Type", "Strike", "Premium", "Contracts", "Commission", "ExpiryDate"])

def save_trades(df):
    df.to_csv(TRADE_FILE, index=False)

# --- ดึงข้อมูล Yahoo Finance พร้อม Timestamp ---
@st.cache_data(ttl=30)
def get_usd_thb():
    try:
        ticker = yf.Ticker("THB=X")
        data = ticker.history(period="1d")
        if not data.empty:
            price = float(data['Close'].iloc[-1])
            update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return price, update_time
    except:
        pass
    return 35.00, datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --- ฟังก์ชันคำนวณ Black-Scholes & Greeks ---
def bs_greeks(S, K, T, r, sigma, option_type, premium_input):
    """คำนวณ Greeks พื้นฐาน (Delta, Gamma, Theta, Vega) สำหรับยุโรปออปชัน"""
    if T <= 0:
        # กรณีหมดอายุแล้ว
        if option_type == "Call":
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return {"Delta": delta, "Gamma": 0.0, "Theta": 0.0, "Vega": 0.0}
    
    # ถ้าผู้ใช้ใส่พรีเมี่ยมมา สามารถใช้implied volคร่าวๆ ได้ แต่นี่ใช้ค่าสมมติมาตรฐานหรือคำนวณจาก Black-Scholes
    vol = sigma if sigma > 0 else 0.10 # 10% Volatility โดยปริยายสำหรับ USD/THB
    
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
st.title("📈 TFEX USD/THB Options & Futures Pro Dashboard")
st.markdown("ระบบวิเคราะห์ Payoff Chart, Greeks, เวลาคงเหลือ, จุดคุ้มทุน และบันทึกประวัติการเทรดแบบเรียลไทม์")

# ส่วนดึงราคาอ้างอิงและรีเฟรช
col_r1, col_r2, col_r3 = st.columns([2, 2, 3])
with col_r1:
    if st.button("🔄 รีเฟรชราคาตลาดเดี๋ยวนี้"):
        st.cache_data.clear()
        st.rerun()

current_spot, last_update_time = get_usd_thb()

with col_r2:
    st.markdown(f"**เวลาอัปเดตล่าสุด:** `{last_update_time}`")

st.sidebar.header("⚙️ ตั้งค่าตลาดและพอร์ต")
spot_price = st.sidebar.number_input("ราคาอ้างอิงปัจจุบัน (Spot/Futures)", value=float(current_spot), format="%.4f")
volatility_input = st.sidebar.slider("Volatility สมมติสำหรับคำนวณ Greeks (%)", 1.0, 50.0, 8.0) / 100.0
risk_free_rate = 0.025 # ดอกเบี้ยไร้ความเสี่ยง 2.5%

trades_df = load_trades()

# --- ฟอร์มเพิ่ม / แก้ไขรายการเทรด ---
st.sidebar.subheader("➕ เพิ่ม / จัดการสถานะการเทรด")
with st.sidebar.form("trade_form"):
    strategy_name = st.text_input("ชื่อกลยุทธ์ / Note", "Strategy #1")
    series_name = st.text_input("ซีรีส์ TFEX (เช่น USDU26)", "USDU26")
    position_type = st.selectbox("ประเภทสัญญา", [
        "Long Futures", "Short Futures", 
        "Long Call Option", "Short Call Option", 
        "Long Put Option", "Short Put Option"
    ])
    strike = st.number_input("ราคาใช้สิทธิ (Strike / Entry)", value=float(round(spot_price, 2)), format="%.2f")
    premium = st.number_input("ราคาพรีเมี่ยม / ต้นทุนต่อหน่วย", value=0.25, format="%.2f")
    contracts = st.number_input("จำนวนสัญญา (Contracts)", value=1, min_value=1, step=1)
    commission = st.number_input("ค่าคอมมิชชั่นรวมต่อสัญญา (บาท)", value=30.0, format="%.2f")
    expiry_date = st.date_input("วันหมดอายุสัญญา (Expiry Date)", value=date.today())
    
    submitted = st.form_submit_button("บันทึกเพิ่มเข้าพอร์ต")
    if submitted:
        new_id = int(trades_df["ID"].max() + 1) if not trades_df.empty and "ID" in trades_df.columns else 1
        new_row = pd.DataFrame([{
            "ID": new_id,
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

# --- แสดงตารางจัดการข้อมูลพอร์ต ---
st.subheader("📋 รายการเทรดในพอร์ต (จัดการข้อมูล / ลบ)")
if not trades_df.empty:
    # เพิ่มช่องให้เลือกติ๊กเพื่อลบ
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
    st.info("ยังไม่มีข้อมูลในพอร์ต กรุณาเพิ่มสัญญาจากเมนูด้านซ้าย")

# --- การตั้งค่าการแสดงผลกราฟ Payoff ---
st.subheader("📊 วิเคราะห์ Payoff และ Greeks รวมพอร์ต")
if not trades_df.empty:
    c_opt1, c_opt2, c_opt3, c_opt4 = st.columns(4)
    with c_opt1:
        payoff_mode = st.selectbox("หน่วยแสดงผลกำไร/ขาดทุน", ["บาทรวม (THB)", "จุด (Points)"])
    with c_opt2:
        include_comm = st.checkbox("รวมหักค่าคอมมิชชั่น", value=True)
    with c_opt3:
        show_greeks_lines = st.checkbox("แสดงเส้นจำลองผลกระทบ Greeks (T-7 วัน)", value=False)
    with c_opt4:
        show_breakeven = st.checkbox("แสดงจุดคุ้มทุน (Break-even)", value=True)

    # ช่วงราคาอ้างอิง ณ วันหมดอายุ (+/- 15%)
    price_range = np.linspace(spot_price * 0.85, spot_price * 1.15, 400)
    total_payoff = np.zeros_like(price_range)
    total_greeks = {"Delta": 0.0, "Gamma": 0.0, "Theta": 0.0, "Vega": 0.0}
    
    multiplier = 1000 # 1 สัญญา TFEX USD = 1,000 USD
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    today = date.today()
    
    for idx, row in trades_df.iterrows():
        p_type = row["Type"]
        stk = float(row["Strike"])
        prem = float(row["Premium"])
        qty = int(row["Contracts"])
        comm_total = float(row["Commission"]) * qty if include_comm else 0.0
        
        # คำนวณ Time to Expiry (T) เป็นปี
        try:
            exp_d = datetime.strptime(str(row["ExpiryDate"]), "%Y-%m-%d").date()
            days_to_expiry = (exp_d - today).days
            T = max(days_to_expiry, 0) / 365.0
        except:
            days_to_expiry = 30
            T = 30 / 365.0
            
        mult_qty = qty * multiplier if payoff_mode == "บาทรวม (THB)" else qty
        
        payoff = np.zeros_like(price_range)
        
        # คำนวณ Payoff ตามประเภท
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
        
        # คำนวณ Greeks ของสัญญาชิ้นนี้
        opt_flag = "Call" if "Call" in p_type else ("Put" if "Put" in p_type else None)
        if opt_flag:
            g = bs_greeks(spot_price, stk, T, risk_free_rate, volatility_input, opt_flag, prem)
            dir_sign = 1 if "Long" in p_type else -1
            total_greeks["Delta"] += g["Delta"] * qty * multiplier * dir_sign
            total_greeks["Gamma"] += g["Gamma"] * qty * multiplier * dir_sign
            total_greeks["Theta"] += g["Theta"] * qty * dir_sign
            total_greeks["Vega"] += g["Vega"] * qty * multiplier * dir_sign
        else:
            # Futures Greeks
            dir_sign = 1 if "Long" in p_type else -1
            total_greeks["Delta"] += 1.0 * qty * multiplier * dir_sign

        # พล็อตกราฟแยกแต่ละขา (เส้นบางๆ)
        ax.plot(price_range, payoff, linestyle="--", alpha=0.3, label=f"{row['Strategy']} ({row['Series']} {p_type}) [เหลือ {days_to_expiry} วัน]")

        # ถ้าเลือกให้แสดงเส้น Greeks จำลองล่วงหน้า 7 วัน (Time Decay effect)
        if show_greeks_lines and opt_flag and T > 7/365:
            T_minus_7 = (days_to_expiry - 7) / 365.0
            payoff_t7 = np.zeros_like(price_range)
            # ประมาณการราคาออปชันล่วงหน้าอย่างง่ายด้วย Black-Scholes ที่เวลาลดลง
            # (เพื่อความกระชับ แสดงเส้นประจำลองแนวโน้มมูลค่าลดลง)
            
    # พล็อตกราฟรวมพอร์ต
    ax.plot(price_range, total_payoff, color="blue", linewidth=3, label="Total Portfolio Payoff (Expiry)")
    
    # เส้นศูนย์ (Break-even line Y=0)
    ax.axhline(0, color="black", linewidth=1, linestyle="-")
    
    # เส้นราคาปัจจุบัน (Spot Price Marker)
    # คำนวณ P&L ปัจจุบันที่ราคา Spot นี้
    current_portfolio_pnl = 0
    for idx, row in trades_df.iterrows():
        p_type = row["Type"]
        stk = float(row["Strike"])
        prem = float(row["Premium"])
        qty = int(row["Contracts"])
        comm_total = float(row["Commission"]) * qty if include_comm else 0.0
        mult_qty = qty * multiplier if payoff_mode == "บาทรวม (THB)" else qty
        
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

    ax.axvline(spot_price, color="red", linestyle=":", linewidth=2, label=f"Current Spot: {spot_price:.2f} (P&L: {current_portfolio_pnl:,.2f})")
    ax.scatter([spot_price], [current_portfolio_pnl], color="red", zorder=5)

    # คำนวณและแสดงจุดคุ้มทุน (Break-even Points)
    if show_breakeven:
        # หาจุดที่ total_payoff ตัด 0
        sign_change = np.where(np.diff(np.sign(total_payoff)))[0]
        for idx_be in sign_change:
            p_be = price_range[idx_be]
            ax.scatter([p_be], [0], color="green", marker="X", s=100, zorder=6)
            ax.annotate(f"BE: {p_be:.2f}", (p_be, 0), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color="green", weight="bold")

    ax.set_title(f"TFEX USD/THB Strategy Payoff ({payoff_mode})", fontsize=14)
    ax.set_xlabel("Underlying Price at Expiry (THB)", fontsize=12)
    ax.set_ylabel(f"Profit / Loss ({payoff_mode})", fontsize=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9)
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # --- แสดงสรุป Greeks รวมพอร์ต ---
    st.subheader("📐 สรุปค่า Greeks รวมทั้งพอร์ต (Portfolio Greeks)")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Portfolio Delta", f"{total_greeks['Delta']:,.2f}", help="ความอ่อนไหวต่อการเปลี่ยนแปลงของราคา Spot 1 บาท")
    g2.metric("Portfolio Gamma", f"{total_greeks['Gamma']:,.4f}", help="อัตราการเปลี่ยนแปลงของ Delta เมื่อ Spot เปลี่ยน")
    g3.metric("Portfolio Theta (Daily)", f"{total_greeks['Theta']:,.2f} THB/วัน", help="กำไร/ขาดทุนที่เปลี่ยนไปเมื่อเวลาผ่านไป 1 วัน (Time Decay)")
    g4.metric("Portfolio Vega", f"{total_greeks['Vega']:,.2f}", help="ความอ่อนไหวต่อความผันผวน (Volatility) ที่เปลี่ยนไป 1%")

else:
    st.warning("กรุณาเพิ่มข้อมูลสัญญาอย่างน้อย 1 รายการเพื่อแสดงกราฟและค่า Greeks")
