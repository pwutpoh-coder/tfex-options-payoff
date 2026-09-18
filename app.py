import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import os

st.set_page_config(page_title="TFEX USD/THB Payoff Calculator", layout="wide")

# ไฟล์สำหรับเก็บบันทึกข้อมูลการเทรด
TRADE_FILE = "trades.csv"

def load_trades():
    if os.path.exists(TRADE_FILE):
        return pd.read_csv(TRADE_FILE)
    return pd.DataFrame(columns=["Strategy", "Type", "Strike", "Premium", "Contracts"])

def save_trades(df):
    df.to_csv(TRADE_FILE, index=False)

# ดึงข้อมูลค่าเงิน USD/THB จาก Yahoo Finance แบบเรียลไทม์
@st.cache_data(ttl=60)
def get_usd_thb():
    try:
        ticker = yf.Ticker("THB=X")
        data = ticker.history(period="1d")
        if not data.empty:
            return float(data['Close'].iloc[-1])
    except:
        pass
    return 35.00 # ค่าสำรองเผื่อกรณีเรียก API ไม่ได้

st.title("📈 TFEX USD/THB Options & Futures Payoff Calculator")
st.write("เครื่องมือวิเคราะห์กำไรขาดทุน (Payoff Chart) สำหรับสัญญาซื้อขายล่วงหน้าและออปชัน USD/THB พร้อมระบบบันทึกพอร์ตการเทรด")

# ดึงราคาปัจจุบัน
current_price = get_usd_thb()

st.sidebar.header("⚙️ ตั้งค่าตลาดและพอร์ต")
spot_price = st.sidebar.number_input("ราคาอ้างอิงปัจจุบัน (Spot/Futures)", value=float(current_price), format="%.4f")

# โหลดข้อมูลเทรดเดิม
trades_df = load_trades()

st.sidebar.subheader("➕ เพิ่มสถานะการเทรด (Position)")
with st.sidebar.form("trade_form"):
    strategy_name = st.text_input("ชื่อกลยุทธ์ / Note", "Trade #1")
    position_type = st.selectbox("ประเภทสัญญา", [
        "Long Futures", "Short Futures", 
        "Long Call Option", "Short Call Option", 
        "Long Put Option", "Short Put Option"
    ])
    strike = st.number_input("ราคาใช้สิทธิ (Strike / Entry Price)", value=float(round(spot_price, 2)), format="%.2f")
    premium = st.number_input("ราคาพรีเมี่ยม / ต้นทุนต่อหน่วย", value=0.25, format="%.2f")
    contracts = st.number_input("จำนวนสัญญา (Contracts)", value=1, min_value=1, step=1)
    
    submitted = st.form_submit_button("บันทึกเข้าพอร์ต")
    if submitted:
        new_row = pd.DataFrame([{
            "Strategy": strategy_name,
            "Type": position_type,
            "Strike": strike,
            "Premium": premium,
            "Contracts": contracts
        }])
        trades_df = pd.concat([trades_df, new_row], ignore_index=True)
        save_trades(trades_df)
        st.sidebar.success("บันทึกสำเร็จ!")
        st.rerun()

# แสดงรายการเทรดทั้งหมด
st.subheader("📋 รายการเทรดในพอร์ตของคุณ")
if not trades_df.empty:
    st.dataframe(trades_df, use_container_width=True)
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🗑️ ล้างข้อมูลทั้งหมด"):
            if os.path.exists(TRADE_FILE):
                os.remove(TRADE_FILE)
            trades_df = pd.DataFrame(columns=["Strategy", "Type", "Strike", "Premium", "Contracts"])
            st.rerun()
else:
    st.info("ยังไม่มีข้อมูลในพอร์ต กรุณาเพิ่มสัญญาจากเมนูด้านซ้าย")

# คำนวณและวาดกราฟ Payoff
if not trades_df.empty:
    st.subheader("📊 กราฟจำลองกำไรขาดทุนรวม (Portfolio Payoff Chart)")
    
    # กำหนดช่วงราคาอ้างอิง ณ วันหมดอายุ (+/- 10% จากราคาปัจจุบัน)
    price_range = np.linspace(spot_price * 0.9, spot_price * 1.1, 300)
    total_payoff = np.zeros_like(price_range)
    
    # ตัวคูณสัญญา TFEX USD Futures (โดยทั่วไป 1 สัญญา = 1,000 USD)
    multiplier = 1000 
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for idx, row in trades_df.iterrows():
        p_type = row["Type"]
        stk = row["Strike"]
        prem = row["Premium"]
        qty = row["Contracts"] * multiplier
        
        payoff = np.zeros_like(price_range)
        
        if p_type == "Long Futures":
            payoff = (price_range - stk) * qty
        elif p_type == "Short Futures":
            payoff = (stk - price_range) * qty
        elif p_type == "Long Call Option":
            payoff = (np.maximum(0, price_range - stk) - prem) * qty
        elif p_type == "Short Call Option":
            payoff = (prem - np.maximum(0, price_range - stk)) * qty
        elif p_type == "Long Put Option":
            payoff = (np.maximum(0, stk - price_range) - prem) * qty
        elif p_type == "Short Put Option":
            payoff = (prem - np.maximum(0, stk - price_range)) * qty
            
        total_payoff += payoff
        ax.plot(price_range, payoff, linestyle="--", alpha=0.4, label=f"{row['Strategy']} ({p_type})")
        
    # พล็อตกราฟเส้นรวมพอร์ต
    ax.plot(price_range, total_payoff, color="purple", linewidth=3, label="Total Portfolio Payoff")
    ax.axhline(0, color="black", linewidth=1, linestyle="-")
    ax.axvline(spot_price, color="red", linestyle=":", label=f"Spot Price: {spot_price:.2f}")
    
    ax.set_title("USD/THB Options & Futures Strategy Payoff", fontsize=14)
    ax.set_xlabel("Underlying Price at Expiry (THB)", fontsize=12)
    ax.set_ylabel("Profit / Loss (THB)", fontsize=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
