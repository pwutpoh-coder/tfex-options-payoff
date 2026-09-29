import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="TFEX Options Payoff Calculator", layout="wide")

st.title("📈 TFEX Options Strategy Payoff & P&L Calculator")
st.markdown("ระบบคำนวณกราฟ Payoff และตารางสรุปกำไร/ขาดทุน")

# 1. Sidebar สำหรับตั้งค่าพารามิเตอร์หลัก
st.sidebar.header("⚙️ ตั้งค่าตลาดและสัญญา")
spot_current = st.sidebar.number_input(
    "ราคาอ้างอิงปัจจุบัน (Spot / Underlying Index)", value=830.0, step=0.5
)
multiplier = st.sidebar.number_input(
    "ตัวคูณสัญญา (Multiplier เช่น SET50 = 200)", value=200, step=10
)

st.sidebar.header("📊 ช่วงราคาสำหรับกราฟ Payoff")
range_pct = st.sidebar.slider("ช่วงราคา (+/- %)", 5, 30, 15)
min_price = spot_current * (1 - range_pct / 100)
max_price = spot_current * (1 + range_pct / 100)
prices = np.linspace(min_price, max_price, 200)

st.sidebar.header("เลกออปชัน (Options Legs)")
num_legs = st.sidebar.number_input(
    "จำนวนเลกออปชัน", min_value=1, max_value=5, value=2
)

legs = []
for i in range(int(num_legs)):
  st.sidebar.subheader(f"Leg {i+1}")
  col1, col2 = st.sidebar.columns(2)
  with col1:
    pos = st.sidebar.selectbox(
        f"สถานะ Leg {i+1}", ["Long (ซื้อ)", "Short (ขาย)"], key=f"pos_{i}"
    )
    opt_type = st.sidebar.selectbox(
        f"ประเภท Leg {i+1}", ["Call", "Put"], key=f"type_{i}"
    )
  with col2:
    strike = st.sidebar.number_input(
        f"Strike Leg {i+1}", value=float(spot_current), step=1.0, key=f"strike_{i}"
    )
    premium = st.sidebar.number_input(
        f"พรีเมียม (Premium) เลก {i+1}", value=15.0, step=0.5, key=f"prem_{i}"
    )
    qty = st.sidebar.number_input(
        f"จำนวนสัญญา Leg {i+1}", value=1, min_value=1, key=f"qty_{i}"
    )

  legs.append({
      "position": pos,
      "type": opt_type,
      "strike": strike,
      "premium": premium,
      "qty": qty,
  })


# 2. ฟังก์ชันกลางสำหรับคำนวณกำไร/ขาดทุนของแต่ละเลก
def calculate_leg_pnl(leg, s_val, mult):
  pos = leg["position"]
  opt_type = leg["type"]
  strike = leg["strike"]
  premium = leg["premium"]
  qty = leg["qty"]

  if opt_type == "Call":
    intrinsic = np.maximum(0, s_val - strike)
  else:
    intrinsic = np.maximum(0, strike - s_val)

  if "Long" in pos:
    pnl = (intrinsic - premium) * mult * qty
  else:
    pnl = (premium - intrinsic) * mult * qty

  return pnl


# 3. คำนวณเส้น Payoff รวมตลอดช่วงราคา
total_payoff = np.zeros_like(prices)
chart_data = {"Underlying Price": prices}

for idx, leg in enumerate(legs):
  l_pnl = np.array([calculate_leg_pnl(leg, p, multiplier) for p in prices])
  chart_data[f"Leg {idx+1}"] = l_pnl
  total_payoff += l_pnl

chart_data["Total Strategy Payoff"] = total_payoff
df_chart = pd.DataFrame(chart_data).set_index("Underlying Price")

# 4. คำนวณกำไร/ขาดทุน ณ ราคาอ้างอิงปัจจุบัน (Spot Current)
current_total_pnl = 0
summary_data = []

for idx, leg in enumerate(legs):
  leg_current_pnl = calculate_leg_pnl(leg, spot_current, multiplier)
  current_total_pnl += leg_current_pnl

  summary_data.append({
      "Leg": f"Leg {idx+1}",
      "Position": leg["position"],
      "Type": leg["type"],
      "Strike": leg["strike"],
      "Premium": leg["premium"],
      "Quantity": leg["qty"],
      "Current P&L (THB)": round(leg_current_pnl, 2),
  })

df_summary = pd.DataFrame(summary_data)

# 5. ส่วนแสดงผลบนหน้าจอ Streamlit
st.subheader("📊 ผลสรุปกำไร/ขาดทุน ณ จุดอ้างอิงปัจจุบัน")
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("ราคาอ้างอิง (Spot)", f"{spot_current:,.2f}")
col_m2.metric(
    "กำไร/ขาดทุนสุทธิปัจจุบัน",
    f"{current_total_pnl:,.2f} บาท",
    delta=f"{current_total_pnl:,.2f} บาท",
)
col_m3.metric("จำนวนเลกทั้งหมด", f"{len(legs)} เลก")

st.markdown("---")
st.subheader("📋 ตารางตรวจสอบรายละเอียดแยกตามเลก")
st.dataframe(df_summary, use_container_width=True)

st.markdown("---")
st.subheader("📈 กราฟ Payoff Profile ของกลยุทธ์")
st.line_chart(df_chart)

