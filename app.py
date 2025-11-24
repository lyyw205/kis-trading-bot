import streamlit as st
import pandas as pd
import sqlite3
import time

# 페이지 기본 설정
st.set_page_config(
    page_title="💰 주식 자동매매 대시보드",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🤖 글로벌 주식 자동매매 현황")

# DB에서 데이터 가져오는 함수
def get_data(query):
    try:
        # [중요] 봇과 동일한 절대 경로 사용
        db_path = r"C:\dev\stock-ml-project\trading.db"
        conn = sqlite3.connect(db_path)
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()

# 상단 지표 등을 담을 공간 확보
placeholder = st.empty()

# 1초마다 화면 갱신
while True:
    with placeholder.container():
        # 1. 데이터 조회
        # 최근 로그 20개
        df_logs = get_data("SELECT time, message FROM logs ORDER BY id DESC LIMIT 20")
        # 최근 매매 내역 10개
        df_trades = get_data("SELECT time, symbol, type, price, qty, profit FROM trades ORDER BY id DESC LIMIT 10")
        
        # 2. 화면 레이아웃 (2단 분할)
        col1, col2 = st.columns([1, 1])
        
        # 왼쪽: 매매 체결 내역
        with col1:
            st.subheader("📝 최근 체결 내역")
            if not df_trades.empty:
                # 보기 좋게 컬럼명 변경
                df_trades.columns = ["시간", "종목", "유형", "가격", "수량", "수익률"]
                st.dataframe(df_trades, use_container_width=True, hide_index=True)
            else:
                st.info("아직 체결된 매매 내역이 없습니다.")

        # 오른쪽: 실시간 로그
        with col2:
            st.subheader("📡 실시간 봇 로그")
            if not df_logs.empty:
                # 로그를 텍스트 리스트로 보여줌
                log_text = ""
                for index, row in df_logs.iterrows():
                    log_text += f"[{row['time']}] {row['message']}\n"
                
                # [수정된 부분] key=f"{time.time()}" 를 추가하여 매번 새로운 위젯으로 인식시킴
                st.text_area("Log View", log_text, height=400, disabled=True, key=f"log_view_{time.time()}")
            else:
                st.info("로그 데이터를 기다리는 중...")

        # 1초 대기 (새로고침)
        time.sleep(1)