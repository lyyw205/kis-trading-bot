# db_backtest_kr.py
from c_config import KR_UNIVERSE_STOCKS
from c_backtest import run_backtest_for_universe

if __name__ == "__main__":
    run_backtest_for_universe(
        KR_UNIVERSE_STOCKS,
        model_setting_key="active_model_path_kr",  # settings 테이블에 이렇게 저장해 두면 좋음
        note_prefix="[KR_STOCK] ",
        backtest_days=60,  # 필요하면 조절
    )
