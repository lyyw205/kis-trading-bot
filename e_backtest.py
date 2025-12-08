# db_backtest_cr.py
from c_config import CR_UNIVERSE_STOCKS
from c_backtest import run_backtest_for_universe

if __name__ == "__main__":
    run_backtest_for_universe(
        CR_UNIVERSE_STOCKS,
        model_setting_key="active_model_path_coin",
        note_prefix="[COIN] ",
        backtest_days=30,  # 코인은 24h라 기간 따로 가져가도 됨
    )
