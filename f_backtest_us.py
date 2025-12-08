# db_backtest_us.py
from c_config import US_UNIVERSE_STOCKS
from c_backtest import run_backtest_for_universe

if __name__ == "__main__":
    run_backtest_for_universe(
        US_UNIVERSE_STOCKS,
        model_setting_key="active_model_path_us",
        note_prefix="[US_STOCK] ",
        backtest_days=60,
    )
