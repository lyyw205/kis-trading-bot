# ml_train_seq_model_us.py
from c_config import US_UNIVERSE_STOCKS
from f_ml_train import train_seq_model_for_universe

if __name__ == "__main__":
    train_seq_model_for_universe(
        US_UNIVERSE_STOCKS,
        region_filter="US",
        model_setting_key="active_model_path_us",
        note_prefix="[US_STOCK] ",
        model_dir="models_us",
    )
