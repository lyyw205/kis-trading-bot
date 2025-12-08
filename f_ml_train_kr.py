# ml_train_seq_model_kr.py
from c_config import KR_UNIVERSE_STOCKS
from f_ml_train import train_seq_model_for_universe

if __name__ == "__main__":
    train_seq_model_for_universe(
        KR_UNIVERSE_STOCKS,
        region_filter="KR",
        model_setting_key="active_model_path_kr",
        note_prefix="[KR_STOCK] ",
        model_dir="models_kr",  # 폴더 나누고 싶으면 이렇게
    )
