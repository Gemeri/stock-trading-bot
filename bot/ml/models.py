from forest import SUB_VERSION
import os
import sys
import importlib

def call_sub_main(ticker, df, backtest_amount, position_open=False):
    if SUB_VERSION == "beta":
        sub_main_path = os.path.join(os.path.dirname(__file__), "sub_beta", "main.py")
        spec = importlib.util.spec_from_file_location("sub_main", sub_main_path)
        sub_main = importlib.util.module_from_spec(spec)
        sys.modules["sub_main"] = sub_main
        spec.loader.exec_module(sub_main)
        sub_main.backtest_amount  = backtest_amount
        if backtest_amount == 0:
            csv_path = "_sub_tmp_live.csv"
            df.to_csv(csv_path, index=False)
            sub_main.CSV_PATH = csv_path
            result = sub_main.run_live(ticker, return_result=True, position_open=position_open)
            os.remove(csv_path)
            return result
        elif backtest_amount == -1:
            csv_path = "_sub_tmp_train.csv"
            df.to_csv(csv_path, index=False)
            sub_main.CSV_PATH = csv_path
            analysis = sub_main.train_models()
            os.remove(csv_path)
            return analysis
        else:
            csv_path = "_sub_tmp_backtest.csv"
            df.to_csv(csv_path, index=False)
            sub_main.CSV_PATH = csv_path
            results_df = sub_main.run_backtest(ticker, backtest_amount, return_df=True)
            os.remove(csv_path)
            return results_df
    else:
        sub_main_path = os.path.join(os.path.dirname(__file__), "sub_old", "main.py")
        spec = importlib.util.spec_from_file_location("sub_main", sub_main_path)
        sub_main = importlib.util.module_from_spec(spec)
        sys.modules["sub_main"] = sub_main
        spec.loader.exec_module(sub_main)
        sub_main.backtest_amount = backtest_amount
        if backtest_amount == 0:
            csv_path = "_sub_tmp_live.csv"
            df.to_csv(csv_path, index=False)
            sub_main.CSV_PATH = csv_path
            result = sub_main.run_live(return_result=True, position_open=position_open)
            os.remove(csv_path)
            return result
        else:
            csv_path = "_sub_tmp_backtest.csv"
            df.to_csv(csv_path, index=False)
            sub_main.CSV_PATH = csv_path
            results_df = sub_main.run_backtest(return_df=True)
            os.remove(csv_path)
            return results_df
