# import pdb
import time

import numpy as np
import pandas as pd
import requests
import schedule
from predictor import Predictor

from utils import next_weekday


@schedule.repeat(schedule.every(1).hour)
def update_prices():

    today = int(time.time())
    params = {
        "period1": f"{today - 86400 * 15}",
        "period2": f"{today}",
        "interval": "1d",
        "filter": "history",
        "frequency": "1d",
        "includeAdjustedClose": "true",
    }

    #!-------------- Get new prices ------------!#
    response = requests.get(
        URL,
        params=params,
        cookies=cookies,
        headers=headers,
    ).text

    response_list = response.split("\n")
    column_names = response_list.pop(0)

    last_5_days = response_list[-5:]
    close_prices = [w.split(",") for w in last_5_days]

    data = pd.DataFrame(close_prices, columns=column_names.split(","))
    data.index = data["Date"]

    data = data[["Close"]]

    output = net.predict(data["Close"].astype(np.float32).values.reshape(1, -1))

    #!-------------- Calculate next 5 weekdays ------------!#
    temp = [round(w, 6) for w in list(output[0])]
    data_pred = pd.DataFrame(data=temp, columns=["Prediction - Next 5 days"])
    data_pred.index = next_weekday(data.index)

    pd.concat([data, data_pred]).to_csv("5_days_forecast.csv", index=True, na_rep="NULL")

    return


cookies = {}
headers = {}

ticker = "AAPL"
URL = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
net = Predictor(full_path="net_64_0.001_128_10.onnx", provider=["CUDAExecutionProvider", "CPUExecutionProvider"])

while True:
    schedule.run_pending()
    time.sleep(3600)
