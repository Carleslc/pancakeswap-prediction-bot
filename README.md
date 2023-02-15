# pancakeswap-prediction-bot
Data analysis and prediction for PancakeSwap Prediction game

## Install

First, install the dependencies:

```sh
pip install --use-pep517 -U -r requirements.txt
```

Then, create a `.env` file from the template:

```sh
cp .env.template .env
```

## Download BNB/USDT data with Binance API

Fill the `.env` with your Binance API `BINANCE_API_KEY` and `BINANCE_API_SECRET`

```sh
python data.py
```

Data will be stored in `data/bnbusdt_7_day_1m.csv` (7 days of BNB/USDT in 1 minute candles)

You can change the length and interval to download in `settings.py`

## Download Pancakeswap Prediction Data with BscScan API

Fill the `.env` with your BscScan API `BSCSCAN_API_KEY`

```sh
python pancake.py
```

Data will be stored in `data/pancake_prediction.csv`
