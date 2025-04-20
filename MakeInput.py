
import numpy as np
def calculate_rsi(df, window=14):
    delta = df['closing'].diff()  # 종가의 변화량
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()  # 상승 평균
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()  # 하락 평균

    # RSI 계산
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_obv(df):
    obv = np.where(df['closing'].diff() > 0, df['volume'], -df['volume'])  # OBV 계산
    obv = np.cumsum(obv)  # 누적 합산
    return obv


def resample_to_5min(df):
    # 5분봉 데이터프레임 생성
    # df_5min = df.resample('5T').agg({
    #     'opening': 'first',
    #     'high': 'max',
    #     'low': 'min',
    #     'closing': 'last',
    #     'volume': 'sum'
    # })
    # df.dropna(inplace=True)

    # RSI와 OBV 계산하여 컬럼 추가
    df['rsi'] = calculate_rsi(df)
    df['obv'] = calculate_obv(df)

    return df