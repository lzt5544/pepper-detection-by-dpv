import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression 
from scipy import signal
import pywt

# 最大最小值归一化
def MMS(data):
    return MinMaxScaler().fit_transform(data)


# 标准化
def SS(data):
    return StandardScaler().fit_transform(data)


# 均值中心化
def CT(data):
    for i in range(data.shape[0]):
        MEAN = np.mean(data[i])
        data[i] = data[i] - MEAN
    return data


# 标准正态变换
def SNV(data):
    m = data.shape[0]
    n = data.shape[1]
    # 求标准差
    data_std = np.std(data, axis=1)  # 每条光谱的标准差
    # 求平均值
    data_average = np.mean(data, axis=1)  # 每条光谱的平均值
    # SNV计算
    data_snv = [[((data[i][j] - data_average[i]) / data_std[i]) for j in range(n)] for i in range(m)]
    return  np.array(data_snv)

# 移动平均平滑
def MA(data, WSZ=11):
    for i in range(data.shape[0]):
        out0 = np.convolve(data[i], np.ones(WSZ, dtype=int), 'valid') / WSZ # WSZ是窗口宽度，是奇数
        r = np.arange(1, WSZ - 1, 2)
        start = np.cumsum(data[i, :WSZ - 1])[::2] / r
        stop = (np.cumsum(data[i, :-WSZ:-1])[::2] / r)[::-1]
        data[i] = np.concatenate((start, out0, stop))
    return data


# Savitzky-Golay平滑滤波
def SG(data, w=11, p=2):
    return signal.savgol_filter(data, w, p)


# 一阶导数
def D1(data):
    n, p = data.shape
    Di = np.ones((n, p - 1))
    for i in range(n):
        Di[i] = np.diff(data[i])
    return Di


# 二阶导数
def D2(data):
    data = np.copy(data)
    if isinstance(data, pd.DataFrame):
        data = data.values
    temp2 = (pd.DataFrame(data)).diff(axis=1)
    temp3 = np.delete(temp2.values, 0, axis=1)
    temp4 = (pd.DataFrame(temp3)).diff(axis=1)
    spec_D2 = np.delete(temp4.values, 0, axis=1)
    return spec_D2


# 多元散射校正
def MSC(data):
    n, p = data.shape
    msc = np.ones((n, p))

    for j in range(n):
        mean = np.mean(data, axis=0)

    # 线性拟合
    for i in range(n):
        y = data[i, :]
        l = LinearRegression()
        l.fit(mean.reshape(-1, 1), y.reshape(-1, 1))
        k = l.coef_
        b = l.intercept_
        msc[i, :] = (y - b) / k
    return msc

# 小波变换
def WT(data):
    data = np.copy(data)
    if isinstance(data, pd.DataFrame):
        data = data.values
    def wave_(data):
        w = pywt.Wavelet('db8')  # 选用Daubechies8小波
        maxlev = pywt.dwt_max_level(len(data), w.dec_len)
        coeffs = pywt.wavedec(data, 'db8', level=maxlev)
        threshold = 0.04
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
        datarec = pywt.waverec(coeffs, 'db8')
        return datarec

    tmp = None
    for i in range(data.shape[0]):
        if (i == 0):
            tmp = wave_(data[i])
        else:
            tmp = np.vstack((tmp, wave_(data[i])))
    return tmp

def plot(data , isshow = True):
    ax = plt.figure()
    line_width = 0.5
    if data is None:
        raise 0
    dim = data.ndim
    if dim > 2:
        raise 0
    if dim == 1:
        plt.plot(np.arange(len(data)), data, linewidth=line_width)
    elif dim == 2:
        for line in data:
            plt.plot(np.arange(len(line)), line, linewidth=line_width)
    if isshow:
        plt.show()
    return ax

def add_plot(ax, data):
    line_width = 0.5
    if data is None:
        raise 0
    dim = data.ndim
    if dim > 2:
        raise 0
    if dim == 1:
        ax.plot(np.arange(len(data)), data, linewidth=line_width)
    elif dim == 2:
        for line in data:
            ax.plot(np.arange(len(line)), line, linewidth=line_width)
    return ax

def pre(data, methods, is_save_fig=False, save_fig_path=None, dpi=100):
    rows = int(np.ceil(np.sqrt(len(methods) + 1)))
    cols = int(np.ceil(len(methods) / rows))
    fig,ax = plt.subplots(nrows=rows, ncols=cols, dpi=dpi, figsize=(15,15))
    add_plot(ax[0][0], data)
    res = {}
    for n, method in enumerate(methods):
        i = int((n + 1)  / cols)
        j = int((n + 1) % cols)
        pre_data = method(data)
        add_plot(ax[i][j], pre_data)
        res[method.__qualname__] = pre_data
    return res

if __name__ == "__main__": #test
    data = pd.read_excel(r'E:\Documents\HuaweiCloud\华为云盘\HUAWEI cloud\实验\花椒实验数据\4-2花椒\转置后汇总.xlsx',sheet_name='回归',index_col=0)
    data = data.iloc[:,:-1].values
    methods = [MA,SG,WT,SNV,MSC,CT,MMS,SS,D1,D2]
    r = pre(data, methods)