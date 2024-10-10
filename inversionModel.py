import hdf5storage
from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import curve_fit

from utils.utils import parse_hdr_file,show_result
class BaseInversionModel(ABC):
    """ 反演算法的父类
               Attributes:
                   waveBand:    摇杆波段数据
                   waveLength:  遥感波段值
                   waveBandIdx: 需要用到的波段值
                   coefficient: 反演算法中的系数

           """
    def __init__(self, waveBand, waveLength, waveBandIdx, coefficient):
        self.waveLength = waveLength
        self.waveBand = waveBand
        self.coefficient = coefficient

        # 根据需要的波段值从数据中波段提取对应索引
        self.waveBandIdx = [waveLength.index(i) for i in waveBandIdx]

    @staticmethod
    @abstractmethod
    def func(x, *args):
        """ 反演公式
            子类必须重写如类似以下形式：
            func(x,a,b,c,d) #其中a,b,c,d为需要拟合的参数
                return x[0]*a+x[1]*b**2+x[3]*c-d #对应反演公式

                    Args:
                        x (ndarray):  用于拟合的因变量(h,w,l)
                        *args: 其余待定的参数

                    Returns:
                        float: 反演结果
                """
        pass

    def inverse(self, waveBand, otherArgs):
        """ 反演计算
                    Args:
                        waveBand (ndarray):  波段值(h,w,l)
                        otherArgs (ndarray): 其余参数(h,w,l)
                    Returns:
                        float: 反演结果
                """
        # ...
        x = waveBand[:, :, self.waveBandIdx]
        if otherArgs:
            x = np.concatenate((x, otherArgs), axis=2)
        args = tuple(self.coefficient)
        return self.func(x, *args)

    def fit(self, waveBand, otherArgs, y):
        """ 拟合参数
                    Args:
                        waveBand (ndarray):  波段值
                        otherArgs (ndarray): 其余参数
                        y (ndarray):    用于拟合的因变量值
                """
        x = waveBand[:, :, self.waveBandIdx]
        if otherArgs:
            x = np.concatenate((x, otherArgs), axis=2)
        self.coefficient = curve_fit(self.func, x, y)[0]


class InversionChlaBDA(BaseInversionModel):
    def __init__(self, waveBand, waveLength, waveBandIdx, coefficient):
        super().__init__(waveBand, waveLength, waveBandIdx, coefficient)

    @staticmethod
    def func(x, a, b):
        """ 叶绿素a的波段算法，见文档7.2
                    Args:
                        x (ndarray):  为波段662 693 740 705的值

                    Returns:
                        float: 反演结果
                """

        # 如果是拟合过程调用的func 需要调整维度以适配函数
        isFit = False
        if not x.ndim == 3:
            x = x.reshape(1, 1, -1)
            isFit = True

        result = a*(1/x[:, :, 0]-1/x[:, :, 1])/(1/x[:, :, 2]-1/x[:, :, 3])-b

        if isFit:
            return result[0, 0]

        return result

def main():
    mat_data = hdf5storage.loadmat('data.mat')
    waveBandArray = mat_data['HSI']
    waveBand = waveBandArray
    waveLength = parse_hdr_file('raw.hdr')
    waveBandIdx = [662.476, 693.526, 740.101, 704.615]
    coefficient = [0.0097, 0.1268]
    chlaBDA = InversionChlaBDA(waveBand, waveLength, waveBandIdx, coefficient)
    chla = chlaBDA.inverse(waveBand, [])

    mask = hdf5storage.loadmat('mask.mat')['tg']
    chla = chla*mask
    show_result(chla)

if __name__ == '__main__':
    main()
