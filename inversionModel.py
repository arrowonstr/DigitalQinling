import hdf5storage
from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import curve_fit


class BaseInversionModel(ABC):
    """ 反演算法的父类
               Attributes:
                   waveBandIdx:    需要用到的波段索引
                   coefficient:    反演算法中的系数
           """
    def __init__(self, waveBandIdx, coefficient):
        self.waveBandIdx = waveBandIdx
        self.coefficient = coefficient

    @staticmethod
    @abstractmethod
    def func(x, *args):
        """ 反演公式
            子类必须重写如类似以下形式：
            func(x,a,b,c,d) #其中a,b,c,d为需要拟合的参数
                return x[0]*a+x[1]*b**2+x[3]*c-d #对应反演公式

                    Args:
                        x (ndarray):  用于拟合的因变量
                        *args: 其余待定的参数

                    Returns:
                        float: 反演结果
                """
        pass

    def inverse(self, waveBand, otherArgs):
        """ 反演计算
                    Args:
                        waveBand (ndarray):  波段值
                        otherArgs (ndarray): 其余参数
                    Returns:
                        float: 反演结果
                """
        x = waveBand[self.waveBandIdx]
        x = np.append(x, otherArgs)
        args = tuple(self.coefficient)
        return self.func(x, *args)

    def fit(self, waveBand, otherArgs, y):
        """ 拟合参数
                    Args:
                        waveBand (ndarray):  波段值
                        otherArgs (ndarray): 其余参数
                        y (ndarray):    用于拟合的因变量值
                """
        x = waveBand[self.waveBandIdx]
        x = np.append(x, otherArgs)
        self.coefficient = curve_fit(self.func, x, y)[0]


class InversionChlaBDA(BaseInversionModel):
    def __init__(self, waveBandIdx, coefficient):
        super().__init__(waveBandIdx, coefficient)

    @staticmethod
    def func(x, a, b):
        """ 叶绿素a的波段算法，见文档7.2
                    Args:
                        x (ndarray):  为波段662 693 740 705的值

                    Returns:
                        float: 反演结果
                """
        return a*(x[0]**(-1)-x[1]**(-1))/(x[2]**(-1)-x[3]**(-1))-b

def main():
    mat_data = hdf5storage.loadmat('data.mat')
    waveBandArray = mat_data['HSI']
    waveBand = waveBandArray[0][0]
    waveBandIdx = [0, 1, 2, 3]
    coefficient = [0.0097, 0.1268]
    chlaBDA = InversionChlaBDA(waveBandIdx, coefficient)
    chla = chlaBDA.inverse(waveBand, [])
    print(chla)

if __name__ == '__main__':
    main()
