from artiq.experiment import *
import numpy as np
class Sin(EnvExperiment):
    def build(self):
        self.setattr_device('core')
        self.setattr_device('core_dma')
        self.setattr_device('zotino0')
        self.period = 0.1*s    #设置信号的周期
        self.sample = 750
        #一个周期的取样数
        t = np.linspace(0, 2*np.pi, self.sample)    #一个周期的所有取样点，取为一个类数组
        self.voltages = 2*np.sin(t)                 #设置电压大小的信息
        self.interval = self.period/self.sample
        self.interval_mu = self.core.seconds_to_mu(self.interval)
    @kernel
    def run(self):
        self.core.reset()
        self.core.break_realtime()
        # self.zotino0.init()
        delay(2*ms)
        counter = 0
        while True:
            self.zotino0.set_dac([self.voltages[counter]], [0])          #设置
            counter = (counter+1) % self.sample                     #周期性地取点
            delay_mu(self.interval_mu)

            # delay(self.interval)

