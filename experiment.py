from sys import flags

from artiq.language.environment import Experiment
from artiq.compiler.builtins import TInt32
import artiq.coredevice.ad9910 as ad9910
from artiq.language.types import TList
from artiq.language.core import at_mu, delay_mu
from artiq.language.core import kernel
from artiq.language.core import now_mu
from artiq.experiment import *
import numpy as np
import h5py
import copy
import time

import sys
import CArray.create_array as c_array

def print_underflow():
    print("RTIO underflow has occured.")

def str2int10(string):
    return(int(string), 2)

@rpc
#将采集到的adc数据转化为电压的大小，同时设置增益大小。
def adc_mu_to_volt_array(data, gains = [0]*8):
    """Convert ADC data in machine units to Volts.

    :param data: 16 bit signed ADC word
    :param gain: PGIA gain setting (0: 1, ..., 3: 1000)
    :return: Voltage in Volts
    """
    dict = {0:20./(1 << 16), 1:2./(1 << 16), 2:0.2/(1 << 16), 3:0.02/(1 << 16)}
    gains_cal = np.array([dict[gain]] for gain in gains)
    return data*gains_cal

'''______________________________实验部分______________________________'''
class Experiment(EnvExperiment):
    """Complete Experiment"""
    kernel_invariants = {'TTL_array' , 'TTL_channels' , 'TTL_channels_num',
                          'TTL_maxtime_num', 'TTL_unit',
                         'TTL_array_mu' , 'dac', 'DAC_array' , 'DAC_channels',
                         'DAC_update' , 'DAC_interval' , "DAC_channels_num",
                         'DAC_t_num' , 'DAC_delay', 'DAC_list' , 'DAC_delay_mu',
                         'DAC_list_mu' , "Sampler_array_cursor" , 'Sampler_sequences_times',
                         'Sampler_channels_nums', 'Sampler_updates' , 'Sampler_updates_delay_mu'
                         'Sampler_pulse_num_max' , 'Sampler_pulse_nums' , 'Sampler_maxtime',
                         'Sampler_sequence_num' , 'Sampler_data_nums', 'Sampler_time_final_delay',
                         'Sampler_time_final_delay_mu', 'DDS_array', 'DDS_channels',
                         'DDS_channels_num', 'DDS_maxtime_num', 'DDS_unit', 'DDS_array_mu',
                         }
    variable = ['time']
    def prepare(self):
        #设置实验间隔
        self.Experiment_delay_mu = self.core.seconds_to_mu(self.Experiment_delay)

        self.h5_need_read = self.h5_need_read.format(self.h5_main, self.h5_second)
        if self.xml_create == True:
            with h5py.File(self.h5_need_read, 'w') as f:
                f["/run_time"] = time.asctime( time.localtime(time.time()))

        time_unit, voltage_unit = self.TTL_unit.split(';')
        if self.xml_create == True:
            TTL_data = c_array.TTLData(self.xml)
            TTL_arrays = TTL_data.CreateArrayH5(self.h5_need_read)
            if TTL_data.arrays_num == 1:
                self.TTL_array = TTL_arrays[0]
                self.TTL_channels = TTL_data.arrays[0].channels
                self.TTL_index_dds = TTL_data.arrays[0].index_dds
