import numpy as np 
import xml.dom.minidom as minidom
#这个是为了测试如何根据需要产生2D数组
import h5py
#a = np.linspace(0,1,10)
#b = np.linspace(1,2,5)

#print(np.vstack((a,b)))

#左闭右开
class Shape(object):
    def __init__(self,name,time,voltage,update):
        self.name=name
        self.time=time
        self.length=len(time)
        self.voltage=voltage
        self.update=update

    def number(self,t1,t2,update):
        return int((t2-t1)*update)

class shape_square(Shape):
    def create_array(self):
        n=self.number(self.time[0],self.time[1],self.update)
        fb_voltage=np.linspace(self.voltage[0],self.voltage[1],n,endpoint=False)
        return fb_voltage

class shape_sin(Shape):
    def create_array(self):
        n=self.number(self.time[0],self.time[1],self.update)
        fb_voltage=np.linspace(self.voltage[0],self.voltage[1],n,endpoint=False)
        return fb_voltage



class DACArray(object):                     #根据dom_array的内容创建一个DACArray的对象，便于分析。
    def __init__(self, dom_array, dict = {'方波': shape_square,'正弦波': shape_sin}):
        # dom_array是xml文件里面array标签对应的Elements对象，在其他程序里面使用getElementsByTagName('array')[index]来获得
        self.array = dom_array
        if dom_array.getAttribute("dma") == "true":
            self.dma = True
        elif dom_array.getAttribute("dma") == "false":
            self.dma = False
        update = int(dom_array.getElementsByTagName('update')[0].firstChild.data)
        sequences = dom_array.getElementsByTagName('sequence')
        self.update = update
        self.channels = []
        self.maxtime = float(0)
        self.sequences = sequences              #波形序列
        self.dict = dict
        self.time_final_delay = float(0)
        self.interval = 1 / update
        self.sequence_num = len(sequences)
        self.unit_dict = {"s": 1.0,"ms":10**(-3),"us":10**(-6),"ns":10**(-9),
                          "V":1.0,"mV":10**(-3),"uV":10**(-6)}
        self.time_final = self.CursorCal(dom_array,'timefinal')


    def CheckElement(self):
        #用于检查dom_array的属性,将其全部打印出来。
        print(self.dma)
        print(self.sequences[0])
        print(self.sequences[0].firstChild)
        print(self.sequences[0].childNodes)
        print(self.sequences[0].getElementsByTagName('pulse')[0].getElementsByTagName('type')[0].firstChild)
        print(self.sequences[0].getElementsByTagName('pulse')[0].getElementsByTagName('timestart')[0].firstChild.data)
        print(self.sequences[0].getElementsByTagName('pulse')[0].getElementsByTagName('timeend')[0].firstChild.data)
        print(self.sequences[0].getElementsByTagName('pulse')[0].getElementsByTagName('voltagestart')[0].firstChild.data)
        print(self.sequences[0].getElementsByTagName('pulse')[0].getElementsByTagName('voltageend')[0].firstChild.data)
        if self.sequences[0].getElementsByTagName('pulse')[0].getElementsByTagName('voltagestart')[0].hasAttribute("unit"):
            print(self.sequences[0].getElementsByTagName('pulse')[0].getElementsByTagName('voltagestart')[0].getAttribute("unit"))

        print(self.sequences[0].getElementsByTagName('pulse')[0].getElementsByTagName('voltagee'))
        print(self.TimeCursorSave(self.sequences[0].getElementsByTagName('pulse')[0]))

    def LoadTime(self):
        final_time_end = []
        channels = []                   #定义不同通道的列表
        update = self.update
        for _ in self.sequences:
            channels.append(int(_.getAttribute("id")))
            time_end = self.CursorCal(_.getElementsByTagName('pulse')[-1],'timeend')
            assert (update*time_end).is_integer()
            final_time_end.append(time_end)
        maxtime = max(final_time_end)
        time_last = final_time_end[-1]
        self.maxtime = maxtime          #不同通道当中结束最晚的时间
        self.time_final_delay = self.time_final - time_last
        self.time_min_delay = self.time_final - maxtime
        self.channels = channels   
        return np.zeros((self.sequence_num, int(update*maxtime)))       #返回一个通道数为行，总数据数量为列的二维数组用于储存数据

    #返回一个浮点型的标定点
    #分析'pulse'标签下，string（timestart, timeend, voltagestart, voltageend）节点的内容。根据单位返回一个浮点数。
    def CursorCal(self,node,string,isfloat = True):
        cursors = []
        if not node.getElementsByTagName(string):           #如果不存在目标的字符串命名的节点，返回一个空列表。
            return cursors
        # 找到string的单位。
        unit = node.getElementsByTagName(string)[0].getAttribute("unit")
        Node_string = node.getElementsByTagName(string)
        for _ in Node_string:
            cursors.append(float(_.firstChild.data)*self.unit_dict[unit])       #将带单位的数据转化成浮点数
        if isfloat:                                         #判断是不是浮点数，如果是，返回SI单位的值
            cursors = float(cursors[0])
        return cursors
        
    
    def TimeCursorSave(self,node):
        time_cursors = []
        time_cursors.append(self.CursorCal(node,'timestart'))
        time_cursors.extend(self.CursorCal(node,'cursor',False))
        time_cursors.append(self.CursorCal(node,'timeend'))
        return time_cursors

    def VoltageCursorSave(self,node):
        voltage_cursors = []
        voltage_cursors.append(self.CursorCal(node,'voltagestart'))
        voltage_cursors.extend(self.CursorCal(node,'voltagecursor',False))
        voltage_cursors.append(self.CursorCal(node,'voltageend'))
        return voltage_cursors


    def Create2DArray(self,voffset = 0):
        final_array = self.LoadTime()
        update = self.update
        for i,_ in enumerate(self.sequences):
            for j,__ in enumerate(_.getElementsByTagName('pulse')):
                cursors = __.getElementsByTagName('cursor')
                frequencies = __.getElementsByTagName('frequency')
                if len(cursors) == 0 and len(frequencies) == 0:
                    type = __.getElementsByTagName('type')[0].firstChild.data
                    time = self.TimeCursorSave(__)
                    voltage = self.VoltageCursorSave(__)
                    if type not in self.dict:
                        type = "方波"
                    shape_object = self.dict.get(type)(type, time, voltage, update)
                    if j == 0:
                        t1 = 0
                    elif j > 0:
                        t1 = t3
                    t2 = time[0]
                    t3 = time[-1]
                    n = shape_object.number(t1,t2,update)
                    array_temp = shape_object.create_array()
                    m = len(array_temp)
                    N = m+n
                    n1 = shape_object.number(0,t1,update)
                    final_array[i,n1:n1+n] = np.linspace(voffset,voffset,n)
                    final_array[i,n1+n:n1+N] = array_temp
        
        return final_array

    




class DACData(object):
    def __init__(self, file, dict = {'方波': shape_square,'正弦波': shape_sin}):
        dom = minidom.parse(file)
        root = dom.documentElement
        dac = root.getElementsByTagName('DAC')[0]
        self.arrays = []
        self.dom_arrays = dac.getElementsByTagName('array')
        for array in self.dom_arrays:
            self.arrays.append(DACArray(array))
        self.dom = dom
        self.root = root
        self.dac = dac
        self.dict = dict
        self.arrays_num = len(self.arrays)
        self.unit_dict = {"s": 1.0,"ms":10**(-3),"us":10**(-6),"ns":10**(-9),"V":1.0,"mV":10**(-3),"uV":10**(-6)}


#用来检验是否名称与值对应正常
    def check_element(self):
        print(self.dom)
        print(self.root)
        print(self.dac)
        print(self.dict)
        print(self.root.toxml())
        
    def Create2DArrayH5(self, filename, voffset= 0):
        final_arrays = []
        with h5py.File(filename, "a") as f:
            f["/DAC/arrays_num"] = self.arrays_num
            for n,array in enumerate(self.arrays):
                final_array = array.Create2DArray(voffset)
                final_arrays.append(final_array)
                f["/DAC/{}/data".format(n)] = final_array
                f["/DAC/{}/channels".format(n)] = array.channels
                f["/DAC/{}/update".format(n)] = array.update
                f["/DAC/{}/interval".format(n)] = array.interval
                f["/DAC/{}/channels_num".format(n)] = array.sequence_num
                f["/DAC/{}/dma".format(n)] = array.dma
                f["/DAC/{}/time_final_delay".format(n)] = array.time_final_delay
                f["/DAC/{}/time_min_delay".format(n)] = array.time_min_delay
        return final_arrays


class TTLArray(object):
    def __init__(self, dom_array):
        self.array = dom_array
        if dom_array.getAttribute("dma") == "true":
            self.dma = True
        elif dom_array.getAttribute("dma") == "false":
            self.dma = False
        sequences = dom_array.getElementsByTagName('sequence')
        self.channels = []
        self.maxtime = float(0)
        self.maxtime_num = 0
        self.time_final_delay = float(0)
        self.sequences = sequences
        self.dict = dict
        self.sequence_num = len(sequences)
        self.unit_dict = {"s": 1.0,"ms":10**(-3),"us":10**(-6),"ns":10**(-9),"V":1.0,"mV":10**(-3),"uV":10**(-6)}
        self.time_final = self.CursorCal(dom_array,'timefinal')
    
    def LoadTime(self):
        final_time_end = []
        pulse_nums = []
        channels = []
        index_dds = []
        for n,_ in enumerate(self.sequences):
            if _.getAttribute("type") == 'dds':
                channels.append(int(_.getAttribute("id")))
                index_dds.append(n)
            else:
                channels.append(int(_.getAttribute("id")))
            time_end = self.CursorCal(_.getElementsByTagName('pulse')[-1],'timeend')
            pulse_nums.append(len(_.getElementsByTagName('pulse')))
            final_time_end.append(time_end)
        maxtime = max(final_time_end)
        time_last = final_time_end[-1]
        self.maxtime = maxtime
        self.time_final_delay = self.time_final - time_last
        self.time_min_delay = self.time_final - maxtime
        self.pulse_nums = pulse_nums
        self.channels = channels
        self.index_dds = index_dds
        self.maxtime_num = max(pulse_nums)   
        return np.zeros((self.sequence_num, self.maxtime_num*2))

    #返回一个浮点型的标定点
    def CursorCal(self,node,string,isfloat = True):
        cursors = []
        if not node.getElementsByTagName(string):
            return cursors
        unit = node.getElementsByTagName(string)[0].getAttribute("unit")
        Node_string = node.getElementsByTagName(string)
        for _ in Node_string:
            cursors.append(float(_.firstChild.data)*self.unit_dict[unit])
        if isfloat:
            cursors = float(cursors[0])
        return cursors

    def TimeCursorSave(self,node):
        time_cursors = []
        time_cursors.append(self.CursorCal(node,'timestart'))
        time_cursors.append(self.CursorCal(node,'timeend'))
        return time_cursors

    def CreateArray(self):
        final_array = self.LoadTime()
        for i,_ in enumerate(self.sequences):
            for j,__ in enumerate(_.getElementsByTagName('pulse')):
                time = self.TimeCursorSave(__)
                if j == 0:
                    t1 = 0
                elif j > 0:
                    t1 = t3
                t2 = time[0]
                t3 = time[-1]
                t_off = t2-t1
                t_on = t3-t2
                final_array[i,2*j] = t_off
                final_array[i,2*j+1] = t_on
        return final_array

class TTLData(object):
    def __init__(self, file):
        dom = minidom.parse(file)
        root = dom.documentElement
        ttl = root.getElementsByTagName('TTL')[0]
        self.arrays = []
        self.dom_arrays = ttl.getElementsByTagName('array')
        for array in self.dom_arrays:
            self.arrays.append(TTLArray(array))
        self.dom = dom
        self.root = root
        self.ttl = ttl
        self.arrays_num = len(self.arrays)
        self.unit_dict = {"s": 1.0,"ms":10**(-3),"us":10**(-6),"ns":10**(-9),"V":1.0,"mV":10**(-3),"uV":10**(-6)}
    
    def CreateArrayH5(self, filename):
        final_arrays = []
        with h5py.File(filename, "a") as f:
            f["/TTL/arrays_num"] = self.arrays_num
            for n,array in enumerate(self.arrays):
                final_array = array.CreateArray()
                final_arrays.append(final_array)
                f["/TTL/{}/data".format(n)] = final_array
                f["/TTL/{}/channels".format(n)] = array.channels
                f["/TTL/{}/index_dds".format(n)] = array.index_dds
                f["/TTL/{}/channels_num".format(n)] = array.sequence_num
                f["/TTL/{}/pulse_nums".format(n)] = array.pulse_nums 
                f["/TTL/{}/maxtime_num".format(n)] = array.maxtime_num                
                f["/TTL/{}/dma".format(n)] = array.dma
                f["/TTL/{}/time_final_delay".format(n)] = array.time_final_delay
                f["/TTL/{}/time_min_delay".format(n)] = array.time_min_delay

        return final_arrays

class SamplerArray(object):
    def __init__(self, dom_array):
        self.array = dom_array
        sequences = dom_array.getElementsByTagName('sequence')
        self.channels_nums = [] #因为总是从第7个通道开始采数据，所以只统计采样通道数
        self.updates = [] #每个sequence对应的采样率
        self.maxtime = float(0)
        self.maxtime_num = 0
        self.time_final_delay = float(0)
        self.sequences = sequences
        self.dict = dict
        self.sequence_num = len(sequences)
        self.unit_dict = {"s": 1.0,"ms":10**(-3),"us":10**(-6),"ns":10**(-9),"V":1.0,"mV":10**(-3),"uV":10**(-6)}
        self.time_final = self.CursorCal(dom_array,'timefinal')
    
    def LoadTime(self):
        final_time_ends = []
        channels_nums = []
        updates = []
        pulse_nums = []
        for _ in self.sequences:
            channels_nums.append(int(_.getAttribute("channel_num")))
            updates.append(int(_.getElementsByTagName('update')[0].firstChild.data))
            time_end = self.CursorCal(_.getElementsByTagName('pulse')[-1],'timeend')
            pulse_nums.append(len(_.getElementsByTagName('pulse')))
            final_time_ends.append(time_end)
        maxtime = max(final_time_ends)
        time_last = final_time_ends[-1]
        self.maxtime = maxtime
        self.time_final_delay = self.time_final - time_last
        self.time_min_delay = self.time_final - maxtime
        self.channels_nums = channels_nums
        self.updates = updates
        self.final_time_ends = final_time_ends
        self.pulse_nums = pulse_nums
        self.pulse_num_max = max(pulse_nums)   
        return np.zeros((self.sequence_num, self.pulse_num_max*2))

    #返回一个浮点型的标定点
    def CursorCal(self,node,string,isfloat = True):
        cursors = []
        if not node.getElementsByTagName(string):
            return cursors
        unit = node.getElementsByTagName(string)[0].getAttribute("unit")
        Node_string = node.getElementsByTagName(string)
        for _ in Node_string:
            cursors.append(float(_.firstChild.data)*self.unit_dict[unit])
        if isfloat:
            cursors = float(cursors[0])
        return cursors

    def TimeCursorSave(self,node):
        time_cursors = []
        time_cursors.append(self.CursorCal(node,'timestart'))
        time_cursors.append(self.CursorCal(node,'timeend'))
        return time_cursors

    def CreateArray(self):
        final_array_cursor = self.LoadTime()
        final_times = []
        final_sequences_mu = []
        data_nums = []
        data_nums_detail = []
        for i,_ in enumerate(self.sequences):
            final_pulses_mu = []
            data_pulses_num = []
            update = self.updates[i]
            channel_num = self.channels_nums[i]
            for j,__ in enumerate(_.getElementsByTagName('pulse')):
                time = self.TimeCursorSave(__)
                data_num = round((time[1]-time[0])*update,2)
                if i == 0:
                    if j == 0:
                        t1 = 0
                        assert data_num.is_integer()
                        final_time = np.linspace(time[0],time[1],int(data_num),endpoint=False,dtype=np.float64)
                    elif j > 0:
                        t1 = t3
                        assert data_num.is_integer()
                        final_time = np.concatenate((final_time,np.linspace(time[0],time[1],int(data_num),endpoint=False)))
                        
                elif i>0:
                    if j == 0:
                        t1 = self.final_time_ends[i-1]
                        assert data_num.is_integer()
                        final_time = np.linspace(time[0],time[1],int(data_num),endpoint=False,dtype=np.float64)
                    elif j > 0:
                        t1 = t3
                        assert data_num.is_integer()
                        final_time = np.concatenate((final_time,np.linspace(time[0],time[1],int(data_num),endpoint=False)))

                t2 = time[0]
                t3 = time[-1]
                t_off = t2-t1
                t_on = t3-t2
                final_array_cursor[i,2*j] = t_off
                final_array_cursor[i,2*j+1] = t_on
                final_pulses_mu.append(np.zeros((int(data_num),channel_num),dtype=np.int64).tolist())
                data_pulses_num.append([int(data_num), channel_num])
            final_sequences_mu.append(final_pulses_mu)
            while len(data_pulses_num) < self.pulse_num_max:
                data_pulses_num.append([np.nan,np.nan])
            data_nums_detail.append(data_pulses_num)
            final_times.append(final_time)
            data_nums.append(len(final_time))
        self.data_nums = data_nums
        self.data_nums_detail = data_nums_detail
        return final_array_cursor,final_times,final_sequences_mu

class SamplerData(object):
    def __init__(self, file):
        dom = minidom.parse(file)
        root = dom.documentElement
        sampler = root.getElementsByTagName('Sampler')[0]
        self.arrays = []
        self.dom_arrays = sampler.getElementsByTagName('array')
        for array in self.dom_arrays:
            self.arrays.append(SamplerArray(array))
        self.dom = dom
        self.root = root
        self.sampler = sampler
        self.arrays_num = len(self.arrays)
        self.unit_dict = {"s": 1.0,"ms":10**(-3),"us":10**(-6),"ns":10**(-9),"V":1.0,"mV":10**(-3),"uV":10**(-6)}
    
    def CreateArrayH5(self, filename):
        final_arrays_cursor = []
        final_arrays_times = []
        final_arrays_mu = []
        with h5py.File(filename, "a") as f:
            f["/Sampler/arrays_num"] = self.arrays_num
            for n,array in enumerate(self.arrays):
                final_array_cursor,final_times,final_sequences_mu = array.CreateArray()
                final_arrays_cursor.append(final_array_cursor)
                final_arrays_times.append(final_times)
                final_arrays_mu.append(final_sequences_mu)
                f["/Sampler/{}/final_times".format(n)] = final_times
                #f["/Sampler/{}/final_sequences_mu".format(n)] = final_sequences_mu
                f["/Sampler/{}/data_nums_detail".format(n)] = array.data_nums_detail
                f["/Sampler/{}/final_array_cursor".format(n)] = final_array_cursor
                f["/Sampler/{}/channels_nums".format(n)] = array.channels_nums
                f["/Sampler/{}/updates".format(n)] = array.updates
                f["/Sampler/{}/pulse_num_max".format(n)] = array.pulse_num_max               
                f["/Sampler/{}/pulse_nums".format(n)] = array.pulse_nums
                f["/Sampler/{}/maxtime".format(n)] = array.maxtime
                f["/Sampler/{}/sequence_num".format(n)] = array.sequence_num
                f["/Sampler/{}/data_nums".format(n)] = array.data_nums
                f["/Sampler/{}/time_final_delay".format(n)] = array.time_final_delay
                f["/Sampler/{}/time_min_delay".format(n)] = array.time_min_delay

        return final_arrays_cursor,final_arrays_times,final_arrays_mu

class DDSArray(object):
    def __init__(self, dom_array):
        self.array = dom_array
        if dom_array.getAttribute("dma") == "true":
            self.dma = True
        elif dom_array.getAttribute("dma") == "false":
            self.dma = False
        sequences = dom_array.getElementsByTagName('sequence')
        self.channels = []
        self.maxtime = float(0)
        self.maxtime_num = 0
        self.time_final_delay = float(0)
        self.sequences = sequences
        self.dict = dict
        self.sequence_num = len(sequences)
        self.unit_dict = {'dB':1.0, 'None': 1.0, "Hz": 1.0, "kHz": 10.0**3,"MHz": 10.0**6,"GHz":10.0**9,"s": 1.0,"ms":10**(-3),"us":10**(-6),"ns":10**(-9),"V":1.0,"mV":10**(-3),"uV":10**(-6)}
        self.time_final = self.CursorCal(dom_array,'timefinal')
        self.keywords = {'sweep': 1, 'frequency':0, 'phase':1, 'amplitude':2, 'phase_and_amplitude':3, 'ramp': 0, 'sin': 1, 'on': 1, 'direct_switch': 10, 'ramp_up': 11, 'bidirectional_ramp': 12, 'continuous_bidirectional_ramp': 13, 'continuous_recirculate': 14}
    
    def LoadTime(self):
        final_time_end = []
        pulse_nums = []
        channels = []
        mode = []
        for _ in self.sequences:
            channels.append(int(_.getAttribute("id")))
            if _.getAttribute("mode") == 'on':
                mode.append(True)
            else:
                mode.append(False)
            time_end = self.CursorCal(_.getElementsByTagName('pulse')[-1],'timeend')
            pulse_nums.append(len(_.getElementsByTagName('pulse')))
            final_time_end.append(time_end)
        maxtime = max(final_time_end)
        time_last = final_time_end[-1]
        self.maxtime = maxtime
        self.time_final_delay = self.time_final - time_last
        self.time_min_delay = self.time_final - maxtime 
        self.channels = channels
        self.pulse_nums = pulse_nums
        self.maxtime_num = max(pulse_nums)
        self.mode = mode 
        return np.zeros((self.sequence_num, self.maxtime_num*2))

    #返回一个浮点型的标定点
    def CursorCal(self,node,string,isfloat = True):
        cursors = []
        if not node.getElementsByTagName(string):
            return cursors
        unit = node.getElementsByTagName(string)[0].getAttribute("unit")
        Node_string = node.getElementsByTagName(string)
        for _ in Node_string:
            cursors.append(float(_.firstChild.data)*self.unit_dict[unit])
        if isfloat:
            cursors = float(cursors[0])
        return cursors

    def TimeCursorSave(self,node):
        time_cursors = []
        time_cursors.append(self.CursorCal(node,'timestart'))
        time_cursors.append(self.CursorCal(node,'timeend'))
        return time_cursors


    def CreateArray(self):
        final_array = self.LoadTime()
        FTW_POW_ASF = np.zeros((self.sequence_num, self.maxtime_num,6), dtype = np.float64)
        Phase_mode = np.zeros((self.sequence_num, self.maxtime_num), dtype = np.int16)
        keywords = self.keywords
        pulse_keywords = np.zeros((self.sequence_num, self.maxtime_num,7), dtype = np.int64)
        for i,_ in enumerate(self.sequences):
            for j,__ in enumerate(_.getElementsByTagName('pulse')):
                if keywords.get(__.getAttribute("mode")):
                    pulse_keywords[i,j,0] = 1
                    pulse_keywords[i,j,1] = keywords.get(__.getAttribute("select"), 0)
                    pulse_keywords[i,j,2] = keywords.get(__.getAttribute("shape"), 0)
                    pulse_keywords[i,j,3] = keywords.get(__.getAttribute("drg"), 1)
                    
                    times = int(__.getAttribute("times"))
                    step_num = int(__.getAttribute("step_num"))
                    node = int(__.getAttribute("node"))
                    if not times:
                        times = 1
                    if not step_num:
                        step_num = 1024
                    if not node:
                        node = 1
                    pulse_keywords[i,j,4] = times
                    pulse_keywords[i,j,5] = step_num
                    pulse_keywords[i,j,6] = node
                else:
                    pulse_keywords[i,j,0] = 0

                time = self.TimeCursorSave(__)
                if j == 0:
                    t1 = 0
                elif j > 0:
                    t1 = t3
                t2 = time[0]
                t3 = time[-1]
                t_off = t2-t1
                t_on = t3-t2
                final_array[i,2*j] = t_off
                final_array[i,2*j+1] = t_on
                FTW_POW_ASF[i,j,0] = self.CursorCal(__,'frequency')
                FTW_POW_ASF[i,j,1] = self.CursorCal(__,'phase_offset')
                FTW_POW_ASF[i,j,2] = self.CursorCal(__,'amplitude')
                FTW_POW_ASF[i,j,3] = self.CursorCal(__,'end')
                FTW_POW_ASF[i,j,4] = self.CursorCal(__,'end_amplitude')
                FTW_POW_ASF[i,j,5] = self.CursorCal(__,'att')
                Phase_mode[i,j] = int(self.CursorCal(__,'phase_mode'))
        self.FTW_POW_ASF = FTW_POW_ASF
        self.Phase_mode = Phase_mode
        self.pulse_keywords = pulse_keywords
        return final_array

class DDSData(object):
    def __init__(self, file):
        dom = minidom.parse(file)
        root = dom.documentElement
        dds = root.getElementsByTagName('DDS')[0]
        self.arrays = []
        self.dom_arrays = dds.getElementsByTagName('array')
        for array in self.dom_arrays:
            self.arrays.append(DDSArray(array))
        self.dom = dom
        self.root = root
        self.dds = dds
        self.arrays_num = len(self.arrays)
        self.unit_dict = {'None': 1, "Hz": 1, "kHz": 10**3,"MHz": 10**6,"GHz":10**9,"s": 1.0,"ms":10**(-3),"us":10**(-6),"ns":10**(-9),"V":1.0,"mV":10**(-3),"uV":10**(-6)}
    
    def CreateArrayH5(self, filename):
        final_arrays = []
        with h5py.File(filename, "a") as f:
            f["/DDS/arrays_num"] = self.arrays_num
            for n,array in enumerate(self.arrays):
                final_array = array.CreateArray()
                final_arrays.append(final_array)
                f["/DDS/{}/data".format(n)] = final_array
                f["/DDS/{}/channels".format(n)] = array.channels
                f["/DDS/{}/channels_num".format(n)] = array.sequence_num
                f["/DDS/{}/pulse_nums".format(n)] = array.pulse_nums
                f["/DDS/{}/maxtime_num".format(n)] = array.maxtime_num
                f["/DDS/{}/FTW_POW_ASF".format(n)] = array.FTW_POW_ASF
                f["/DDS/{}/Phase_mode".format(n)] = array.Phase_mode  
                f["/DDS/{}/pulse_keywords".format(n)] = array.pulse_keywords             
                f["/DDS/{}/dma".format(n)] = array.dma
                f["/DDS/{}/mode".format(n)] = array.mode
                f["/DDS/{}/time_final_delay".format(n)] = array.time_final_delay
                f["/DDS/{}/time_final".format(n)] = array.time_final
                f["/DDS/{}/time_min_delay".format(n)] = array.time_min_delay

        return final_arrays

if __name__ == "__main__":
    A = DACData("artiq-work\config.xml")
    A.check_element()
    with h5py.File("1.h5", "w") as f:
            f["/run_time"] = 1
    A.Create2DArrayH5("1.h5")
    #print((A.create_2D_array().shape)[1])
    B = TTLData("artiq-work\config.xml")
    B.CreateArrayH5("1.h5")
    C = SamplerData("artiq-work\config.xml")
    C.CreateArrayH5("1.h5")
    #a = np.array([1,2,3])
    #print(np.tile(a,(8,1)).T)
    