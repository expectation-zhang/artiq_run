from xml.dom import minidom
#最小化dom实现
with open('./config.xml' , 'r' , encoding = 'utf-8') as f:
    doc = minidom.parse(f)          #输入xml文件返回一个Document对象，代表整体。
                                    #如果操作字符串形式的xml，则可以使用parseString（）
root = doc.documentElement      #将doc实例成一个Element对象，一般代表了最外面的节点（根节点）。
# print(root.nodeName)            #节点名称
# print(root.nodeValue)           #节点属性值
# print(root.nodeType)            #节点类型
# print(root)
# print(root.ELEMENT_NODE)        #

#子节点的访问
dom_array = root.getElementsByTagName('DAC')[0].getElementsByTagName('array')[0]                           #getElementsByTagName('array')获得特定标签的Element对象的列表
# print(dom_array)
# print(dom_array.getAttribute('dma'))
# print(root.getElementsByTagName('update')[0].firstChild.data)                 #
# print(root.getElementsByTagName('update')[1].firstChild.data)
# sequences = dom_array.getElementsByTagName('sequence')
# print(sequences[0].getElementsByTagName('pulse')[-1].getElementsByTagName('timeend')[0].getAttribute('unit'))


#用来测试CursorCal
unit_dict = {"s": 1.0,"ms":10**(-3),"us":10**(-6),"ns":10**(-9),
             "V":1.0,"mV":10**(-3),"uV":10**(-6)}
node = dom_array.getElementsByTagName('sequence')[0].getElementsByTagName('pulse')[-1]
def curcal(node,string,isfloat=True):
    cursors = []
    if not node.getElementsByTagName(string):
        return cursors
    unit = node.getElementsByTagName(string)[0].getAttribute("unit")
    Node_string = node.getElementsByTagName(string)
    for _ in Node_string:
        cursors.append(float(_.firstChild.data) * unit_dict[unit])  # 将带单位的数据转化成浮点数
        if isfloat:  # Tr
            cursors = float(cursors[0])
        return cursors

def test_curcal():
    print(curcal(node, 'timestart'))
    print(curcal(node, 'timeend'))
    print(curcal(node, 'cursor', False))
    print(curcal(node, 'voltagestart'))
    print(curcal(node, 'voltageend'))
    print(curcal(node, 'voltagecursor'))
if __name__ == "__main__":
    test_curcal()
