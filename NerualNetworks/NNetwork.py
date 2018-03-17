import numpy as np

'''
 1. 关于非线性转化方程(non-linear transformation function)

 sigmoid函数(S 曲线)用来作为activation function:

      1.1 双曲函数(tanh)
     
      1.2  逻辑函数(logistic function)
'''
def tanh(x):
    '''双曲线函数'''
    return np.tanh(x)

def tanh_deriv(x):
    '''双曲线函数导数'''   
    return 1.0 - np.tanh(x)*np.tanh(x)

def logistic(x):
    '''逻辑函数'''
    return 1/(1+np.exp(-x)) 

def logistic_deriv(x):
    '''逻辑函数导数'''
    return  logistic(x)*(1-logistic(x))

class NerualNetwork(object):
    def __init__(self,layers,activation='tanh'):
        '''      
        1，多层向前神经网络(Multilayer Feed-Forward Neural Network)  
            反向传播（Backpropagation）被使用在多层向前神经网络上
            多层向前神经网络由以下部分组成：
                输入层(input layer), 隐藏层 (hidden layers), 输出层 (output layers)
            输入层(input layer)是由训练集的实例特征向量传入
            隐藏层的个数可以是任意的，输入层有一层，输出层有一层
            每层由单元(units)组成,每个单元(unit)也可以被称作神经结点，根据生物学来源定义
            从输入层开始经过连接结点的权重(weight)传入下一层，每一层的输出是下一层的输入
            作为多层向前神经网络，理论上，如果有足够多的隐藏层(hidden layers) 和足够大的训练集, 可以模拟出任何方程

        2，设计神经网络结构；
            使用神经网络训练数据之前，必须确定神经网络的层数，以及每层单元的个数
            特征向量在被传入输入层时通常被先标准化(normalize）到0和1之间 （为了加速学习过程）
            离散型变量可以被编码成每一个输入单元对应一个特征值可能赋的值
                比如：特征值A可能取三个值（a0, a1, a2), 可以使用3个输入单元来代表A。
                    如果A=a0, 那么代表a0的单元值就取1, 其他取0；                    
                    如果A=a1, 那么代表a1de单元值就取1，其他取0，以此类推
            神经网络即可以用来做分类(classification）问题，也可以解决回归(regression)问题
                对于分类问题，如果是2类，可以用一个输出单元表示（0和1分别代表2类），如果多余2类，每一个类别用一个输出单元表示
                所以输入层的单元数量通常等于类别的数量
            没有明确的规则来设计最好有多少个隐藏层，根据实验测试和误差，以及准确度来实验并改进

        layer为list,list的维度代表有多少层神经网络，每个维度代表每层有多少个神经元
        '''
        if activation=='tanh':
            self.activation=tanh
            self.activation_deriv=tanh_deriv
        elif activation=='logistic':
            self.activation=logistic
            self.activation_deriv=logistic_deriv
        
        self.weights=[]
        for i in range(1,len(layers)-1): 
            self.weights.append((2*np.random.random((layers[i-1]+1,layers[i]+1))-1)*0.25)
            self.weights.append((2*np.random.random((layers[i]+1,layers[i+1]))-1)*0.25)

    def fit(self,x,y,learning_rate=0.2,epochs=10000):
        '''
            x为矩阵数据集合，每一行为条数据实例                
            y为分类标记结果
        '''
        x=np.atleast_2d(x)
        temp=np.ones([x.shape[0],x.shape[1]+1])
        temp[:,0:-1]=x
        x=temp
        # 分类标记（）
        y=np.array(y)

        # 通过迭代性的来处理训练集中的实例
        for k in range(epochs):
            i=np.random.randint(x.shape[0])
            # 网络的正向所有神经元
            a=[x[i]]

             # 正向更新
            for l in range(len(self.weights)):
                # 每层输入与相对应的权重进行点积
                dotvalue=np.dot(a[l],self.weights[l])
                # 将点积进行非线性转换
                trans=self.activation(dotvalue)
                # 加到各层当中
                a.append(trans)

            # 神经网络最后一层为最预测层。计算预测层与分类标记的误差
            error=y[i]-a[-1]
            # 通过误差计算反向所有神经元
            deltas=[error*self.activation_deriv(a[-1])]
            
            #反向更新
            for l in range(len(a)-2,0,-1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
            
            deltas.reverse()
            for m in range(len(self.weights)):
                layer=np.atleast_2d(a[m])
                delta=np.atleast_2d(deltas[m])
                self.weights[m]+=learning_rate*layer.T.dot(delta)
    
    def predict(self,x):
        x=np.array(x)
        temp=np.ones(x.shape[0]+1)
        temp[0:-1]=x
        a=temp
        for i in range(0,len(self.weights)):
            a=self.activation(np.dot(a,self.weights[i]))
        return a


