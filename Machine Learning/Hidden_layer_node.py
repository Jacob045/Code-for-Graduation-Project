import math

# 输入节点
m = 8
# 输出节点
n = 47
# 隐藏节点数
s = math.sqrt(0.43*m*n+0.12*n**2+2.54*m+0.77*n+0.35) + 0.51
print('According to Calculate, the hidden layer node is {:.0f}'.format(s))