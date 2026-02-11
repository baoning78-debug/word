

# ndarray

## 特性

<img src="https://api2.mubu.com/v3/document_image/30043502_27e64c06-0eff-45a3-b56b-957550824525.png?" alt="img" style="zoom:80%;" />

### 多维性

```python
import numpy as np
```

```python
arr = np.array(5) #创建0维的ndarray数组
print(arr)
print('arr的维度：',arr.ndim) #number of dimensions
```

```python
arr = np.array([1,2,3]) #创建1维的ndarray数组
print(arr)
print('arr的维度：',arr.ndim) #number of dimensions
```

```python
arr = np.array([[1,2,3], [4,5,6]]) #创建1维的ndarray数组
print(arr)
print('arr的维度：',arr.ndim) #number of dimensions
```



### 同质性

```python
arr = np.array([1,'hello']) #不同的数据类型会被转换成相同的数据类型
print(arr)
```

```python
arr = np.array([1,2.5]) 
print(arr)
```

### 属性

<img src="https://api2.mubu.com/v3/document_image/30043502_da9422ae-051c-468a-e8cc-4b57fba5f835.png?" alt="img" style="zoom: 67%;" />

<img src="https://api2.mubu.com/v3/document_image/30043502_b7b13f87-3720-407c-b4d7-009b9f24edeb.png?" alt="img" style="zoom: 67%;" />

```python
arr.shape #数组的形状

arr.ndim #数组的维度

arr.size #总元素个数

arr.dtype #元素类型

arr.T #转置
```

## ndarray的创建 

<img src="https://api2.mubu.com/v3/document_image/30043502_f49c71a7-0086-4450-b7c9-0a1ff680052f.png?" alt="img" style="zoom:80%;" />

<img src="https://api2.mubu.com/v3/document_image/30043502_79818446-cf0f-4cfd-eb92-80bc91606eaf.png?" alt="img" style="zoom:80%;" />

```python
arr = np.array([1,2,3])
print(arr.ndim) #属性
print(arr)
```

```python
list1 = [4,5,6]
arr = np.array(list1,dtype=np.float64)
print(arr.ndim)
print(arr)
```

```python
#copy
arr1 = np.copy(arr)
print(arr1)
arr1[0] = 8
print(arr1)
print(arr)
```

<img src="https://api2.mubu.com/v3/document_image/30043502_9c70ff19-321f-4e61-cd13-34f32c24ba44.png?" alt="img" style="zoom:80%;" />

```python
#预定义形状
#全0
arr = np.zeros((2,3),dtype=int)
print(arr)
print(arr.dtype)
```

```python
#全0
arr = np.zeros((2,),dtype=int)
print(arr)
print(arr.dtype)
```

```python
#全1
arr = np.ones((2,3))
print(arr)
print(arr.dtype)
```

```python
#未初始化
arr = np.empty((4,3)) #随机
print(arr)
```

```python
arr = np.full((3,4),2025)
print(arr)
print(arr.ndim)
```

```python
arr1 = np.zeros_like(arr)
print(arr1)
arr2 = np.empty_like(arr)
print(arr2)
arr3 = np.ones_like(arr)
print(arr3)
arr4 = np.full_like(arr,2026)
print(arr4)
```

### 数列

```python
#等差数列 2 4 6 8
arr = np.arange(1,15,1) #start end step(步长)
print(arr)
```

```python
#等间隔数列
arr = np.linspace(1,10,5) #start end 分成几份  等间隔
print(arr)
```

```python
arr = np.linspace(0,100,5,dtype=int) #start end 分成几份  等间隔
print(arr)
arr = np.arange(0,101,25) #start end step(步长)
print(arr)
```

```python
#对数间隔数列
arr = np.logspace(0,4,3,base=2) #先等间隔分成3份，再以base为底算幂（默认base=10）
print(arr)
```

### 特殊矩阵

```python
#单位矩阵：主对角线上的数字为1，其余的数字为0
arr = np.eye(3,4,dtype = int)
print(arr)
```

```python
#对角矩阵：主对角线上非零的数字，其他数字为0
arr = np.diag([1,2,3])
print(arr)
```

### 随机数组的生成

```python
#生成0到1之间的随机浮点数(均匀分布)
arr = np.random.rand(2,3)
print(arr)
```

```python
#生成指定范围区间的随机浮点数
arr = np.random.uniform(3,6,(2,3)) 
print(arr)
```

```python
#生成指定范围区间的随机整数
arr = np.random.randint(3,30,(2,3)) 
print(arr)
```

```python
#生成随机分布（正态分布）
arr = np.random.randn(2,3)
print(arr)
```

```python
# 设置随机种子
np.random.seed(20) #种子一样，每次生成固定一致
arr = np.random.randint(1,10,(2,5))
print(arr)
```

## ndarray的数据类型
布尔类型bool

整数类型int uint

浮点数float

复数complex

<img src="https://api2.mubu.com/v3/document_image/30043502_9dc63cc4-5f56-4ba5-decf-b4344449bcd8.png?" alt="img" style="zoom:67%;" />

```python
arr = np.array([1,0,2,0],dtype='bool')
print(arr)
arr = np.array([1,0,1,0],dtype= np.bool_)
print(arr)
```

```python
arr = np.array([1,0,127,0],dtype= np.int8)
print(arr)
```

## 索引与切片——查询数据

<img src="https://api2.mubu.com/v3/document_image/30043502_11213442-0eb2-44e4-db18-7ae7c1bf8415.png?" alt="img" style="zoom:80%;" />

```python
# 一维数组的索引与切片
arr = np.random.randint(1,100,20)
print(arr)
print(arr[10])
print(arr[:]) #获取全部的数据
print(arr[2:5]) #获取第3个到第5个（左包右不包）
print(arr[slice(2,15,3)]) #start end step
print(arr[(arr>10) & (arr<70)]) #布尔索引
```

```python
# 二维数组的索引和切片
arr = np.random.randint(1,100,(4,6))
print(arr)
print(arr[1,3]) #索引

print(arr[:,:])
print(arr[1,:])
print(arr[1,2:5])

print(arr[arr>50]) #布尔索引，返回一维数组
print(arr[2] [arr[2]>50])#过滤行数据,省略列
print(arr[:,2][arr[:,2]>50])#过滤列数据
```

## 5.ndarray的运算

```python
#算数运算
#一维数组
a = np.array([1,2,3])
b = np.array([4,5,6])
print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a ** 2)
```

```python
#原生python
c = [1,2,3]
d = [4,5,6]
print(c + d) #拼接
for i in range(len(c)):
    d[i]=d[i]+c[i]
print(d)
```

```python
#二维矩阵
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = np.array([[4,5,6],[7,8,9],[1,2,3]])
print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a ** 2)

print(a@b) #矩阵乘法
```

```python
#数组与标量
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a+3)
print(a*3)
```

```python
#广播机制:1.获取形状 2.是否可广播
#同一维度：相同，1
a = np.array([1,2,3]) #1*3
b = np.array([[4],[5],[6]]) #3*1
print(a*b)
print(a+b)
print(b-a)
```

# numpy中的常用函数
![image-20260208112529902](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260208112529902.png)

## 1.基本数学函数

```python
#计算平方根
print(np.sqrt(9))
print(np.sqrt([1,4,9])) #得到浮点数

arr = np.array([1,25,81])
print(np.sqrt(arr))
```

```python
#计算指数e^x
print(np.exp(1))
```

```python
#计算自然对数 lnx
print(np.log(2.718281828459045))
```

```python
 #计算正弦值、余弦值
print(np.sin(np.pi/2))
print(np.cos(np.pi))
```

```python
#计算a的b次幂
arr = np.array([-1,1,2,-3])
print(np.power(arr,2))
```

```python
#四舍五入
print(np.round([3.2,4.5,4.52,8.1,9.6]))
```

```python
#向上取整，向下取整
arr = np.array([1.6,25.1,81.7])
print(np.ceil(arr)) #向上取整
print(np.floor(arr)) #向下取整
```

```python
#检测缺失值
np.isnan([1,2,np.nan,3])
```

## 2.统计函数
求和  计算平均值  计算中位数  标准差  方差

查找最大值 最小值

计算分位数 累积和 累积差

```python
arr = np.random.randint(1,20,8)
print(arr)
```

```python
#求和
print(np.sum([1,2,3]))
```

```python
#计算平均值
print(np.mean([1,2,4]))
```

```python
#计算中位数 
#奇数个：排序后中间的数值
#偶数个：中间两个数的平均值
print(np.median([1,2,4]))
print(np.median([1,2,5,80]))
```

```python
#计算标准差，方差
print(np.var([1,2,3]))
print(np.std([1,2,3]))
```

```python
#计算最大值，最小值
print(arr)
print(np.max(arr),np.argmax(arr)) #最大值和索引
print(np.min(arr),np.argmin(arr))
```

```python
#分位数

#中位数
print(np.median([1,2,3]))
print(np.median([1,2,3,4]))
np.random.seed(0)
arr = np.random.randint(0,100,4)
print(arr)
print(np.median(arr))

print(np.percentile(arr,0))
print(np.percentile(arr,25))
print(np.percentile(arr,50))
print(np.percentile(arr,75))
print(np.percentile(arr,100))
```

```python
#累积和，累积积
arr = np.array([1,2,3])
print(np.sum(arr))
print(np.cumsum(arr))
print(np.cumprod(arr))
```

## 比较函数


比较是否大于，小于，等于

逻辑与、或、非

检查数组中是否有一个True,是否所有的都为True

```python
#是否大于
print(np.greater([3,4,5,6,7],4))
#是否小于
print(np.less([3,4,5,6,7,8],4))
#是否等于
print(np.equal([3,4,5,6,7,8],4))
print(np.equal([3,4,5],[4,4,4]))#相同形状
```

```python
#逻辑与
print(np.logical_and([1,0],[1,1]))
#逻辑或
print(np.logical_or([0,0],[1,0]))
#逻辑非
print(np.logical_not([1,0]))
```

```python
#检查元素是否至少有一个元素为true
print(np.any([0,1,0,0,0]))
#检查是否全部元素为true
print(np.all([1,1,0,0,0]))
```

```python
#自定义条件
#print(np.where(条件,符合条件,不符合条件的))
arr = np.array([1,2,3,4,5])
print(np.where(arr>3,arr,0))
print(np.where(arr>3,1,0))
```

```python
score = np.random.randint(50,100,20)
print(score)
print(np.where(score>=60,'及格','不及格'))
```

```python
print(np.where(
    score<60,'不及格',np.where(
        score<80,'良好','优秀'
    )
))
```

```python
#np.select(条件，返回的结果)
print(np.select([score>80,(score>60)&(score<80),score<60],['优秀','良好','不及格']))
```

## 排序函数

```python
np.random.seed(0)
arr = np.random.randint(1,100,20)
print(arr)

arr.sort()      #arr改变
print(arr)

print(np.sort(arr)) #原始数组没有改变
print(arr)

print(np.argsort(arr)) #索引
```

```python
#去重函数+排序
print(np.unique(arr))
```

```python
#数组的拼接
arr1 = np.array([1,2,3])
arr2 = np.array([4,5,6])
print(arr1+arr2)
print(np.concatenate((arr1,arr2)))
```

```python
#数组的分割
print(np.split(arr,5))         #必须能等分
print(np.split(arr,[6,12,18])) #给出切割位置
```

```python
#调整数组形状
print(np.reshape(arr,[4,5]))   #必须能等分
```

# 矩阵运算

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260210110400063.png" alt="image-20260210110400063" style="zoom:80%;" />

### 1.矩阵的加减

同维度的矩阵才能进行加减运算

对应位置元素逐一相加减

```python
# 定义两个同维度矩阵
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 矩阵加法
add_result = A + B
# 矩阵减法
sub_result = A - B

print("矩阵加法结果：\n", add_result)
print("矩阵减法结果：\n", sub_result)
```



### 2.矩阵数乘（标量乘法）

标量（单个数值）与矩阵的每个元素相乘。

```python
A = np.array([[1, 2], [3, 4]])
scalar = 3

# 数乘运算
mul_scalar = A * scalar  # 或 np.multiply(A, scalar)

print("矩阵数乘结果：\n", mul_scalar)
```

### 3.矩阵乘法（点乘）线性代数矩阵乘

第一个矩阵的列数必须等于第二个矩阵的行数；
结果矩阵的维度为：第一个矩阵行数 × 第二个矩阵列数；
计算方式：结果矩阵第 i 行第 j 列元素 = 第一个矩阵第 i 行 × 第二个矩阵第 j 列 对应元素相乘后求和。

```python
# A: 2行3列，B: 3行2列（满足列数=行数）
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8], [9, 10], [11, 12]])

# 矩阵乘法（两种等价方式）
mul_matrix = np.dot(A, B)  # 方式1：np.dot
# mul_matrix = A @ B       # 方式2：@ 运算符（Python 3.5+）

print("矩阵乘法结果：\n", mul_matrix)
```

### 4.矩阵对应元素相乘（哈达玛积）

仅同维度矩阵可运算，对应位置元素直接相乘（区别于矩阵点乘）。

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 对应元素相乘
hadamard_product = A * B  # 或 np.multiply(A, B)

print("哈达玛积结果：\n", hadamard_product)
```

### 5.numpy广播机制（先扩展再逐元素相乘）

NumPy 允许不同形状的数组进行算术运算（加减乘除），前提是它们的维度满足 “广播兼容”，核心规则：
将同一维度进行比较，维度大小要么相等，要么其中一个是 1；
维度为 1 的数组会被 “拉伸”（复制）到与另一个数组相同的维度大小，然后执行逐元素运算（而非矩阵乘法）。

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260210104646180.png" alt="image-20260210104646180" style="zoom: 67%;" />

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260210105031427.png" alt="image-20260210105031427" style="zoom:67%;" />

### 6.向量的外积

无论输入形状如何，先将两个向量展平为一维数组（b 展平为`[-1,0,1]`，c 展平为`[2,1,3]`）；

外积结果是一个矩阵，维度为「第一个向量长度 × 第二个向量长度」（这里是 3×3）；

结果矩阵第 i 行第 j 列元素 = 第一个向量第 i 个元素 × 第二个向量第 j 个元素。

```python
import numpy as np
b = np.array([[-1],[0],[1]])  # 3×1 列向量（矩阵）
c = np.array([[2],[1],[3]])  # 3×1 列向量（矩阵）
e = np.array([[2]])          # 1×1 标量矩阵

outer_product = np.outer(b, c)
print(outer_product)
# 输出：
# [[-2 -1 -3]
#  [ 0  0  0]
#  [ 2  1  3]]
```

### 7.矩阵转置

矩阵的行和列互换（i 行 j 列元素变为 j 行 i 列）

```python
A = np.array([[1, 2, 3], [4, 5, 6]])

# 转置运算（两种方式）
transpose_A = A.T        # 方式1：.T 属性
# transpose_A = np.transpose(A)  # 方式2：np.transpose

print("原矩阵：\n", A)
print("转置矩阵：\n", transpose_A)
```

### 8. 逆矩阵

仅方阵（行数 = 列数）可能有逆矩阵；
逆矩阵满足：A × A⁻¹ = A⁻¹ × A = 单位矩阵；
若矩阵的行列式为 0（奇异矩阵），则无逆矩阵。

```python
# 定义可逆方阵
A = np.array([[1, 2], [3, 4]])

# 计算逆矩阵
inv_A = np.linalg.inv(A)

# 验证：A × 逆矩阵 = 单位矩阵（浮点精度下接近0的数会显示为0）
verify = np.dot(A, inv_A)
print("逆矩阵：\n", inv_A)
print("验证（A×A⁻¹）：\n", verify)

# 处理奇异矩阵（行列式为0）
singular_A = np.array([[1, 2], [2, 4]])
try:
    np.linalg.inv(singular_A)
except np.linalg.LinAlgError as e:
    print("错误：", e)  # 输出奇异矩阵无逆矩阵的报错
```

### 9. 矩阵行列式

仅方阵有行列式，是矩阵的一个标量特征，逆矩阵存在的前提是行列式≠0。

```python
A = np.array([[1, 2], [3, 4]])

# 计算行列式
det_A = np.linalg.det(A)
print("矩阵行列式：", det_A)  # 结果应为 -2.0
```

### 10. 矩阵的迹（迹运算）

方阵的迹是主对角线（左上到右下）元素的和。

```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 计算迹
trace_A = np.trace(A)
print("矩阵的迹：", trace_A)  # 1+5+9=15
```

### 11. 矩阵的秩

矩阵的秩是其线性无关的行 / 列的最大数，反映矩阵的 “有效维度”。

```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 计算秩
rank_A = np.linalg.matrix_rank(A)
print("矩阵的秩：", rank_A)  # 该矩阵秩为2（第三行可由前两行线性表示）
```

### 12.方阵求幂

仅方阵可求幂，A^n 表示 A 自乘 n 次（n 为正整数）。

```python
A = np.array([[1, 2], [3, 4]])

# 矩阵求幂（A² = A×A）
power_A = np.linalg.matrix_power(A, 2)
print("矩阵平方：\n", power_A)
```

### 13.矩阵范数

范数是矩阵的 “长度” 度量，常用的有 L1、L2、无穷范数。

```python
A = np.array([[1, -2], [3, -4]])

# 计算不同范数
l1_norm = np.linalg.norm(A, ord=1)    # L1范数（列和最大值）
l2_norm = np.linalg.norm(A, ord=2)    # L2范数（最大奇异值）
inf_norm = np.linalg.norm(A, ord=np.inf)  # 无穷范数（行和最大值）

print("L1范数：", l1_norm)
print("L2范数：", l2_norm)
print("无穷范数：", inf_norm)
```

### 14.方阵的特征值，特征向量

核心定义：对于 n 阶方阵 A，如果存在数 λ（特征值）和非零 n 维向量 v（特征向量），满足 A·v = λ·v，则 λ 是 A 的特征值，v 是对应 λ 的特征向量。
注意：仅适用于方阵（行数 = 列数），非方阵无法计算特征值 / 特征向量。

```python
# 先构造一个3×3方阵（比如B）
B = np.array([[3,2,0],[1,0,1],[-2,0,4]])
eigenvalues, eigenvectors = np.linalg.eig(B)
print("特征值：", eigenvalues)       # 3个标量（B是3×3）
print("特征向量矩阵：\n", eigenvectors)  # 3×3矩阵，每一列是对应特征值的特征向量
```

