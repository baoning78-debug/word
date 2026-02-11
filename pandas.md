# pandas

<img src="https://api2.mubu.com/v3/document_image/30043502_03f416a6-eb67-4882-baf5-1314ae7d78c3.png?" alt="img" style="zoom:80%;" />

## series的创建

```python
import pandas as pd
s = pd.Series([1,2,3,4,5])
print(s)
#自定义索引
s = pd.Series([10,2,3,4,5],index=['A','B','C','D','E'])
print(s)
#定义name
s = pd.Series([10,2,3,4,5],index=['A','B','C','D','E'],name = '月份')
print(s)
s1 = pd.Series(s,index=["A","C"])
print(s1)
```

```python
#通过字典来创建
s = pd.Series({"a":1,"b":2,"c":3,"d":4,"e":5})
print(s)
s1 = pd.Series(s,index=["a","c"])
print(s1)
```



## series的属性

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260209193303866.png" alt="image-20260209193303866" style="zoom:80%;" />

```python
print(s.index)
print(s.values)
print(s.shape,s.ndim,s.size)
s.name = 'test'
print(s.dtype,s.name)
print(s.loc['a':'c'])#显式索引，按标签
print(s.iloc[1:3])#隐式索引，按位置
print(s.at['a'])#不支持切片,只能找到精确值
print(s.iat[0])#隐式，按位置
```

## series的常见用法

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260209200613793.png" alt="image-20260209200613793" style="zoom:80%;" />

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260209200825005.png" alt="image-20260209200825005" style="zoom:80%;" />

```python
#访问数据
#print(s[0])#不推荐使用，易混淆
print(s['c'])
print(s[s<3])
s['f']=6
print(s.head())#默认打印前5行，填数字可控制
print(s.tail())#默认打印后5行，填数字可控制
print(s.tail(1))
```

```python
#常见函数
import numpy as np
s = pd.Series([10,2,np.nan,None,3,4,5],index=['A','B','C','D','E','F','G'],name='data')
print(s)
```

```python
s.head() #默认打印前5行，填数字可控制
s.tail() #默认打印后5行，填数字可控制
```

```python
#查看所有的描述性信息
s.describe()
```

```python
#获取元素个数
s.count()
```

```python
#获取索引
print(s.keys())
print(s.index)
```

```python
#检查缺失值
print(s.isna())
s.isna()
```

```python
#查看元素是否在列表里
s.isin([4,5,6])
```

```python
#统计方法
s.describe()
print(s.mean())#平均值
print(s.sum())#总和
print(s.std())#标准差
print(s.var())#方差
print(s.max())#最大值
print(s.min())#最小值
print(s.median())#中位数
```

```python
s.sort_values()
s.quantile(0.25)#分位数
```

```python
s['H'] = 4
s.mode()#众数,按出现频率(次数)
```

```python
print(s.value_counts())#每个元素的计数
```

```python
#去重
s.drop_duplicates() #返回series
s.unique()    #返回列表
s.nunique()   #去重后的元素个数
```

```python
#排序   值，索引
s.sort_index()     #按索引排序
s.sort_values()    #按值排序
```

