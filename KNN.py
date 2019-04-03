#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:Theodore Sun
# datetime:2019/4/2 14:32
# software: PyCharm

from numpy import *
from collections import Counter
import matplotlib.pyplot as plt

"""
K-近邻（KNN）算法
《机器学习实战》
https://www.cnblogs.com/ybjourney/p/4702562.html
KNN是通过测量不同特征值之间的距离进行分类。
它的思路是：如果一个样本在特征空间中的k个最相似
(即特征空间中最邻近)的样本中的大多数属于某一个类别，
则该样本也属于这个类别，其中K通常是不大于20的整数。
KNN算法中，所选择的邻居都是已经正确分类的对象。
该方法在定类决策上只依据最邻近的一个或者几个样本的类别来
决定待分样本所属的类别。
算法的描述为：
1）计算测试数据与各个训练数据之间的距离；
2）按照距离的递增关系进行排序；
3）选取距离最小的K个点；
4）确定前K个点所在类别的出现频率；
5）返回前K个点中出现频率最高的类别作为测试数据的预测分类。
"""


# 创建数据集
def create_dataset():
    group = array([[1.0, 2.0], [1.2, 0.1], [0.1, 1.4], [0.3, 3.5]])
    label_array = ['A', 'A', 'B', 'B']
    return group, label_array


# 导入数据
def file_2_matrix(file_name):
    fr = open(file_name)
    # 读取文件内容
    contain = fr.readlines()
    count = len(contain)
    return_mat = zeros((count, 3))
    class_label_vector = []
    index = 0
    for line in contain:
        # 截取所有回车字符
        line = line.strip()
        list_from_line = line.split('\t')
        # 选取前三个元素，存储在特征矩阵中
        return_mat[index, :] = list_from_line[0 : 3]
        # 将列表的最后一列存储到向量class_label_vector中
        class_label_vector.append(list_from_line[-1])
        index += 1
    # 将列表的最后一列由字符串转化为数字，便于以后的计算
    dict_class_label = Counter(class_label_vector)
    class_label = []
    kind = list(dict_class_label)
    for item in class_label_vector:
        if item == kind[0]:
            item = 1
        elif item == kind[1]:
            item = 2
        else:
            item = 3
        class_label.append(item)
    # 将文本中的数据导入到列表中
    return return_mat, class_label


# 归一化数据，保证特征等权重
def auto_norm(data_set):
    min_values = data_set.min(0)
    max_values = data_set.max(0)
    ranges = min_values - max_values
    # 建立与dataset结构一致的矩阵
    norm_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    for i in range(1, m):
        norm_data_set[i, :] = (data_set[i, :1] - min_values) / ranges
    return norm_data_set, ranges, min_values


# 通过KNN进行分类
def classify_by_knn(test_data_set, train_data_set,
                    label, k):
    data_size = train_data_set.shape[0]
    # 计算欧式距离
    # tile：在列方向上重复test_data_set 1次，行记录上data_size次
    test_data_set_tile = tile(test_data_set, (data_size, 1))
    print("测试数据按照训练数据转换后数据为：", test_data_set_tile)
    diff_ = test_data_set_tile - train_data_set
    print("测试数据转换之后与训练数据相减结果为：", diff_)
    sq_diff = diff_ ** 2
    # 行向量分别相加，从而得到一个新的行向量
    square_dist = sum(sq_diff, axis=1)
    dist = square_dist ** 0.5
    print("欧式距离为：", dist)

    # 对距离进行排序
    # 根据元素的值从大到小对元素进行排序，返回下标
    sorted_dist_index = argsort(dist)
    class_count = {}
    for i in range(k):
        vote_label = label[sorted_dist_index[i]]
        # 对选取的K个样本所属的类别个数进行统计
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    print("样本统计信息为：", class_count)
    # 选取出现类别次数最多的类别
    max_count = 0
    classes_ = ""
    for key, value in class_count.items():
        if value > max_count:
            max_count = value
            classes_ = key
    return classes_


# 测试
def dating_test():
    rate_ = 0.10
    dating_data_mat_, dating_labels_ = file_2_matrix('')
    norm_mat_, ranges, min_values_ = auto_norm(dating_data_mat_)
    m = norm_mat_.shape[0]
    test_num = int(m * rate_)
    error_count = 0.0
    for i in range(1, test_num):
        classify_result = classify_by_knn(norm_mat_[i, :],
                                          norm_mat_[test_num:m, :],
                                          dating_labels_[test_num:m],
                                          3)

        print("分类后的结果为：", classify_result)
        print("原结果为：", dating_labels_[i])

        if classify_result != dating_labels_[i]:
            error_count += 1.0

    print("误分率为：", error_count / float(test_num))


# 预测函数
def classify_person():
    result_list = ['一点也不喜欢', '有一丢丢喜欢', '灰常喜欢']
    percent_tats = float(input("玩视频所占的时间比?"))
    miles = float(input("每年获得的飞行常客里程数?"))
    ice_cream = float(input("每周所消费的冰淇淋公升数?"))
    dating_data_mat__, dating_labels__ = file_2_matrix('')
    norm_mat__, ranges__, min_values__ = auto_norm(dating_data_mat__)
    in_arr = array([miles, percent_tats, ice_cream])
    classifier_result_ = classify_by_knn(
        (in_arr - min_values__) / ranges__,
        norm_mat__,
        dating_labels__,
        3)
    print("你对这个人的喜欢程度:",
          result_list[classifier_result_ - 1])


# 测试knn算法
def test_knn():
    # 测试方法
    data_set_, label_array_ = create_dataset()
    print("训练数据集：", data_set_)
    print("数据对应的标签信息：", label_array_)
    test_data_set_ = array([1.1, 0.3])
    output = classify_by_knn(test_data_set_,
                             data_set_,
                             label_array_, 3)
    print("测试数据为：", test_data_set_, "分类结果为：", output)


# 实际示例
def test_dating():
    # 绘图
    datingDataMat, datingLabels = file_2_matrix('')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 0],
               datingDataMat[:, 1],
               15.0 * array(datingLabels),
               15.0 * array(datingLabels))
    plt.show()

    classify_person()


test_knn()
# test_dating()

