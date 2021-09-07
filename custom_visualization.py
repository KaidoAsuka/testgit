#testing
"""
使用Graphviz的python接口实现一个树模型的可视化
sklearn的tree.tree_plot()函数能实现对决策树的可视化，但是它的树模型和XGBoost的树模型有所不同，
而且它可视化的内容并不是我们想要的。另外，XGBoost自带的可视化功能也比较简陋。
因此，在这里实现一个对XGBoost训练结果的可视化功能，里面加入我们自定义内容。
并生成一个表格，方便后期统计。
"""

"""
用法：
1. 将XGBoost的树模型结果,以及测试集数据传入generate_node_info(), 会返回一个包含我们所需要信息的字典。
2. 将该字典传入tree_plot_custom(), 将返回树模型可视化图
3. 将该字典传入result_dataframe(), 将返回一个记录所有详细信息的表格
"""

import re

import graphviz
import numpy as np
import pandas as pd


def get_edges(row):
    """
    记录下每个节点向下连接的edge
    -----------------------------------
    Parameters :
    row: DataFrame中的一行，带有该节点信息。
    -----------------------------------
    Returns :
    edge: dict，包含edge的信息
    -----------------------------------
    """
    # 　获取edge尾节点的索引.
    yes_node = int(re.sub(pattern='[0-9]-', repl="", string=row['Yes']))
    no_node = int(re.sub(pattern='[0-9]-', repl="", string=row['No']))
    missing_node = int(re.sub(pattern='[0-9]-', repl="", string=row['Missing']))

    # 分辨missing(缺失值）被分到了哪个节点
    if missing_node == yes_node:
        RHS = [yes_node, 'Yes/Missing']
        LHS = [no_node, 'No']
    if missing_node == no_node:
        RHS = [yes_node, 'Yes']
        LHS = [no_node, 'No/Missing']

    # 记录下上面获取到的信息到字典并返回
    edge = {'RHS': RHS, 'LHS': LHS}
    return edge


def get_info(row, data, y_data):
    """
    获取以及计算当前节点的分析数据
    -----------------------------------
    Parameters :
    row: pd.Series, DataFrame中的一行，带有该节点信息。
    -----------------------------------
    Returns :
    info: dict，带有当前节点的分析数据
    -----------------------------------
    """
    node_index = row['Node']
    cur_data = data[node_index]  # 先从data里面获取当前节点的样本
    y_cur_data = y_data.loc[cur_data.index, :]  # 根据样本索引去对应的y的部分

    # 除了在原始信息中包含的Gain和Cover，我们还需要统计Counts,Positive,Negative,Ratio.
    # 有的叶节点是完全纯净的（只有一种标签的样本)，我们需要处理这种特殊情况
    # 一旦我们发现这是一个纯净的节点，我们将不会计算Ratio，并将其标记为np.nan
    compute_ratio = True  # 用一个状态变量记录我们是否需要计算Ratio
    try:
        positive = y_cur_data.value_counts()[1]  # 统计正样本
    except:
        positive = 0
        compute_ratio = False
    try:
        negative = y_cur_data.value_counts()[0]  # 统计负样本
    except:
        negative = 0
        compute_ratio = False
    # 判断是否计算Ratio
    if compute_ratio:
        ratio = positive / negative
    else:
        ratio = np.nan

    # 把获取到的信息记录到字典里并且返回
    info = {
        'Gain': row['Gain'],
        'Cover': row['Cover'],
        'Counts': y_cur_data.shape[0],
        'Positive': positive,
        'Negative': negative,
        'Ratio': ratio
    }
    return info


def generate_node_info(X_data, y_data, tree_dataframe):
    """
    生成一个包含树模型全部节点信息的字典，以便我们依据此进行作图和绘制表格
    -----------------------------------
    Parameters :
    X: pandas.DataFrame, 最上级节点处的数据,通常为测试集
    tree_dataframe: pandas.DataFrame，由XGBoost生成的树模型的信息
    -----------------------------------
    Returns :
    nodes: dict，包含作图所需要的树模型的全部节点信息
    -----------------------------------
    """
    nodes = dict()
    data = dict()  # 我们定义一个字典data，里面记录每个节点的数据，以便get_info()函数使用
    temp_info = dict()  # 定义一个存放临时信息的字典
    col_name = X_data.columns.to_list()  # 特征名的列表

    # 初始化最初的节点处的数据
    data[0] = X_data
    temp_info[0] = {'depth':0, 'path':[0]}

    # 对tree_dataframe的每一行，即每一个节点进行遍历，生成信息并记录
    for index, row in tree_dataframe.iterrows():
        node_index = row['Node']  # 当前节点索引
        # 如果节点不是叶节点：
        if row['Feature'] != 'Leaf':
            # 特征的记录格式是 ’f + index'，我们依据此找到特征名feature_name
            index = re.sub(pattern='f', repl="", string=row['Feature'])
            feature_name = col_name[int(index)]
            edges = get_edges(row)  # 获取edge
            info = get_info(row, data, y_data) # 获取分析数据
            # 将这些数据整理好记录到node里面
            node = {
                'type': 'node',
                'feature_name': feature_name,
                'split_point': row['Split'],
                'RHS_node': edges['RHS'][0],
                'RHS_annotation': edges['RHS'][1],
                'LHS_node': edges['LHS'][0],
                'LHS_annotation': edges['LHS'][1],
                'info': info,
                'depth': temp_info[node_index]['depth'],
                'path': temp_info[node_index]['path']
            }

            # 记录好信息后，我们根据上面的信息对样本进行分割，给下个节点使用
            cur_data = data[node_index]
            data[node['RHS_node']] = cur_data.loc[cur_data[node['feature_name']] < node['split_point']]
            data[node['LHS_node']] = cur_data.loc[cur_data[node['feature_name']] >= node['split_point']]

            # 分配缺失值
            missing_data = cur_data.loc[cur_data[node['feature_name']].isnull()]
            # 因为Missing(缺失值）不是在右边就是在左边，所以我们依据此进行判断
            if node['RHS_annotation'] == 'Yes/Missing':
                data[node['RHS_node']] = pd.concat([data[node['RHS_node']], missing_data])
            else:
                data[node['LHS_node']] = pd.concat([data[node['LHS_node']], missing_data])

            # 记录一些信息到temp_info，给下个节点使用
            for i in ['RHS_node','LHS_node']:
                temp_info[node[i]] = {
                    'depth': (node['depth'] + 1),
                    'path': node['path'] + [node[i]]
                }


        # 如果是叶节点，那么不需要分裂，也没有edge
        if row['Feature'] == 'Leaf':
            info = get_info(row, data, y_data)
            node = {
                'type': 'leaf',
                'info': info,
                'depth': temp_info[node_index]['depth'],
                'path': temp_info[node_index]['path']
            }

        # 　最后把当前节点信息加到树模型的全部节点信息的字典nodes里面
        nodes[node_index] = node

    return nodes


def plot_tree_custom(nodes):
    """
    绘制自定义树模型图，基于graphviz
    -----------------------------------
    Parameters :
    nodes: dict, 由generate_node_info()生成的树模型节点信息字典
    -----------------------------------
    Returns :
    graph: graphviz.graph，树模型图
    -----------------------------------
    """
    # 创建一个空图
    graph = graphviz.Digraph('tree')
    graph.attr(rankdir='LR')  # 横向

    # 样式参数
    node_style = {'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#78bceb'}
    leaf_style = {'shape': 'box', 'style': 'filled', 'fillcolor': '#e48038'}

    # nodes['info']中键值的中文字典
    chinese_dict = {
        'Gain': '增益',  # ”gain” is the average gain of splits which use the feature
        'Cover': '平均覆盖数',
        # ”cover” is the average coverage of splits which use the feature where
        # coverage is defined as the number of samples affected by the split
        'Counts': '总样本数',
        'Positive': '正样本数',  # y = 1
        'Negative': '负样本数',  # y = 0
        'Ratio': '正/负样本比'  # #{y = 1} / #{y = 0}
    }

    # 对各节点遍历
    for index, node in nodes.items():
        # 首先，我们需要整理一下需要在节点图像中呈现的内容
        rule_str = ""
        if node['type'] == 'node':  # 只检查对node类型写分裂规则, leaf不分裂
            rule_str = '规则:' + node['feature_name'] + '<' + str(node['split_point']) + r'\n'
        info_str = ""
        # 根据中文字典将英文替换，然后输出info中的内容
        for key, value in node['info'].items():
            info_str += (chinese_dict[key] + ':' + str(value) + r'\n')

        contents = rule_str + info_str

        # 然后根据节点的类型将相应的元素加到图中
        if node['type'] == 'node':
            # 添加一个节点
            graph.node(str(index), label=contents, **node_style)
            # 如果类型是node，添加edge
            graph.edge(str(index), str(node['RHS_node']), label=node['RHS_annotation'])
            graph.edge(str(index), str(node['LHS_node']), label=node['LHS_annotation'])
        if node['type'] == 'leaf':
            # 添加一个叶节点
            graph.node(str(index), label=contents, **leaf_style)

    return graph


def show_rules(nodes, node_index):
    """
    生成分裂到该节点的具体规则的字符串
    -----------------------------------
    Parameters :
    nodes: dict, 由generate_node_info()生成的树模型节点信息字典
    node_index: 当前节点索引
    -----------------------------------
    Returns :
    rules: string, 具体规则信息
    -----------------------------------
    """
    path = nodes[node_index]['path']
    rules = ""
    # 依次将路径上的规则加到rules字符串里面
    for i in range(1, len(path)):
        # 我们需要检查该节点的父节点以确认规则
        parent_node_index = path[i - 1]
        if nodes[parent_node_index]['RHS_node'] == path[i]:
            symbol = nodes[parent_node_index]['RHS_annotation']
        else:
            symbol = nodes[path[i - 1]]['LHS_annotation']
        # 得到一条具体的规则
        rule = symbol + ' : ' + nodes[parent_node_index]['feature_name'] + '<' + str(
            nodes[parent_node_index]['split_point']) + '\n'
        # 将这条规则并入到全部规则中
        rules += rule

    return rules


def result_dataframe(nodes):
    """
    生成一个Dataframe，记录我们自定义信息，以便于后期分析
    -----------------------------------
    Parameters :
    nodes: dict, 由generate_node_info()生成的树模型节点信息字典
    -----------------------------------
    Returns :
    custom_tree_data: pandas.DataFrame
    -----------------------------------
    """
    custom_tree_data = pd.DataFrame(
        columns=['ID', 'type', 'depth', 'rules', 'Gain', 'Cover', 'Counts', 'Positive', 'Negative', 'Ratio'])
    for index in range(len(nodes)):
        # 创建一行数据
        row = {
            'ID': index,
            'type': nodes[index]['type'],
            'depth': nodes[index]['depth'],
            'rules': show_rules(nodes, index)
        }
        # 剩下的信息都在 : nodes[index]['info']，用循环取出
        for key, value in nodes[index]['info'].items():
            row[key] = value
        # 将这一行数据增加到DataFrame
        custom_tree_data = custom_tree_data.append(row, ignore_index=True)

    return custom_tree_data
