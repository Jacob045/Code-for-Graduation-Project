## 论文结构树（第三章及第四章）
- 第三章 榆中地区气象要素和亮温数据对比
  - 3.1 榆中地区气象要素对比
    - 3.1.1 温度数据年际变化
    - 3.1.2 温度数据季节变化
    - 3.1.3 相对湿度年际变化
    - 3.1.4 相对湿度季节变化
    - 3.1.5 温度数据和相对湿度数据日变化
  - 3.2 榆中地区亮温数据对比
    - 3.2.1 年际变化
    - 3.2.2 季节变化和日变化
    - 3.2.3 观测亮温和模拟亮温对比
  - 3.3 本章小结
- 第四章 神经网络反演温湿度廓线
  - 4.1 数据预处理
    - 仪器噪声处理
    - 质量控制
  - 4.2 神经网络训练
    - 神经网络数据集介绍
    - 神经网络隐藏层节点选择
    - 评判依据
  - 4.3 训练结果对比
    - 温度训练结果对比
    - 相对湿度结果对比
- 第五章 结论与展望

---
## 代码文件树(Code)
- README.md(本文件)
- Data Analysis
    - 3.1.1_3.1.3_Wyoming_year.ipynb
      用于气象数据年际变化特征分析
    - 3.1.2_3.1.4_season_mse.ipynb
      用于气象数据季节变化特征分析中的mse计算
    - 3.1.2_3.1.4_Wyoming_season.ipynb
      用于气象数据季节变化特征分析
    - 3.1.5 microwave_day.ipynb
      使用微波辐射计自带反演数据进行气象数据日变化特征分析
    - 3.2.1 microwave_year.ipynb
      用于观测亮温数据年际变化特征分析
    - 3.2.2 microwave_day.ipynb
      用于观测亮温数据日变化特征分析
    - 3.2.2 microwave_season.ipynb
      用于观测亮温数据季节变化分析
    - extract_Wyoming.ipynb
      从dat文件中提取探空数据，输出CSV文件
    - read_filepath_module.py
      用于搜索路径下包含相应字符串的文件（可用于二级文件夹）
- Machine Learning
  - Basic_Data(Data_summary.ipynb结果输出)
  - model
  - model_Rh 相对湿度训练模型
  - model_Tem 温度训练模型
  - training_1
  - Compare_bt.ipynb
  - Compare.py
  - Data_summary.ipynb(将各年份原始数据处理后汇总成对应年份csv文件)
  - Extract_Inversion_Rh.ipynb(提取地基微波辐射计反演相对湿度数据)
  - Extract_Inversion_Tem.ipynb(提取地基微波辐射计反演温度数据)
  - Extract_Observation_bt.ipynb(提取地基微波辐射计观测亮温数据)
  - Extract_Observation_Rh.ipynb(提取观测相对湿度数据)
  - Extract_Observation_Tem.ipynb(提取观测温度数据)
  - Extract_Simulated_bt.ipynb(提取模拟亮温数据)
  - Fault_data.ipynb
  - Hidden_layer_node.py(隐藏层节点计算公式)
  - Moving_average.ipynb(滑动平均处理部分，已经集成到Data_summary.ipynb中)
  - NN_bt.ipynb(亮温训练模型--未使用)
  - NN_Rh.ipynb(相对湿度训练模型)
  - NN_Rh0504.ipynb(相对湿度训练0504模型)
  - NN_Rh_Compare.ipynb(相对湿度结果对比)
  - NN_Tem.ipynb(温度训练模型)
  - NN_Tem0429.ipynb(温度训练0429模型)
  - NN_Tem0429.py(温度训练0429模型)
  - NN_Tem_Compare.ipynb(温度结果对比)
  - QC.ipynb(质量控制)
  - regession.ipynb(Tensorflow例程)
  - search_filepath_module.py
    用于搜索路径下包含相应字符串的文件（可用于二级文件夹）
- Trash
- map.ipynb(绘制第二章两个气象站之间的位置关系图)
- requirements.txt(所有代码运行所需要的库)
---
## Tips
- 同名ipynb文件和py文件中，ipynb文件用于搭建、改进模型过程中，完成后使用py文件训练
- 所有代码在Python3.7环境行运行，库配置文件保存在requirements.txt文件中
