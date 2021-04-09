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
  - 3.3 小结
- 第四章 数据预处理
  - 4.1 故障数据处理
  - 4.2 仪器噪声处理
- 第五章 利用神经网络改进微波辐射计反演算法
  - 5.1 温度训练结果对比
  - 5.2 水汽密度训练结果对比
  - 5.3 相对湿度训练结果对比
---
## 代码文件树
- README.md(本文件)
- Data Analysis
    - 3.1Wyoming_year.ipynb
    - 3.2mircowave_season.ipynb
    - 3.2season_mse.ipynb
    - 3.2Wyoming_season.ipynb
    - 3.3mircowave_day.ipynb
    - 4.1mircowave_year.ipynb
    - 4.2mircowave_day.ipynb
    - conver_t_td_to_rh_sh.ipynb
    - extract_Wyoming.ipynb
    - read_filepath_module.py
- Machine Learning
  - Data_summary.ipynb(将各年份原始数据处理后汇总成对应年份csv文件)
  - Fault_data.ipynb
  - Moving_average.ipynb(滑动平均处理部分，已经集成到Data_summary.ipynb中)
  - NN_Rh.ipynb(相对湿度训练)
  - NN_Temperature.ipynb(温度训练)
  - QC.ipynb
  - regession.ipynb(Tensorflow例程)
  - search_filepath_module.py
---
## Tips
- ipynb文件用于搭建、改进模型过程中，完成后使用py文件训练
