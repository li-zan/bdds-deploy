# 桥梁病害检测与分割系统

### :triangular_flag_on_post: 更新日志：

#### :construction:开发中...

- 注意力交互分割操作撤销功能



- **[2024/5/10]**

  - :sparkles:支持`注意力框`精细分割功能实现
  - :bug:用户精细分割后同步统计结果更新
  - :zap:优化视频帧序列推理效果退化现象
  - :sparkles:新增可视化视频实时推理过程展示
  
- **[2024/5/9]**
  - :sparkles:支持用户多次添加`注意力点`来精细化分割结果
  - :zap:优化YOLO-WORLD多目标检测时漏检严重现象
- **[2024/5/6]**
  - :sparkles:新增`混凝土掉块露筋`+`锈蚀`检测与分割
  - :hammer:新增statistic模块，用于对检测与分割结果做数据统计与分析​
  - :wrench:新的配置项‘Display Label’，显示或隐藏预测标签
  - :sparkles:新增病害图片检测与分割统计面板
  - :rocket:增加相关examples样例
  - :construction:视频检测初步开发实现

- **[2024/4/28]**
  - :hammer:优化代码结构，模块化构建，新增utils/yolo_world模块

  - :sparkles:新增优化efficient_sam模块，支持tiny,small两种类型网络初始化

  - :bug:修复yolo_world相同网络参数不一致预测结果的bug
- **[2024/4/26]**	:tada:集成YOLO-World+EfficientSAM，实现`混凝土掉块露筋`检测与分割