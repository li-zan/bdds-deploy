# 桥梁病害检测与分割系统

### :triangular_flag_on_post: 更新日志：

- **[2024/4/28]**

  - :hammer:优化代码结构，模块化构建，新增utils/yolo_world模块

  - :sparkles:新增优化efficient_sam模块，支持tiny,small两种类型网络初始化
  - :bug:修复yolo_world相同网络参数不一致预测结果的bug

- **[2024/4/26]**	:tada:集成YOLO-World+EfficientSAM，实现`混凝土掉块露筋`检测与分割