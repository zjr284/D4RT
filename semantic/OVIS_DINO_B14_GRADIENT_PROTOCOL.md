# 第74轮：固定B14头的辅助损失梯度诊断

第73研究7270fc0已核验推送main2f7f1edb后立即接续。第71保留64000 GT锚后auxSAM源val仅小幅高GT32，第72预留COCO仍未确认收益；本轮只探索优化机制，不训练/校准/确认/部署变更，不耗LVVIS40或读新外部输出。不将梯度统计称泛化因果或GT几何准确率。

仅原80训练视频/20480源GT/142080后帧teacher特征，冻结已有GT32、extraGT、auxSAM三状态×seed0/1/2的选中checkpoint，不重构历史初始头、不选择最佳状态。每个seed重放第71全部2000步base generator(seed)与extraGT/teacher两个独立generator(20261071+seed)，FG16+BG16，索引SHA须与71逐值重放一致；只预定前200步用于分析，不按梯度/分数挑批。SAM与oracle使用同一后帧teacher索引，oracle仅改771既有纠正中的命中目标、只是诊断。

各静态checkpoint对每批四项32点CE.mean分别求原始FP32欧氏参数梯度（19994维：weight再bias）：base_GT、extra_source_GT、later_SAM、same_later_oracle。不包括AdamW动量/二阶预条件/权重衰减，不模拟训练轨迹；检查点处重放早期批次不等于那些步的真实训练梯度。解析公式delta=(softmax(logits)−onehot)/32，weight梯度delta.T@features、bias梯度delta.sum。

主描述在GT32状态：SAM/extraGT各自相对同批base的余弦、负dot批比例、梯度范数比和补充范数超过base比例；另报告先平均200梯度再算余弦，不能与每批cos均值混淆。其余两状态及oracle−SAM梯度相对差完整保存。0范数分母下限1e−12，保存全部批次，不据此选新λ/阈值/确认图。负欧氏dot仅表示局部原始梯度冲突，不证明AdamW更新冲突/泛化退化唯一原因，GTextra自身也可能因抽样冲突。

先冻结协议/源码、71已核验输入/三状态九头/源GT与纠正缓存指纹，再计算。保存200批索引/所有逐批loss/norm/dot/cos/ratio、四项平均梯度本地NPZ，仅聚合文本入Git。独立不用训练Sampler重放完整索引/前缀与目标；用autograd CE求导复算所有7200梯度并检查解析梯度（相对2e−5、绝对2e−6；统计cos/dot按范数尺度误差2e−5），全部loss/统计/平均梯度与输入指纹。浮点容差只梯度求导实现数值差，非模型选择/确认成绩容差。

保持全部三种子/反例，不新训练；核验后报告/文本快照独立提交推送，当前运行立即下一实际必要控制。默认native_gt不变，历史COCO负结果不动，不上传数据/缓存/权重/第三方源码。
