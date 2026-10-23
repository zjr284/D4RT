# 第109轮：同帧SAM正/负监督的确定性匹配训练

108的60训练teacher、489万左右高置信BG像素与三seed同帧配对必须先独立核验并单独发布；实际像素总数以108冻结summary为准。本轮仍旧60train/20val，不重用旧test20/外部40作新确认、不改默认。107的后帧误报观察是事后假设来源，不是本轮已证实机制。

五臂三seed固定500步、同105/106初始、头lr1e-3/FF rank8 alpha16 LoRA lr1e-4/AdamW .01、217626可训练参数；独立源语义tail，冻结几何主路。每步同GT_video原GT64+原extra_GT_FG32+108 extra_GT_BG32共128查询，aux_video原SAM_FG32+108同帧SAM_BG32共64查询，全部臂相同192候选前向。只需重新提108新增BG位置特征，原GT/FG特征与video memory复用106的256输入缓存，明确BF16边界并核对同输入RGB重建。GT/FG选点、源帧和类别均保持；新增BG只来自SAM规则，后帧GT诊断不参与训练/过滤。

- GT：CE原GT64。
- SAM_FG：CE_GT+.25 CE_SAM_FG32。
- SAM_FG_double：CE_GT+.50 CE_SAM_FG32，控制辅助损失总权重；相同32正点加权，不冒称64独立正点。
- SAM_FGBG：CE_GT+.25 CE_SAM_FG32+.25 CE_SAM_BG32；原FG曝光/系数保留，新加BG监督。
- GTextra_FGBG：CE_GT+.25 CE_extra_GT_FG32+.25 CE_extra_GT_BG32；控制额外监督量，GT仍仅首帧。部分GT点可重复，不冒称独立样本。

GT64有效监督32000点，SAMFG/extraGTFG各16000；BG臂再16000。所有臂前向预算相同，监督量随臂明确不同。语义输入t_tgt=t_cam=t_src，base BF16/autocast、tail原参数FP32、输入BF16；视频仍8帧256，不称完整32帧权重配置。运行必须torch.use_deterministic_algorithms(True)及启动CUBLAS_WORKSPACE_CONFIG=:4096:8，记录实际backend/环境。若确定性算子拒绝则失败保存，不换精度或默默关闭开关；首次实际训练的前8步用临时镜像逐值检查参数后继续原预算，镜像不作候选。

15新模型均先冻结再评价原20val40960点及目标/相机变体；每步及时落盘loss和进度，保存首梯度、step499适配器/优化器、最终权重。原106不同前向/非确定性结果只作历史描述，不作为本轮唯一因果对照。

主SAM_FGBG−SAM_FG；必要SAM_FGBG−SAM_FG_double、SAM_FGBG−GTextra_FGBG及SAM_FGBG−GT。完整五臂原宏/前景宏、逐类、逐视频、首帧/后帧BG误报与FG正确率保留，不能只选BG下降或一个seed正就称有效。未来外测资格须FGBG对上述四对照全部三seed汇总宏差正且各对照视频等权均差正；仅允许另立新数据协议，不按探索排名改部署。

15模型×固定首两val B1/B2×原/变体共90真实接口：同输入冻结base六几何逐值相同；B1语义逐值缓存、B2同批手工参照、源分数变体逐值不变。不把原几何预测不变当真实三维精度保证。

独立核验全部新增BG特征及旧查询/缓存指纹、15条完整500步确定性训练轨迹/预算/首梯度/末优化器、15完整验证和90手工接口/人口指标/四对照/资格。若完整重放失败，保留失败先查原因，不能照搬106的限定范围声称“完整通过”；任何范围修订另文披露。代码/协议/报告/文本聚合单独提交推送，图像/teacher logits/标签/模型/预测数组不上传。
