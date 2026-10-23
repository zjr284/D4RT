# 第108轮：原60训练视频的SAM高置信背景准备

107独立发布后执行。107观察SAM256后帧BG误报73.05%，GT13.00%；源帧两者约9.77%/8.03%。本轮只准备监督与配对日程，检验“后帧仅正监督”机制所需数据，不训练分类器、不评分新验证/测试、不称已证明补BG有效。

固定原60train、每段原8张JPEG、原首帧annotation实例提示、官方SAM2.1 tiny既有权重SHA7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69，不付费/下载新权重。类别只来自首帧GT，生成不读取后帧GT；原positive-only teacher和105/106旧点不覆盖。沿现有teacher.py confident_labels的背景规则：先把每对象native mask logits以FP32 bilinear/align_corners=False缩到128，所有对象max logit<−2，再对负区作5×5最小池化（外侧补0），留下2像素安全带。只在原teacher为−1的位置添加0；原正类及首帧GT逐值保持。新SAM正结果不替换旧正类，冲突与新增/空背景统计保留。

生成过程保存每帧每对象缩放后FP32 logits、对象ID/类别与合并标签，仅本地；便于独立按规则核验，不上传掩码/图像。CUDA BF16 autocast，torch.use_deterministic_algorithms(True)，进程CUBLAS_WORKSPACE_CONFIG=:4096:8显式固定；若算子拒绝、OOM、输入/标签不符，保存失败不临时换方法。允许中断后显式--resume，只跳过已绑定指纹的完整视频，不重算/覆盖其产物。

冻结原99三seed各500步GT64/额外GT_FG32/SAM_FG32日程。seed202610108+seed独立PCG64顺序：每一步对原32SAM_FG逐点，在同aux_video和同t_src的新BG池均匀抽一个(x,y,t)，保留原FG点和类别；再从同GT_video原源帧BG候选均匀抽32点作额外GT_BG。与FG同帧配对避免仅以t_src区分正负。BG池为空则保留缺失位置/步数，不换视频、改阈值或重新抽FG，后续匹配训练资格false。

独立核验所有60源GT/旧teacher不变、新FP32 logits背景阈值与安全带、所有标签合并/原正类保持、三seed1500行全部随机日程/同帧配对和指纹。后帧GT只在独立诊断已选BG点的错误率（含重复与唯一点）时使用；不据GT筛点、修标签或选择teacher，不把诊断BG精度当自然像素precision。所有错误/空视频保留，不设GT精度选点门槛。

109只有108准备完整、独立核验和单独发布后另冻结协议才训练：GT、SAM_FG、SAM_FG+BG、等量GT_FG+BG，同192候选点前向、同原FG曝光/系数与额外BG曝光说明、固定确定性配置，不复用旧不同前向/非确定性模型作唯一因果对照。107所见是事后假设，109仍只旧60/20探索，无新泛化确认；不重评旧test20/外部40、不换默认。
