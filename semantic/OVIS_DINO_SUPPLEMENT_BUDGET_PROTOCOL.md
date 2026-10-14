# 第三十一轮：追加GT与传播标签的相同查询预算对照

第三十轮完成推送后立即启动。固定原JPEG448特征/80训练20验证/骨干/2000步/10010参数，不训练编码器、不评价确认、不消费余27段、不改native_gt部署。复用第30 source_gt/prop_quarter/prop_one九头与偏移，新增gt_quarter/gt_one共六头seed0/1/2。

每步沿第30相同source generator seed=s抽256源GT（128fg+128bg），追加沿seed20261030+s再抽256源GT。第30追加的是后帧教师，本轮追加源GT，因此总查询曝光同为1024000、源基础曝光512000、更新数和损失权重匹配；追加组的坐标/时刻/特征/标签分布改变，不能把差异称纯标签噪声效应或同一批坐标GT纠正。追加GT不新增独立训练视频；两个GT sampler独立generator、从同已有源集合采样。

loss=(meanCE_source+λ*meanCE_extra)/(1+λ)，λ=.25/1固定，同AdamW .001/.01与50步原始验证宏F1 checkpoint规则。新方案用历史共享背景偏移网格验证校准，旧方案不重训或重校准。各组类/时间直方图、索引哈希须保存并重放；基础GT曝光须与第30和旧纯GT一致。

主对照是prop_quarter−gt_quarter、prop_one−gt_one，另列所有相对纯GT。只对反复使用的验证集探索，不能据结果宣称泛化、把旧/最终40再算新确认或上线新头。15份验证输出/六新训练轨迹独立核验，保存负结果/文本快照，白名单独立提交推送后立即下一必要研究。入口`.venv/bin/python -m semantic.dino_supplement_budget_control`，产物`semantic_artifacts/dino_supplement_budget_control/`。
