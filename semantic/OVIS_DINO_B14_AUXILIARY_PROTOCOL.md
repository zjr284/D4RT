# 第71轮：保留B14源GT锚的同预算辅助监督

第70 f9ea023已提交推送后立即接续，仅原80train/20val，不重评COCO、不消费COCOreserve80/LVVISreserve40。此前弱SAM的锚访问约64k且随机每批，单GT32 .830191更强；本轮保留原GT32每步16FG+16BG索引/64000锚访问和CE系数，再比较追加监督，固定λ1不搜网格。

复用gt32/weak旧三头各组。只新增extra_gt/aux_sam/aux_oracle三方法×seed0/1/2=9头，2000步AdamW .001/.01、768维19994参数。每步基准GT32为base，base generator(seed)逐批哈希必须与第58 B14_GT32一致；额外32点（16FG16BG）用独立generator(seed20261071+seed)。extra_gt额外从同源GT池抽样；aux_sam额外从原teacher t_src>0池（142080查询）依teacher硬标签FG/BG抽样；aux_oracle与aux_sam额外索引逐批完全相同，仅改为第21已验证GT纠正目标。原GTsource为20480查询，全部特征/坐标/标签固定、无新骨干/教师/GT筛点。

目标固定L=CE(base_GT).mean+CE(extra_target).mean，各项分母32/系数1；base查询/目标曝光/项系数与旧GT32保持，梯度数值因模型状态不同不能说逐批相同。每新头64基准+额外行/步、总128000抽样曝光；extra_gt源锚总128000，aux两组源锚64000+后帧64000，不混称所有GT曝光精确相等。三组计算曝光相同、前两组首帧人工GT预算相同；补充坐标/时间/标签分布不同，只归于完整追加方案。aux_oracle额外后帧GT仅诊断，不可部署/选为候选，实际target FG/BG不保证各半。

原raw val宏每50步选checkpoint、同分早，三种子共享BG偏移原grid−8..8步.25/同分abs小负优先。主aux_sam−extra_gt（相同总计算/首帧GT预算），次aux_oracle−aux_sam和aux_sam−gt32；完整保存三种子/负结果。固定可行候选顺序gt32/extra_gt/aux_sam/weak，oracle排除；另比较原全局source-val候选class_video .831876，不按COCO最高改选/部署。为未来独立教师问题，source val在native weak和aux_sam两种教师方案中选较高者（同分native优先），匹配对照分别B14_GT256或extra_gt，先冻结后才能用新确认，当前无确认评分。

直接冻结69/70已核验80特征/GT纠正/锚、21前后必要旧头/报告、代码与协议；保存base/supp索引SHA、类别/时间/actual目标直方图、总曝光/GT锚、最佳checkpoint。独立9采样重放/base哈希/后帧SAM-oracle共享、15val输出/校准/非oracle候选/teacher选择/3对照与全部旧指纹；通过后报告/自有文本快照独立提交推送，当前运行立即下一必要研究。数据/缓存/权重不入Git，默认native_gt保持。
