# 第二十六轮：整批几何视频编码与语义tokens共同复用

第二十五轮推送后立即开始，不训练/重校准、不消费新视频或重评最终集。继承冻结原JPEG语义缓存，新增有界整批几何memory缓存，调用原模型encode_video/video与decode_queries/query公共接口，不重写查询解码器或几何头。

几何缓存键包括完整video的形状、精度、设备、逐字节内容，aspect_ratio形状/精度/内容和原几何参数版本/存储；必须区分B=1/B=2，不能把单视频编码拼为双视频。每次继续来源内容指纹与几何RGB核对；query、目标/相机时间不进入视频编码键，因为原encode_video只读video/aspect_ratio，解码始终重新使用原query。两批次LRU有界，可clear。骨干须eval；语义分类分数不缓存，三头重载继续重新计算。

必要替身单元覆盖整批形状、比例/图像/正常参数版本改变失效、query改变复用memory而解码变化、来源拒绝、LRU/clear。真实固定第25相同3段完整2048查询三种子、B=2首两段；与原无缓存native接口同输入逐值比较六几何与语义分数/类别；改变tgt/cam、source、256子集、aspect_ratio [B]仍逐值比较，记录全部差异和编码次数。改变比例必须重新编码，不因比例token可选而省键。

固定每案例10对交替计时：第25仅语义缓存 versus本轮共同缓存。分别清空/预热，所有记录保存，CUDA同步、峰值显存及首调用成本报告，不择优。第25路径仍每次原几何前向，本轮只省重复视频编码，不能据此声称生产吞吐或新语义性能。参数文件、像素、源路径、模型选择均冻结，真实权重/OVIS图像不修改。核验/复盘/独立提交推送后继续必要实验；入口`.venv/bin/python -m semantic.dino_geometry_cache_probe`，产物`semantic_artifacts/dino_geometry_cache_probe/`。
