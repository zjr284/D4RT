# 第47轮：一次原失败范围的HTTP状态诊断

承接第46核验/提交/推送，固定原936262206–936262235范围、同官方val ZIP总大小/版本、原清单/模型。一次新匿名公开确认GET加一次30字节Range GET，HTTPS/TLS保留，不重试。总实际body<=1MiB，错误正文只读最多4096字节，保存本地SHA及纯类别标志，不输出临时URL、cookie或正文。

记录响应status、Content-Range/Length/Encoding/Type、Last-Modified/ETag、Retry-After和是否包含quota/login文本标志；这些标志只是错误文本检测，不替代官方原因解释。206需字节长度30、PK local header、范围/版本全部相符才认为当前原范围可读；200/fullbody拒绝。HTTPError保留具体status，而非一律transport_error。

本轮不取新图像或评分，不替换候选。不对403/认证/配额执行绕过/更换账户/重复请求。若206恢复，只能说明此刻单请求成功，不能确定第46历史状态码或根因；随后预注册低请求次数连续档案获取控制，单独固定明确2GiB上限和完整性条件。若权限/配额明确阻塞，保存并转向其他合法来源，不能把资源准备当模型进展。
