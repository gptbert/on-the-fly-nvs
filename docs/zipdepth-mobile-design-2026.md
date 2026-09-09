# ZipDepth 手机端推理与重建接入技术方案

核实日期：2026-09-09。状态：技术设计，尚未下载权重、执行模型转换或完成真机验证。

上游源码固定为 [91f3fd21e131641f51e8d35736d1958350180e3a](https://github.com/fabiotosi92/ZipDepth/tree/91f3fd21e131641f51e8d35736d1958350180e3a)，该提交日期为 2026-08-22。本地参考提交为 8b377c9；工作区正在发生 R3 相关修改，下面以接口和函数为定位依据，不把已有选型文档中的“默认模型”描述视为当前运行状态。本次仅新增此方案。

## 1. 技术决策

采用 **ZipDepth-base 的 NPU 版本作为手机端相对逆深度候选后端**。先在服务器建立正确性基准，再分别导出 iOS Core ML 和 Android LiteRT 模型。第一期手机负责采集、深度预览和关键帧深度生成，现有 CUDA 服务器负责位姿估计或接收外部位姿，以及 3D Gaussian Splatting 优化。

ZipDepth 的轻量化解决的是单帧深度推理成本。它不输出相机位姿，不能独立完成 SLAM，不提供已标定的米制深度，也不使当前 CUDA/CuPy/高斯光栅化代码自动具备手机运行能力。R3 等联合几何前端与 ZipDepth 是不同模式；保留它们各自的深度、位姿和尺度契约，不默认拼接 R3 位姿与未经对齐的 ZipDepth 深度。

建议交付顺序：**服务器可切换深度后端 → 手机独立预览 → 录制文件与几何侧车接入 → 实时关键帧上传**。全手机端三维优化作为独立后续项目评估。

## 2. 已核实的依据与限制

| 项目 | 核实结果 | 对方案的影响 |
| --- | --- | --- |
| 发布日期 | 官方 README 标记 2026 年 7 月发布代码和权重；论文 v1 提交于 7 月 9 日 | 用户提供的月份正确；月份发布公告不等同于精确代码首发日 |
| 模型 | 融合后约 6.1M 参数，CNN 编码器与轻量解码器 | 适合开展端侧工程验证 |
| 权重 | zipdepth_base.pth 与 zipdepth_base_npu.pth | 手机固定选择后者，并使用匹配的上采样结构 |
| 输出 | 全分辨率、具有 scale/shift 歧义的相对逆深度 | 不能直接取倒数后标成“米” |
| 时间一致性 | 论文明确指出视频闪烁问题 | 预览平滑与几何融合分别设计 |
| 导出工具 | 仓库提供 ONNX、TorchScript 导出脚本 | Core ML、LiteRT 的产品导出和验证仍需自己完成 |

来源：[官方仓库](https://github.com/fabiotosi92/ZipDepth)、[论文日期](https://arxiv.org/abs/2607.08771)、[论文方法与局限](https://arxiv.org/html/2607.08771v1)。

官方公开数据为 384×384 输入、20 次预热后 200 次前向的中位延迟：

| 设备 | 作者报告的后端 | 延迟 / 吞吐 |
| --- | --- | --- |
| iPhone 12 | Core ML ANE，NPU 路径 | 2.7 ms / 375 FPS |
| iPad Pro M4 | Core ML ANE，NPU 路径 | 1.4 ms / 715 FPS |
| Xiaomi Poco X3 NFC | TFLite GPU FP16 | 62.5 ms / 16 FPS |

这些是作者报告的模型性能，不含本应用完整相机、像素转换、预览、上传和重建成本，也不是目标手机的实测保证。[官方部署表](https://zipdepth.github.io/#deployment)、[论文表 6 的测量口径](https://arxiv.org/html/2607.08771v1#S4.SS4)。

仅按 6.1M 参数估算，FP32/FP16 权重数据约 24.4/12.2 MB；这不包括序列化信息、激活、运行时和相机缓冲区。RAM 和安装包增量必须实测。

## 3. 上游源码审查：导出前必须处理

**当前 ONNX 导出存在非等价图改写，不能把“成功生成 ONNX”作为转换验收。**

原模型 GlobalContextBlock 通过可学习卷积产生空间权重，再以 softmax 和矩阵乘法聚合特征；当前 export_onnx 将它改成固定平均池化。除非学习权重恰好均匀，否则两者不是同一计算。导出前 sanity check 只检查已改写模型的输出形状，没有比较原模型。影响大小尚未实测。[原结构](https://github.com/fabiotosi92/ZipDepth/blob/91f3fd21e131641f51e8d35736d1958350180e3a/zipdepth/model/architecture.py#L255)、[导出实现](https://github.com/fabiotosi92/ZipDepth/blob/91f3fd21e131641f51e8d35736d1958350180e3a/scripts/export.py#L60)。

处理方式：

1. 建立未改写的 NPU 版 PyTorch FP32 基准，保存输入张量与原始输出。
2. 在独立模型副本上执行 eval 和 RepVGG/Conv-BN 融合，先验证融合等价性。
3. Core ML 与 LiteRT 都从这份融合后的 PyTorch 模型转换；ONNX 路线保留可学习上下文计算，只做经过验证的等价算子表达。
4. 若某加速器无法承载原计算，先接受可度量的 CPU/GPU 分图，或另立“平均池化变体”实验并重新评估精度；不沿用原模型的准确率结论。
5. 导出后逐级比较：原 FP32 → 融合 FP32 → 目标 FP32 → 目标 FP16；每一级单独记录误差，不允许用每图仿射对齐掩盖转换偏差。

另外，两个上采样头分别含 mask_pred 和 where_conv 参数，不能将普通 checkpoint 配合一个开关当作通用权重。上游加载使用 strict=False 并仅打印 missing keys，工程加载器必须对推理参数缺失或 shape 不匹配报错，只有明确列出的训练残留键可忽略。[上采样源码](https://github.com/fabiotosi92/ZipDepth/blob/91f3fd21e131641f51e8d35736d1958350180e3a/zipdepth/model/architecture.py#L365)、[加载源码](https://github.com/fabiotosi92/ZipDepth/blob/91f3fd21e131641f51e8d35736d1958350180e3a/scripts/export.py#L37)。

## 4. 系统架构与运行模式

~~~mermaid
flowchart LR
    Camera[手机相机帧与时间戳] --> Prep[方向与色彩转换 / 尺寸映射]
    Prep --> Zip[ZipDepth 端侧推理]
    Zip --> Preview[深度预览]
    Zip --> Packet[关键帧 RGB + 相对逆深度 + 元数据]
    Camera --> Pose[可选 ARKit / ARCore 位姿与传感器深度]
    Pose --> Packet
    Packet --> Store[录制文件 / 后续实时接收端]
    Store --> Geometry[几何适配与尺度对齐]
    Legacy[现有匹配与三角化] --> Geometry
    Geometry --> GS[服务器 SceneModel / 3DGS]
    GS --> Viewer[重建结果查看]
~~~

| 模式 | 深度 / 位姿来源 | 用途 |
| --- | --- | --- |
| server_zipdepth | 服务器 ZipDepth / legacy 匹配与 BA | 最先落地，验证替换单帧深度的重建影响 |
| mobile_preview | 手机 ZipDepth / 无需位姿 | 完全离线预览、性能和热稳定性验证 |
| mobile_capture | 手机 ZipDepth / 服务器 BA，或可用的 AR 位姿 | 保存配对采集文件，再重建 |
| mobile_live | 手机关键帧深度与 RGB / 配套位姿来源 | 接收协议验证后才开放 |

上述模式名是设计标签，不是当前可用 CLI。R3 模式继续使用自身一致的几何结果，ZipDepth 不自动覆盖它的深度。

## 5. 输入输出与尺度契约

### 5.1 图像预处理

统一逻辑输入为 RGB、float32、NCHW、范围 [0,1]；端侧内部可按运行时改成 NHWC/FP16。官方推理代码是 BGR→RGB、除以 255，没有 ImageNet mean/std 标准化。不得再额外套 DA2 的预处理模板。[官方 predictor](https://github.com/fabiotosi92/ZipDepth/blob/91f3fd21e131641f51e8d35736d1958350180e3a/zipdepth/inference/predictor.py#L179)。

固定导出规格先采用 H×W=384×384 做对照，再增加 384×512 与 512×384 的 4:3 横竖屏规格。192×256/256×192 作为低功耗候选，须独立验收；不要从 384×384 论文结果外推其质量。真实 16:9 采集可增加单独规格或等比缩放加 padding。

生产路径保持完整视野，记录方向、镜像、resize、padding/crop 和有效区域。填充像素不得参加尺度估计与损失。不要将宽屏画面直接压成方形作为默认采集路径。官方脚本是短边缩放并将边长舍入到 32 的倍数，端侧固定规格与它存在差异，需在同一预处理张量上比较模型。

输入与深度必须通过显式图像变换映射到同一坐标系。对纯缩放与平移，保存 K_model=A×K_capture，并明确像素中心约定；旋转、镜像还要同步变换相机坐标与位姿，不能只改显示方向。恢复到采集网格后才能与该网格的特征点匹配。插值边界和 align_corners 约定写入 manifest，用标定点测试验证。

### 5.2 相对逆深度

建议统一 DepthPrediction 逻辑契约：

| 字段 | 含义 |
| --- | --- |
| idepth | 服务器边界统一为 [1,1,H,W] float32；语义 relative_inverse_depth |
| valid_mask | 同网格布尔有效区；区分无效值与合法相对逆深度 |
| confidence | 同网格 [0,1] 的工程权重；不是模型输出的概率 |
| frame_id / timestamp_ns | 会话内唯一帧号、相机采集时间戳及其时钟域 |
| image_transform / K | 输入到采集坐标的映射、相机内参 |
| model_id / source / alignment | 权重和预处理版本、后端、是否已对齐及目标坐标系 |

原始输出先保存，供导出比对和复现。legacy BA 路径可将有效区按 median/MAD 标准化，再对齐三角化逆深度；归一化结果允许为负，因此在仿射对齐前不能按“非正即无效”删除所有负值。

令 r_i 为相对逆深度，z_i 为可靠三角化点在对应相机坐标的正 Z 值，以鲁棒拟合求解：

~~~text
minimize_(a>0, b)  Σ w_i · Huber(a·r_i + b − 1/z_i)
q(u,v) = a·r(u,v) + b
Z(u,v) = 1/q(u,v)    仅对有限且 q>epsilon 的已对齐有效样本
~~~

无尺度视觉 BA 中的 Z 仍是场景单位；只有参考几何具有经验证的米制尺度，才可标注为米。单个距离锚点不足以同时约束 a 和 b，应使用有深度跨度且空间分布良好的多个可靠样本。

初始门槛建议：至少 30 个内点、覆盖至少 6 个 4×4 图像网格、设计矩阵条件数合格；均为待验证参数。MAD 近零、有效点不足、负尺度、残差过大时，标记 alignment_invalid，不用 clamp 伪造完整深度。暂停该帧深度约束和依赖深度的稠密初始化；由正常位姿/关键帧恢复策略处理。

### 5.3 置信度与时间一致性

ZipDepth 没有原生不确定性输出。第一期保留当前 Sobel 边缘启发式，增加 finite/padding/对齐残差掩码，并将其标为工程权重。边缘小不代表深度正确，不能因墙面平滑就认为几何可靠。

预览可使用带重置的时域平滑改善观感；用于重建的深度默认保留原始预测。只有获得相容位姿与尺度后，才能将上一帧深度投影到当前帧，在通过遮挡、动态区域和残差检查的像素上融合。禁止直接对相邻帧同一像素的原始相对逆深度做 EMA。

ARKit/ARCore 的外部位姿要同时标记 c2w/w2c、相机轴向、世界原点、长度单位和 session_id；跟踪丢失或重定位产生新 gauge 时分段或统一重对齐。外部传感器深度与 RGB 的时间、内外参也必须配准。米制深度不再经过每帧 median/MAD 归一化。

## 6. 当前仓库的接入改动

下面是待实施改动，不代表这些配置已经可用。实现前以当时工作区代码复核，避免覆盖并行的 R3 工作。

| 位置 | 计划改动 |
| --- | --- |
| [scene/mono_depth.py](../scene/mono_depth.py) | 拆出深度后端工厂；DA2 保留原行为；增加 ZipDepth 推理适配，统一返回契约 |
| [geometry/provider.py](../geometry/provider.py) | DefaultGeometryProvider 从工厂获取深度；ExternalGeometryProvider 缺失字段使用同一工厂；R3 模式独立 |
| [model_store.py](../model_store.py) | 增加 ZipDepth 权重、源码版本与导出产物登记；沿用 MODELS_DIR 和原子写入，增加 SHA-256 校验 |
| [args.py](../args.py)、Compose 配置 | 增加 depth_backend、输入规格、精度选项；DEPTH_MODEL 继续只表示 DA2 的 vits/vitb 等，不混入 zipdepth |
| [dataloaders/image_dataset.py](../dataloaders/image_dataset.py) | 将侧车深度/置信度规范成 4D；解析 schema、变换、有效区和尺度元数据；拒绝无法解释的格式 |
| [scene/keyframe.py](../scene/keyframe.py) | 按 relative_affine / scene_inverse_depth / metric_depth 分流对齐；保存有效区金字塔；统一无效深度恢复行为 |
| [scene/scene_model.py](../scene/scene_model.py) | 深度损失使用有效区和校准权重归一化；有效权重为零时该项为零；应用 NaN 前先筛除无效值 |
| [dataloaders/stream_dataset.py](../dataloaders/stream_dataset.py) | 后续接入同帧 RGB/深度/位姿数据包；普通视频流本身不携带这些数据 |

当前源码审查发现的前置问题：

- MonoDepthInternal 是 CUDA half + CUDA graph 专用包装，不能直接复制到手机。ZipDepth 首先使用独立 CPU/CUDA eager 适配验证，CUDA graph 是后续优化。
- 现有侧车 loader 的 _ensure_chw 通常产生 [1,H,W]，Keyframe.align_depth 的双线性插值按 [N,C,H,W] 消费；需在统一边界修正并做真实侧车到 Keyframe 的联调。
- get_t_s / align_samples 的 MAD 除法缺少退化保护；精确拟合时 err.median()==0，而严格小于筛选可能去掉全部样本。要覆盖常量图、零点、少点、完美拟合和异常点测试。
- ExternalGeometryProvider 可以把外部深度与 fallback 深度生成的置信度混在一起；改为根据实际采用的深度生成其权重，或采用明确的有效区权重。
- 当前 SceneModel 的 depth_loss 为未乘深度置信度的全图平均。仅返回零置信度不能保证坏深度不参与训练，还必须同时保护高斯采样、反投影和深度损失入口。
- 现有完整 K 会被压缩成单 focal，渲染相机使用中心主点。第一期应将 RGB/深度一致校正到支持的虚拟相机；如保留原始全内参，则需完整扩展投影、匹配、光栅化和导出契约，不能只取 fx/fy 平均值。

预期配置示例（实现后才可使用）：

~~~text
--geometry_provider default
--depth_backend zipdepth
--depth_variant base_npu
--depth_input_height 384
--depth_input_width 512
--depth_precision fp16
~~~

深度后端选择只影响 legacy 或明确允许的 external fallback 路径。选 R3 同时显式指定 ZipDepth 时应报不兼容配置，或通过单独实验适配器处理，不能静默覆盖。

建议产物布局：models/zipdepth/<source-revision>/<checkpoint-sha256>/ 下保存原权重、各规格导出模型与 manifest.json。源码放独立第三方依赖目录，不混入权重目录；缓存 key 包含 checkpoint、图变换版本、尺寸、精度和转换器版本。显式本地文件缺失、哈希不符或加载不完整时立即报错，不悄悄替换成另一种深度模型。

## 7. iOS 与 Android 部署路线

### 7.1 iOS：Core ML 优先

路线：融合后的 PyTorch → 经验证的 TorchScript trace / torch.export → coremltools → FP16 ML Program (.mlpackage) → 应用编译产物。Apple 当前转换指南仍将 trace 列为成熟推荐路线，export 支持需随锁定工具版本验证。[Apple 官方流程](https://apple.github.io/coremltools/docs-guides/source/convert-pytorch-workflow.html)。

第一版使用 Tensor 输入隔离模型误差，再增加 ImageType/CVPixelBuffer 输入以减少复制。比较 CPU_ONLY 与允许加速器的配置，记录实际执行计划和算子分配；允许使用 ANE 不代表全图都在 ANE。用 Xcode Instruments / Core ML profiling 检查耗时与内存。

ML Program 最低部署目标为 iOS 15；最终最低系统版本还要结合所用 API 和目标设备冻结。FP32 基准导出必须显式设置 compute_precision=FLOAT32，不能从 float32 输入/输出推断中间计算精度；生产 FP16 另存产物。[Apple 部署与精度说明](https://apple.github.io/coremltools/docs-guides/source/convert-to-ml-program.html)。

相机层采用 AVFoundation；需要 AR 位姿时以 ARSession 的 capturedImage、timestamp、intrinsics、camera transform 作为同帧来源，避免两个独立采集会话错配。预处理可用 Metal，预览从 GPU 纹理完成，避免每帧深度转 PNG/JPEG。

### 7.2 Android：LiteRT GPU 优先，NPU 后续验证

路线：融合后的 PyTorch → litert_torch → .tflite → LiteRT CompiledModel GPU；CPU 作为验证和兼容回退。锁定转换器、运行时、NDK、目标 ABI 和最低系统版本。[官方 PyTorch 转换](https://developers.google.com/edge/litert/conversion/pytorch/overview)、[Android 运行时选择](https://developers.google.com/edge/litert/android)。

2026-09-09 查阅的官方版本表列 LiteRT 2.2.0。其 2.x Interpreter 路径仅 CPU，新实现应使用 CompiledModel；不要拼接新版本依赖和旧 GPU delegate 示例。ZipDepth 的官方 TFLite 演示是可行性依据，未证明当前版本组合无需适配。

CameraX 的分析队列采用只保留最新帧策略，及时释放 ImageProxy。YUV 变换明确 plane stride、色彩矩阵、full/limited range、旋转和镜像；统一模型输入测试必须覆盖这些变换。ARCore 模式从同一 Frame 配对图像与 pose，设备不支持 AR 功能时仍可使用服务器 BA 模式。

ONNX Runtime CPU 保留为数值参考与兼容路线；Qualcomm 设备可另测 ORT QNN 或 LiteRT 厂商 NPU。QNN 官方新旧文档对 HTP 精度支持描述不同，因此按实际 ORT/QNN/SoC 组合决定 FP16/量化能力，不预设“一定 INT8”或“一定全 NPU”。[QNN 当前仓库说明](https://github.com/onnxruntime/onnxruntime-qnn/blob/main/docs/execution_providers/QNN-ExecutionProvider.md)、[LiteRT NPU 支持](https://developers.google.com/edge/litert/next/npu)。

### 7.3 精度与调度

FP32 用于基准，FP16 为首选候选。INT8 排到 FP16 真机验收之后，使用覆盖实际采集域的代表集校准，并报告量化后深度边缘与重建质量。量化文件不意味着 GPU 按 INT8 执行；加速器与 CPU 分图的同步开销也可能抵消收益。[官方 GPU 限制说明](https://developers.google.com/edge/litert/performance/gpu)。

相机预览可保持 30 FPS，深度按能力设为 10–15 Hz 起步，服务器上传只发 1–5 Hz 候选关键帧；这些是本项目起始目标。每个模型会话单个推理任务，最多一个待处理最新帧；记录被替换帧号，不让推理队列无限增长。

异步任务开始前必须复制输入，或持有有效的相机缓冲区直到读取结束，再释放 ImageProxy/ARCore Image/CVPixelBuffer。最新帧覆盖不能修改正在推理的缓冲区；上传的 RGB 和深度引用同一份不可变帧记录。用有界双缓冲或引用计数管理生命周期。

热状态或 p95 延迟持续越界时依次降深度频率、切换已验收的低分辨率规格；不能使用未经验证的随机动态尺寸。切换后重置预览历史和局部对齐状态，记录 source/profile 变化。CPU 回退需降频并可观察，不能悄悄保持“高速模式”标签。

## 8. 采集文件与实时传输

第一期先录制再导入：每个被选中的 RGB 帧同时保存原始逆深度、有效区、配套元数据；服务器转换为按图像 stem 对应的 geometry 侧车。内存加载前核验图像哈希、frame_id、尺寸和时间戳。新增有效区等字段需先扩展 loader，不能认为当前侧车已全部支持。

移动端便于实现的传输格式是：JSON 元数据 + 明确字节序的 float16/float32 单通道二进制深度 + mask + 原始/编码 RGB。服务器在受控边界校验并转换为 .npz；不要求手机实现 PyTorch .pt，不传彩色深度图，也不用未经标度声明的 8 位灰度保存几何。

最小包字段：schema_version、session_id、frame_id、capture_timestamp_ns、clock_domain、RGB 文件 hash、RGB/深度尺寸、pixel format、图像变换、原始 K、相对逆深度语义、dtype、有效区、model/checkpoint/profile id，以及可选 pose convention、units、tracking state。压缩内容解码后校验实际长度和尺寸。

实时阶段采用一条可靠有序连接传配对数据，应用层用 frame_id 关联并应答；缺失帧或不匹配的 RGB/depth 包整帧拒收。时间戳用于运动/延迟分析，不能只按到达时间配对。网络断开时本地继续记录，恢复后按 session/frame 去重补传；未建立新世界坐标关联前，不跨 AR session 拼接位姿。

容量示例：384×512 的单通道 FP16 深度每帧 384 KiB；10 Hz 未压缩约 3.75 MiB/s（31.5 Mbit/s），尚不含 RGB、mask 和协议。因此手机推理本身不保证节省网络流量。优先只传关键帧，压缩收益按真实数据测量。

## 9. 验收矩阵

以下阈值为本项目建议起点，需在开发前冻结，不能测试失败后随意放宽。模型转换、深度质量和三维重建分别验收。

| 层次 | 方法 | 建议通过条件 |
| --- | --- | --- |
| 预处理与契约 | 彩条、标定板、横竖屏、crop/pad、真实 YUV 输入 | 通道和范围正确；投影往返误差 ≤0.5 像素；shape/frame_id 一致 |
| 权重加载 | 缺文件、错头、损坏哈希、missing key | 均明确失败；正常离线加载无需下载 |
| FP32 转换 | 相同输入逐像素对照原模型，至少 100 张覆盖场景图片 | 全部有限；NRMSE ≤1e-4，近常量输出另用最大绝对误差 ≤1e-5；无图算义改动 |
| FP16 转换 | 同上，另看 p99 像素误差及深度边缘 | NRMSE ≤1e-2；经参考 GT 对齐的深度指标相对原 FP32 恶化 ≤1%；边缘无系统性退化 |
| 对齐健壮性 | 常量图、少点、完美拟合、异常值、负尺度、丢失跟踪 | 无 NaN；无效帧不参与深度损失/初始化；恢复路径可重复 |
| 短序列重建 | 固定图像、pose 模式、迭代数、Gaussian 预算和评估视图 | 全部完成；记录逐场景结果，不把单图深度指标当重建指标 |
| 替换质量 | DA2 原规格与 ZipDepth 产品规格对照；额外做同规格效率实验 | 中位 PSNR 下降 ≤0.3 dB、LPIPS 上升 ≤0.01，且无不可接受的新增浮点/破洞/薄结构丢失 |
| 手机持续运行 | 各目标档位连续 15 分钟，包含相机和预览 | 深度发布 ≥10 Hz、采集到深度可用 p95 ≤100 ms 为标准档目标；不达标则定位低功耗档，不声称达标 |
| 资源 | 真机内存、热状态、功耗/电量、复制成本 | 深度模块增量内存初始预算 ≤150 MiB，无持续增长、闪退或系统过热中断；预算不是已测事实 |
| 实时链路 | 丢帧、断网、乱序、重连、重复包、旋转中断 | 无跨帧深度错配；队列有界；可继续离线采集 |

NRMSE 定义为 RMSE(pred−reference)/max(std(reference), epsilon)，在同一有效区计算；近常量判断及 epsilon 随数据单位冻结。转换比较禁止额外拟合 scale/shift；深度准确率评估可按相对深度协议对齐，但须明确这是使用参考 GT 的离线评价，不能用来声称在线米制精度。

真实采集覆盖：纹理房间、白墙、走廊回环、玻璃/镜面、细椅腿、电线、室外植被、弱光、快速旋转、行人动态区域。iOS 至少一台较老 ANE 设备与目标机，Android 至少目标高通设备和一台不同厂商/中低档设备；具体型号尚未指定。

性能记录包括冷启动、预热、预处理、纯模型、后处理、采集到结果延迟、上传、服务器几何处理和 3DGS 更新。报告 p50/p95、10–15 分钟热稳态和实际算子后端，不只报平均 FPS。最终包包含设备/OS、源码 revision、权重 hash、工具版本、输入列表和可复跑 benchmark。

## 10. 实施拆分与交付物

| 阶段 | 工作 | 交付与退出条件 | 单工程师估算 |
| --- | --- | --- | --- |
| P0 | 权重与源码固定、原模型基准、融合/导出等价性审查 | manifest、参考张量集、转换误差报告；发现非等价图时先修正或隔离 | 2–3 工作日 |
| P1 | 深度工厂、ZipDepth 适配、尺度/有效区健壮性、服务器重建 A/B | 可选服务器后端与真实序列报告，既有模式回归通过 | 3–5 工作日 |
| P2 | 一个 iOS 和一个 Android 原生推理样例、固定规格与热测试 | .mlpackage、.tflite、可安装样例、真机性能与后端分配报告 | 5–8 工作日 |
| P3 | 配对录制、侧车导入，再做实时接收与重连 | 真实手机采集到服务器重建闭环、异常恢复报告 | 3–5 工作日 |

合计约 13–21 工作日，前提是具备目标手机与可用 CUDA 服务器；算子适配、全内参改造或特定 NPU 支持可能增加工期。P0/P1 未过门槛时先保留实验后端；P2 也可作为独立离线预览能力交付。

版本记录应附上 ZipDepth 的 MIT LICENSE、教师来源说明及当前重建仓库的许可。ZipDepth README 说明训练伪标签来自 Depth Anything V2 Large，其官方条款标为 CC-BY-NC-4.0；这两个事实不能自行推导学生权重必然继承或完全不受影响。若未来分发商业产品，确认具体使用与分发范围。来源：[ZipDepth LICENSE](https://github.com/fabiotosi92/ZipDepth/blob/91f3fd21e131641f51e8d35736d1958350180e3a/LICENSE)、[DA2 许可](https://github.com/DepthAnything/Depth-Anything-V2#license)、[本项目许可](../LICENSE.md)。

本方案完成的是官方信息核实、源码审查与实施设计。所有性能预算、转换阈值和排期均为工程目标；实际模型导出、服务器质量对比及真机结论待上述阶段验证。
