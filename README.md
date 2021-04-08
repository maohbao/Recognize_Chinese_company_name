# Recognize Chinese company name from text snippet 

## Train
The main training procedure is in `trainer.py`

Examples to start training are in `scripts/reproduce`.

Note that you may need to change `DATA_DIR`, `BERT_DIR`, `OUTPUT_DIR` to your own
dataset path, bert model path and log path, respectively.

## Evaluate and predict
`trainer.py` will automatically evaluate on dev set every `val_check_interval` epochs,
and save the topk checkpoints to `default_root_dir`.

To evaluate them, use `evaluate.py`. The evaluate result is as follow:
```
'span_f1': tensor(0.9509, device='cuda:0'),
 'span_precision': tensor(0.9532, device='cuda:0'),
 'span_recall': tensor(0.9487, device='cuda:0'),
 'val_loss': tensor(0.0158, device='cuda:0')
 ```

To predict, use `predict.py`. The predict result is as follow:
```
香 港 启 德 专 业 会 计 秘 书 有 限 公 司 成 立 于 2006 年, 是 专 业 的 商 务 顾 问 及 公 司 注 册 服 务 机 构, 为 客 户 提 供 企 业 服 务 和 顾 问 咨 询 服 务
【Company】:  香 港 启 德 专 业 会 计 秘 书 有 限 公 司

 公 司 成 立 初 期 ， 是 以 design house 形 式 ， 至 力 于 智 能 无 铅 焊 接 、 智 能 锁 附 系 统 、 电 子 电 动 工 具 的 软 硬 件 的 研 发 和 pcba 控 制 板 的 设 计 与 制 作 。 深 圳 市 艾 迪 赛 科 技 有 限 公 司 位 于 深 圳 市 宝 安 区 沙 井 后 亭 第 二 工 业 区 。
【Company】:  深 圳 市 艾 迪 赛 科 技 有 限 公 司

 截 至 目 前 ， 我 们 已 拥 有 120 多 项 专 利 和 80 多 项 软 件 著 作 权 ， 在 德 国 和 奥 地 利 设 有 技 术 研 发 中 心 ， 在 上 海 、 北 京 、 广 州 、 武 汉 、 嘉 兴 、 无 锡 、 昆 明 、 莫 斯 科 等 城 市 设 有 子 公 司 或 工 厂 。
【Company】: None

 2014 年 1 月 正 式 在 新 三 板 挂 牌 上 市 ， 证 券 简 称 xxx 。 集 科 研 、 生 产 、 销 售 、 服 务 于 一 体 的 专 业 公 司 北 京 力 码 科 信 息 技 术 股 份 有 限 公 司 是 一 家 专 注 从 事 工 业 、 商 业 及 物 联 网 标 识 系 统 整 体 解 决 方 案 研 发 和 推 广 。
【Company】:  北 京 力 码 科 信 息 技 术 股 份 有 限 公 司
```

Key ideas to get this working are due to [this github](https://github.com/ShannonAI/mrc-for-flat-nested-ner).
