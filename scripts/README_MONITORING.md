# 训练进度监控指南

## 方法1: 使用检查脚本（推荐）

```bash
# 一键查看所有信息
./scripts/check_training_progress.sh
```

这会显示：
- 当前作业队列
- 最近作业历史
- 最新训练日志（最后20行）
- 错误日志
- 模型文件状态
- 训练进度（步数和损失）

## 方法2: 实时监控日志

```bash
# 自动监控最新作业
./scripts/watch_training.sh

# 或指定作业ID
./scripts/watch_training.sh <job_id>
```

## 方法3: 手动查看日志

```bash
# 查看最新的训练日志
tail -f outputs/train_toy_image_lesion_*.log

# 或找到最新的日志文件
LATEST=$(ls -t outputs/train_toy_image_lesion_*.log | head -1)
tail -f "$LATEST"

# 查看最后N行
tail -50 outputs/train_toy_image_lesion_*.log

# 查看训练进度（只显示step和loss）
grep "step.*loss" outputs/train_toy_image_lesion_*.log | tail -10
```

## 方法4: 使用 sacct 查看作业历史

```bash
# 查看今天的作业
sacct -u $USER --format=JobID,JobName,State,Start,End,Elapsed,MaxRSS

# 查看特定作业的详细信息
sacct -j <job_id> --format=JobID,JobName,State,Start,End,Elapsed,MaxRSS,NodeList,ReqMem,AllocCPUS

# 查看最近1小时的作业
sacct -u $USER -S $(date -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) --format=JobID,JobName,State,Elapsed
```

## 方法5: 查看作业详细信息

```bash
# 查看作业详细信息
scontrol show job <job_id>

# 查看所有作业
squeue -u $USER -o "%.10i %.20j %.8T %.10M %.6D %R %.8e"
```

## 方法6: 检查模型文件

```bash
# 检查模型文件是否存在及大小
ls -lh toy_image_lesion_diffusion.pt

# 查看文件修改时间（判断是否在更新）
stat toy_image_lesion_diffusion.pt

# 或使用
ls -lht *.pt | head -5
```

## 方法7: 监控GPU使用情况

```bash
# 如果作业正在运行，可以查看GPU使用
srun --jobid=<job_id> --pty bash -c 'nvidia-smi'

# 或者进入计算节点后
nvidia-smi
```

## 方法8: 查看训练输出目录

```bash
# 查看所有输出文件
ls -lht outputs/train_toy_image_lesion_* | head -10

# 查看文件大小变化（判断是否在写入）
watch -n 5 'ls -lh outputs/train_toy_image_lesion_*.log'
```

## 快速命令总结

```bash
# 1. 查看队列
squeue -u $USER

# 2. 查看最新日志（实时）
tail -f outputs/train_toy_image_lesion_*.log

# 3. 查看训练进度
grep "step.*loss" outputs/train_toy_image_lesion_*.log | tail -5

# 4. 查看作业历史
sacct -u $USER --format=JobID,JobName,State,Elapsed -S today

# 5. 检查模型文件
ls -lh toy_image_lesion_diffusion.pt
```

## 判断训练是否完成

训练完成的标志：
1. `squeue` 中看不到作业
2. 日志最后一行显示 "Training completed"
3. 模型文件 `toy_image_lesion_diffusion.pt` 存在且大小合理（约300-500KB）
4. `sacct` 显示作业状态为 "COMPLETED"
