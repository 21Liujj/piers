ckpt_zoo（checkpoint 归档与一键验证）

这个目录用于把工作区里分散的 ckpt 统一“收纳”到一个地方，避免你记不清每次实验到底用的是哪个权重。

目录结构：
  ckpt_zoo/
    files/            # 归档后的 ckpt（默认是软链接 symlink，不会重复占用磁盘）
    index.json        # 机器可读清单（给脚本用）
    CKPT_INDEX.txt    # 人类可读索引（给你查阅/写实验记录用）
    val_logs/         # 用 ckpt_zoo 跑 validation.py 的日志输出

常用命令：
  1) 扫描并归档（默认 symlink，安全不破坏原始目录）
     python tools/ckpt_zoo.py collect

  2) 查看有哪些 ckpt
     python tools/ckpt_zoo.py list
     python tools/ckpt_zoo.py list --grep iou60

  3) 选择 ckpt 跑验证（DensePASS / WildPASS）
     python tools/ckpt_zoo.py val --id 73 --dataset DensePASS --class_set 13
     python tools/ckpt_zoo.py val --match best_iou60.36 --dataset DensePASS

说明：
  - 如果同一个关键词 match 到多个 ckpt，脚本会自动选择 mtime 最新的那一个。
  - files/ 下多为指向实验目录的软链接；若原始路径被删，会出现「断链」——此时用
    `find <repo> -type f -name 'best_iou60.26.pth'` 找现存文件，再 `ln -sf` 修复或重新 collect。
  - 如果你希望“真的复制”到 files/（而不是链接），用：
       python tools/ckpt_zoo.py collect --mode copy

