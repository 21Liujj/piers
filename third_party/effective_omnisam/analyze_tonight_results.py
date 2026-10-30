#!/usr/bin/env python3
"""
今晚实验结果分析脚本
用法: python analyze_tonight_results.py
"""
import json
import glob
from pathlib import Path
from collections import defaultdict

print("="*80)
print("🔍 OmniSAM Tonight Experiment Results Analysis")
print("="*80)

# ========== Step 1: 收集所有结果 ==========
print("\n📊 Step 1: Collecting all n500 results...")

results = {}
for json_file in sorted(glob.glob("experiments/**/eval_results.json", recursive=True)):
    if 'n500' not in json_file:
        continue
    
    try:
        with open(json_file) as f:
            data = json.load(f)
        
        config_name = str(Path(json_file).parent.name)
        results[config_name] = {
            'mIoU': data.get('mIoU', 0),
            'BFScore': data.get('BFScore', 0),
            'per_class': data.get('per_class_iou', {}),
            'time': data.get('avg_time_per_image', 0),
        }
        print(f"  ✓ {config_name}")
    except Exception as e:
        print(f"  ✗ {json_file}: {e}")

print(f"\n找到 {len(results)} 个实验结果")

# ========== Step 2: 计算派生指标 ==========
print("\n📈 Step 2: Computing derived metrics...")

for config, res in results.items():
    per_class = res['per_class']
    
    # 小物体平均 (traffic_light, pole, traffic_sign)
    small_obj_classes = []
    for cls_name in ['traffic_light', 'pole', 'traffic_sign']:
        if cls_name in per_class:
            small_obj_classes.append(per_class[cls_name])
    res['small_obj_avg'] = sum(small_obj_classes) / len(small_obj_classes) if small_obj_classes else 0
    
    # 稀有类平均 (fence, terrain, train)
    rare_classes = []
    for cls_name in ['fence', 'terrain', 'train']:
        if cls_name in per_class:
            rare_classes.append(per_class[cls_name])
    res['rare_class_avg'] = sum(rare_classes) / len(rare_classes) if rare_classes else 0

# ========== Step 3: 按模块分组 ==========
print("\n🗂️  Step 3: Grouping by module...")

pano_configs = {k: v for k, v in results.items() if 'min128' in k or 'min256' in k or 'min512' in k}
boundary_configs = {k: v for k, v in results.items() if 'boundary' in k}
rare_configs = {k: v for k, v in results.items() if 'rare' in k}

print(f"  Pano-SMB: {len(pano_configs)} configs")
print(f"  Boundary-Guard: {len(boundary_configs)} configs")
print(f"  Class-Aware: {len(rare_configs)} configs")

# ========== Step 4: 打印所有结果 ==========
print("\n" + "="*80)
print("📋 All Results Summary")
print("="*80)
print(f"{Config:<35} {mIoU:>8} {BFScore:>8} {SmallObj:>8} {Rare:>8} {Time:>6}")
print("-"*80)

for config in sorted(results.keys()):
    res = results[config]
    print(f"{config:<35} {res[mIoU]:>8.2f} {res[BFScore]:>8.2f} "
          f"{res[small_obj_avg]:>8.2f} {res[rare_class_avg]:>8.2f} {res[time]:>6.2f}")

# ========== Step 5: 应用简化选择规则 ==========
print("\n" + "="*80)
print("🎯 Decision Making (Simplified Rules)")
print("="*80)

# ========== 决策1: Pano-SMB ==========
print("\n【决策1: Pano-SMB - 选择最优 min_tokens】")
print("-"*80)
print("规则:")
print("  1. mIoU 必须 ≥ 60.15 (基本要求)")
print("  2. 在满足1的情况下，选 mIoU 最高的")
print("  3. 如果 mIoU 相差 < 0.05，选 BFScore 最高的")
print()

if pano_configs:
    # 过滤: mIoU >= 60.15
    candidates = {k: v for k, v in pano_configs.items() if v['mIoU'] >= 60.15}
    
    if not candidates:
        print("⚠️  警告: 没有配置满足 mIoU ≥ 60.15，放宽标准...")
        candidates = pano_configs
    
    # 找 mIoU 最高的
    best_miou = max(candidates.items(), key=lambda x: x[1]['mIoU'])
    
    # 检查是否有其他配置 mIoU 接近
    similar = {k: v for k, v in candidates.items() 
               if abs(v['mIoU'] - best_miou[1]['mIoU']) < 0.05}
    
    if len(similar) > 1:
        # 有多个接近的，选 BFScore 最高的
        best_pano = max(similar.items(), key=lambda x: x[1]['BFScore'])
        print(f"✅ 多个配置 mIoU 接近，选 BFScore 最高的")
    else:
        best_pano = best_miou
        print(f"✅ mIoU 最高的配置")
    
    print(f"\n推荐配置: {best_pano[0]}")
    print(f"  mIoU:     {best_pano[1]['mIoU']:.2f}")
    print(f"  BFScore:  {best_pano[1]['BFScore']:.2f}")
    print(f"  SmallObj: {best_pano[1]['small_obj_avg']:.2f}")
else:
    print("❌ 未找到 Pano-SMB 配置")
    best_pano = None

# ========== 决策2: Boundary-Guard ==========
print("\n【决策2: Boundary-Guard - 选择最优 boundary_reserve_ratio】")
print("-"*80)
print("规则:")
print("  1. BFScore 必须 > 39.86 (必须超过 MTC baseline)")
print("  2. SmallObj 平均必须有提升 (期望 > 63.8)")
print("  3. 在满足1、2的情况下，选 SmallObj 最高的")
print()

if boundary_configs:
    # 过滤: BFScore > 39.86
    candidates = {k: v for k, v in boundary_configs.items() if v['BFScore'] > 39.86}
    
    if not candidates:
        print("⚠️  警告: 没有配置 BFScore > 39.86，选 BFScore 最高的...")
        candidates = boundary_configs
    
    # 过滤: SmallObj > 63.8
    candidates2 = {k: v for k, v in candidates.items() if v['small_obj_avg'] > 63.8}
    
    if not candidates2:
        print("⚠️  警告: SmallObj 提升不明显，但仍从候选中选择...")
        candidates2 = candidates
    
    # 选 SmallObj 最高的
    best_boundary = max(candidates2.items(), key=lambda x: x[1]['small_obj_avg'])
    
    print(f"✅ SmallObj 提升最明显的配置")
    print(f"\n推荐配置: {best_boundary[0]}")
    print(f"  mIoU:     {best_boundary[1]['mIoU']:.2f}")
    print(f"  BFScore:  {best_boundary[1]['BFScore']:.2f}")
    print(f"  SmallObj: {best_boundary[1]['small_obj_avg']:.2f} ⭐")
else:
    print("❌ 未找到 Boundary-Guard 配置")
    best_boundary = None

# ========== 决策3: Class-Aware ==========
print("\n【决策3: Class-Aware - 选择最优 rare_reserve_ratio】")
print("-"*80)
print("规则:")
print("  1. mIoU 必须 ≥ 60.15")
print("  2. RareClass 平均必须有提升 (期望 > 61.5)")
print("  3. 在满足1、2的情况下，选 RareClass 最高的")
print()

if rare_configs:
    # 过滤: mIoU >= 60.15
    candidates = {k: v for k, v in rare_configs.items() if v['mIoU'] >= 60.15}
    
    if not candidates:
        print("⚠️  警告: 没有配置 mIoU ≥ 60.15，放宽标准...")
        candidates = rare_configs
    
    # 过滤: RareClass > 61.5
    candidates2 = {k: v for k, v in candidates.items() if v['rare_class_avg'] > 61.5}
    
    if not candidates2:
        print("⚠️  警告: RareClass 提升不明显，但仍从候选中选择...")
        candidates2 = candidates
    
    # 选 RareClass 最高的
    best_rare = max(candidates2.items(), key=lambda x: x[1]['rare_class_avg'])
    
    print(f"✅ RareClass 提升最明显的配置")
    print(f"\n推荐配置: {best_rare[0]}")
    print(f"  mIoU:      {best_rare[1]['mIoU']:.2f}")
    print(f"  RareClass: {best_rare[1]['rare_class_avg']:.2f} ⭐")
else:
    print("❌ 未找到 Class-Aware 配置")
    best_rare = None

# ========== Step 6: 总结和下一步 ==========
print("\n" + "="*80)
print("🎯 Final Summary & Next Steps")
print("="*80)

print("\n【今晚实验结论】")
if best_pano:
    print(f"  ✅ Pano-SMB 最优:  {best_pano[0]}")
if best_boundary:
    print(f"  ✅ Boundary 最优:  {best_boundary[0]}")
if best_rare:
    print(f"  ✅ Class-Aware 最优: {best_rare[0]}")

print("\n【明天的 n2000 实验】")
print("  需要跑的配置:")
print("    1. baseline_n2000 × 5 seeds (已有)")
print("    2. MTC_n2000 × 5 seeds (已有)")
if best_pano:
    print(f"    3. {best_pano[0]}_n2000 × 5 seeds")
if best_boundary:
    print(f"    4. {best_boundary[0]}_n2000 × 5 seeds")
if best_rare:
    print(f"    5. {best_rare[0]}_n2000 × 5 seeds (可选)")
print("    6. full_system_n2000 × 5 seeds (组合所有最优配置)")

print("\n  预计时间: 15-20 × 1.5h = 22.5-30h (分两天完成)")

# ========== 保存结果 ==========
output = {
    'all_results': results,
    'best_configs': {
        'pano_smb': best_pano[0] if best_pano else None,
        'boundary_guard': best_boundary[0] if best_boundary else None,
        'class_aware': best_rare[0] if best_rare else None,
    }
}

with open('tonight_analysis_summary.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n💾 结果已保存到: tonight_analysis_summary.json")
print("="*80)
