import pandas as pd
import numpy as np
import time

# ==========================================
# 1. 基础配置 (Configuration)
# ==========================================
# 属性水平定义
# 顺序: 智能(3), 上下文(3), 隐私(2), 价格(3)
levels = [3, 3, 2, 3] 
attribute_names = ["智能程度", "上下文窗口", "隐私保护", "月订阅价格"]

# 文本映射字典
level_text_map = {
    "智能程度": {
        0: "LV1: 基础辅助（易幻觉，仅适合润色/闲聊）",
        1: "LV2: 进阶推理（擅长代码/推导，准确率高）",
        2: "LV3: 专家创作（深度思维链，可独立写论文）"
    },
    "上下文窗口": {
        0: "LV1: 标准（只能读1篇长论文，记不住前文）",
        1: "LV2: 较长（可读整本教材/整个代码库）",
        2: "LV3: 超长（全资料库记忆，了解你的一切）"
    },
    "隐私保护": {
        0: "LV1: 默认开启（您的数据会被用于训练AI）",
        1: "LV2: 严格保密（企业级无痕，数据绝不外泄）"
    },
    "月订阅价格": {
        0: "20元/月",
        1: "60元/月",
        2: "130元/月"
    }
}

# ==========================================
# 2. 核心算法工具 (Core Algorithm Utils)
# ==========================================
def fullfact(levels):
    """生成全因子设计矩阵 (72种组合)"""
    n = len(levels)
    nb_lines = np.prod(levels)
    H = np.zeros((nb_lines, n))
    level_repeat = 1
    range_repeat = np.prod(levels)
    for i in range(n):
        range_repeat //= levels[i]
        lvl = []
        for j in range(levels[i]):
            lvl += [j] * level_repeat
        rng = lvl * range_repeat
        level_repeat *= levels[i]
        H[:, i] = rng
    return H

def evaluate_design(df_subset, df_universe):
    """评估设计：返回 (最大相关系数, 平衡性惩罚分)"""
    # 1. 相关性 (Orthogonality)
    corr_matrix = df_subset.corr().abs()
    np.fill_diagonal(corr_matrix.values, 0)
    max_corr = corr_matrix.max().max()
    
    # 2. 平衡性 (Balance)
    balance_penalty = 0
    for col in df_subset.columns:
        counts = df_subset[col].value_counts()
        expected_len = len(df_universe[col].unique())
        if len(counts) < expected_len: 
            balance_penalty += 100 
        else:
            balance_penalty += counts.std()
            
    return max_corr, balance_penalty

# ==========================================
# 3. 寻找最优设计 (Optimization)
# ==========================================
def find_best_design(num_tasks=9, iterations=50000):
    print(f"正在启动蒙特卡洛搜索，尝试 {iterations} 种随机组合...")
    start_time = time.time()
    
    design_matrix = fullfact(levels)
    df_universe = pd.DataFrame(design_matrix, columns=attribute_names)
    
    total_profiles = num_tasks * 2
    best_score = float('inf')
    best_indices = None
    
    for i in range(iterations):
        current_indices = np.random.choice(len(df_universe), total_profiles, replace=False)
        df_temp = df_universe.iloc[current_indices]
        
        max_corr, balance_penalty = evaluate_design(df_temp, df_universe)
        
        # 权重: 相关性优先 (x100)
        current_score = max_corr * 100 + balance_penalty 
        
        if current_score < best_score:
            best_score = current_score
            best_indices = current_indices
            
            # 极佳结果提前终止
            if max_corr < 0.15 and balance_penalty < 2.0:
                break
    
    print(f"搜索完成，耗时 {time.time() - start_time:.2f} 秒")
    return df_universe, best_indices

# ==========================================
# 4. 质量检验模块 (New Verification Module)
# ==========================================
def verify_final_design(df_universe, best_indices, num_tasks):
    """
    在最后输出详细的设计质量诊断报告
    """
    print("\n" + "="*30)
    print(" 📊 最终设计质量诊断报告 (Diagnosis) ")
    print("="*30)
    
    # 重建最终的 18 个 Profile 用于检验
    # 注意：这里我们检验的是除陷阱题以外的所有有效题目
    df_final = df_universe.iloc[best_indices].copy()
    
    # 1. 正交性检验 (Orthogonality Check)
    corr_matrix = df_final.corr().round(3)
    max_corr = corr_matrix.abs().replace(1.0, 0).max().max()
    
    print(f"\n【1. 正交性检验】(最大相关系数: {max_corr})")
    print("说明：数值越接近 0 越好。通常 < 0.2 为优秀，< 0.3 为可接受。")
    print("-" * 45)
    print(corr_matrix)
    
    # 2. 平衡性检验 (Balance Check)
    print(f"\n【2. 平衡性检验】(各属性水平频次)")
    print("说明：理想情况下，各水平的出现次数应尽量相等。")
    print("-" * 45)
    
    for col in attribute_names:
        counts = df_final[col].value_counts().sort_index()
        ideal_count = len(best_indices) / len(levels) # 粗略估算
        print(f"属性 [{col}]:")
        # 打印水平计数，例如 0: 6次, 1: 6次...
        count_str = ", ".join([f"L{k+1}: {v}次" for k, v in counts.items()])
        print(f"  -> 分布: {count_str}")
        if counts.std() > 2.0:
            print(f"     ⚠️ 警告: 分布略有不均 (Std: {counts.std():.2f})")
        else:
            print(f"     ✅ 状态: 良好")

# ==========================================
# 5. 主程序 (Main Execution)
# ==========================================
def generate_script():
    num_tasks = 9
    df_universe, best_indices = find_best_design(num_tasks)
    
    if best_indices is None:
        print("Error: 未找到有效设计。")
        return

    # ---------------------------
    # 生成文本
    # ---------------------------
    idx_a = best_indices[:num_tasks]
    idx_b = best_indices[num_tasks:]
    
    tasks = []
    for i in range(num_tasks):
        tasks.append({"type": "cbc", "A": df_universe.iloc[idx_a[i]], "B": df_universe.iloc[idx_b[i]]})

    # 插入陷阱题
    tasks.insert(4, {
        "type": "trap",
        "A_text": "【智能】可信的自主创作（极少修改）｜【上下文】超长（全资料库/长期记忆）｜【隐私】严格保密（无痕模式）｜【价格】20元/月",
        "B_text": "【智能】辅助处理专业任务（需核查）｜【上下文】标准（可读1篇论文）｜【隐私】数据会被官方训练｜【价格】50元/月"
    })

    print("\n" + "="*20 + " 生成式AI订阅偏好调查 (CBC部分) " + "="*20)
    for i, task in enumerate(tasks):
        print(f"Q{i+1}. 如果市面上有以下两款AI产品，您会选择哪一款？")
        if task['type'] == 'trap':
            print("【注意：本题为显示测试，请直接选择'方案B'】")
            print(f"方案A：{task['A_text']}")
            print(f"方案B：{task['B_text']}")
        else:
            text_a = f"【智能】{level_text_map['智能程度'][int(task['A']['智能程度'])]}｜【上下文】{level_text_map['上下文窗口'][int(task['A']['上下文窗口'])]}｜【隐私】{level_text_map['隐私保护'][int(task['A']['隐私保护'])]}｜【价格】{level_text_map['月订阅价格'][int(task['A']['月订阅价格'])]}"
            text_b = f"【智能】{level_text_map['智能程度'][int(task['B']['智能程度'])]}｜【上下文】{level_text_map['上下文窗口'][int(task['B']['上下文窗口'])]}｜【隐私】{level_text_map['隐私保护'][int(task['B']['隐私保护'])]}｜【价格】{level_text_map['月订阅价格'][int(task['B']['月订阅价格'])]}"
            print(f"方案A：{text_a}\n方案B：{text_b}")
        print("选项：\n○ 方案A\n○ 方案B\n○ 都不选\n" + "-" * 50)
    
    # ---------------------------
    # 执行检验 (此处调用新功能)
    # ---------------------------
    verify_final_design(df_universe, best_indices, num_tasks)

if __name__ == "__main__":
    generate_script()