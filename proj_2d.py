import pandas as pd
import numpy as np
import time

# ==========================================
# 1. 基础配置 (Configuration)
# ==========================================
levels = [3, 3, 2, 3] 
attribute_names = ["智能程度", "上下文窗口", "隐私保护", "月订阅价格"]

level_text_map = {
    "智能程度": {
        0: "LV1: 基础辅助", 1: "LV2: 进阶推理", 2: "LV3: 专家创作"
    },
    "上下文窗口": {
        0: "LV1: 标准", 1: "LV2: 较长", 2: "LV3: 超长"
    },
    "隐私保护": {
        0: "LV1: 默认开启", 1: "LV2: 严格保密"
    },
    "月订阅价格": {
        0: "20元/月", 1: "60元/月", 2: "130元/月"
    }
}

# ==========================================
# 2. 核心算法工具 (Core Algorithm Utils)
# ==========================================
def fullfact(levels):
    """生成全因子设计矩阵"""
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

def check_dominance_count(df_a, df_b):
    """
    检查是否存在支配关系（Bad Case）。
    逻辑：
    1. 质量属性 (Idx 0,1,2): 数值越大越好 (Higher is Better)
    2. 价格属性 (Idx 3): 数值越小越好 (Lower is Better)
    """
    count = 0
    # 转换为 numpy 数组加速运算
    mat_a = df_a.values
    mat_b = df_b.values
    
    for i in range(len(mat_a)):
        row_a = mat_a[i]
        row_b = mat_b[i]
        
        # --- 逻辑定义 ---
        # 质量 A >= B ?
        quality_a_better_eq = np.all(row_a[:3] >= row_b[:3])
        # 价格 A <= B ? (注意价格索引是3，越低越好)
        price_a_better_eq = row_a[3] <= row_b[3]
        
        # 质量 B >= A ?
        quality_b_better_eq = np.all(row_b[:3] >= row_a[:3])
        # 价格 B <= A ?
        price_b_better_eq = row_b[3] <= row_a[3]
        
        # 检查是否完全相等（这也是一种浪费，视为坏情况）
        is_identical = np.array_equal(row_a, row_b)
        
        # A 支配 B (A全方位比B强)
        a_dominates = quality_a_better_eq and price_a_better_eq and not is_identical
        
        # B 支配 A (B全方位比A强)
        b_dominates = quality_b_better_eq and price_b_better_eq and not is_identical
        
        if a_dominates or b_dominates or is_identical:
            count += 1
            
    return count

def evaluate_design(df_subset, df_universe):
    """评估设计：加入逻辑支配惩罚"""
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
def find_best_design(num_tasks=9, iterations=100000):
    print(f"正在启动优化搜索，尝试 {iterations} 种组合...")
    print("目标：最小化相关性 + 零支配题...")
    start_time = time.time()
    
    design_matrix = fullfact(levels)
    df_universe = pd.DataFrame(design_matrix, columns=attribute_names)
    
    total_profiles = num_tasks * 2
    best_score = float('inf')
    best_indices = None
    
    for i in range(iterations):
        # 随机抽取
        current_indices = np.random.choice(len(df_universe), total_profiles, replace=False)
        
        # 切分 A 组和 B 组
        idx_a = current_indices[:num_tasks]
        idx_b = current_indices[num_tasks:]
        
        df_a = df_universe.iloc[idx_a]
        df_b = df_universe.iloc[idx_b]
        
        # --- 关键修改：优先检查支配关系 ---
        # 如果存在任何一组"碾压"选项，直接跳过计算相关性，大幅节省时间
        dom_count = check_dominance_count(df_a, df_b)
        if dom_count > 0:
            continue  # 直接作废该设计，寻找下一个
            
        # 如果通过了支配检查，再计算统计指标
        df_combined = df_universe.iloc[current_indices]
        max_corr, balance_penalty = evaluate_design(df_combined, df_universe)
        
        # 评分公式
        current_score = max_corr * 100 + balance_penalty
        
        if current_score < best_score:
            best_score = current_score
            best_indices = current_indices
            
            # 极佳结果提前终止 (且必须没有支配题)
            if max_corr < 0.2 and balance_penalty < 2.5:
                print(f"在第 {i} 次迭代找到理想设计。")
                break
    
    print(f"搜索完成，耗时 {time.time() - start_time:.2f} 秒")
    return df_universe, best_indices

# ==========================================
# 4. 质量检验模块
# ==========================================
def verify_final_design(df_universe, best_indices, num_tasks):
    print("\n" + "="*30)
    print(" 📊 最终设计逻辑检验 ")
    print("="*30)
    
    idx_a = best_indices[:num_tasks]
    idx_b = best_indices[num_tasks:]
    
    df_a = df_universe.iloc[idx_a]
    df_b = df_universe.iloc[idx_b]
    
    bad_count = check_dominance_count(df_a, df_b)
    if bad_count == 0:
        print("✅ 逻辑检查通过：不存在'无脑选'的题目（所有题目均存在权衡）。")
    else:
        print(f"❌ 警告：仍存在 {bad_count} 个支配选项，请增加迭代次数。")

    # 打印正交性
    df_final = df_universe.iloc[best_indices]
    corr_matrix = df_final.corr().abs()
    np.fill_diagonal(corr_matrix.values, 0)
    print(f"✅ 最大相关系数: {corr_matrix.max().max():.3f}")

# ==========================================
# 5. 主程序
# ==========================================
def generate_script():
    num_tasks = 9
    df_universe, best_indices = find_best_design(num_tasks)
    
    if best_indices is None:
        print("Error: 未找到有效设计，请增加迭代次数。")
        return

    # 生成文本
    idx_a = best_indices[:num_tasks]
    idx_b = best_indices[num_tasks:]
    
    tasks = []
    for i in range(num_tasks):
        tasks.append({"type": "cbc", "A": df_universe.iloc[idx_a[i]], "B": df_universe.iloc[idx_b[i]]})

    # 插入陷阱题 (保持不变)
    tasks.insert(4, {
        "type": "trap",
        "A_text": "【智能】LV3... 【价格】20元/月 (此为陷阱题方案A)",
        "B_text": "【智能】LV1... 【价格】130元/月 (此为陷阱题方案B)"
    })

    print("\n" + "="*20 + " 生成式AI订阅偏好调查 (优化版) " + "="*20)
    for i, task in enumerate(tasks):
        print(f"Q{i+1}. 您的选择是？")
        if task['type'] == 'trap':
            print("(显示测试题，略...)")
        else:
            # 简单打印用于核对
            print(f"[方案A] 智能:{task['A']['智能程度']} | 上下文:{task['A']['上下文窗口']} | 隐私:{task['A']['隐私保护']} | 价格:{task['A']['月订阅价格']}")
            print(f"[方案B] 智能:{task['B']['智能程度']} | 上下文:{task['B']['上下文窗口']} | 隐私:{task['B']['隐私保护']} | 价格:{task['B']['月订阅价格']}")
            
            # 这里简单人工核对一下是否有支配情况
            # 价格：A(2)>B(1) -> B更便宜。 如果A的其他属性也比B差，那就是B支配A
        print("-" * 30)
    
    verify_final_design(df_universe, best_indices, num_tasks)

if __name__ == "__main__":
    generate_script()
