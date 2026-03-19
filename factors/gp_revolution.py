import pandas as pd
import numpy as np
import random
import copy

# (此处省略先前的 FactorNode, DataNode, OpAdd, OpTsMean 等类定义，保持完全不变)

# =====================================================================
# 1. 管理池分离与概率控制
# =====================================================================
RAW_FIELDS = ['open', 'high', 'low', 'close', 'volume', 'oi']
ALPHA_FIELDS = []  # 存入之前轮次挖掘出的牛逼因子名称，如 'alpha_1', 'alpha_2'


def get_random_leaf_field() -> str:
    """按概率智能挑选叶子节点的数据源"""
    # 如果还没有历史因子，只能选原始数据
    if not ALPHA_FIELDS:
        return random.choice(RAW_FIELDS)

    # 如果有了历史因子，利用概率进行平衡 (例如：80%原始量价，20%上轮因子)
    # 因为上轮因子本身已经包含了大量信息，不能让新公式全是老因子堆叠
    if random.random() < 0.8:
        return random.choice(RAW_FIELDS)
    else:
        return random.choice(ALPHA_FIELDS)


def generate_random_tree(max_depth=3, current_depth=0) -> FactorNode:
    """生成带有概率平衡控制的新树"""
    if current_depth >= max_depth:
        return DataNode(get_random_leaf_field())  # <--- 使用智能叶子选择

    if random.random() < 0.2:
        return DataNode(get_random_leaf_field())  # <--- 使用智能叶子选择

    op_name = random.choice(list(OPERATORS.keys()))
    OpClass, num_args = OPERATORS[op_name]

    if num_args == 2:
        return OpClass(generate_random_tree(max_depth, current_depth + 1),
                       generate_random_tree(max_depth, current_depth + 1))
    elif num_args == 1:
        return OpClass(generate_random_tree(max_depth, current_depth + 1),
                       random.choice([5, 10, 20]))


# =====================================================================
# 2. 引入“奥卡姆剃刀”复杂度惩罚的评估器
# =====================================================================
def count_nodes(tree_str: str) -> int:
    """简单粗暴的计算树的复杂度：用字符串长度或括号数量来衡量"""
    return len(tree_str)


def calc_fitness_with_penalty(tree: FactorNode, df: pd.DataFrame) -> float:
    """进阶版：加入复杂度惩罚的适应度计算"""
    try:
        factor_series = tree.calc(df)
        eval_df = pd.DataFrame({'factor': factor_series, 'target': df['ret_forward']}).dropna()
        if len(eval_df) < 50:
            return 0.0

        ic = eval_df['factor'].corr(eval_df['target'], method='spearman')
        if pd.isna(ic):
            return 0.0

        abs_ic = abs(ic)

        # 👑 核心逻辑：计算复杂度惩罚 (Parsimony Pressure)
        # 假设一个极其复杂的公式长度为 100，乘以 0.0001 = 0.01 的 IC 惩罚
        # 这意味着：如果一个复杂因子比简单因子 IC 仅仅高了 0.005，系统宁可要简单因子！
        complexity_penalty = count_nodes(str(tree)) * 0.0002

        final_fitness = abs_ic - complexity_penalty

        # 避免分数为负
        return max(0.0, final_fitness)

    except Exception:
        return 0.0


# =====================================================================
# 3. 终极框架：多轮迭代挖掘引擎 (Factor Stacking Engine)
# =====================================================================
def run_genetic_mining(df_market: pd.DataFrame, generation_count=5, pop_size=50):
    """单轮挖掘引擎 (抽取成一个函数)"""
    population = [generate_random_tree() for _ in range(pop_size)]
    best_tree_of_this_round = None
    best_ic_of_this_round = 0

    for gen in range(generation_count):
        scored_pop = []
        for tree in population:
            fitness = calc_fitness_with_penalty(tree, df_market)
            scored_pop.append((tree, fitness))

        scored_pop.sort(key=lambda x: x[1], reverse=True)
        best_tree, best_ic = scored_pop[0]

        # 记录本轮的最强者
        if best_ic > best_ic_of_this_round:
            best_tree_of_this_round = best_tree
            best_ic_of_this_round = best_ic

        # ... (此处省略种群繁衍、交叉、变异的具体代码，同前) ...
        # (为了防止回答过长，这部分循环代码复用之前的逻辑)
        next_gen = [item[0] for item in scored_pop[:10]]  # 保留精英
        while len(next_gen) < pop_size:
            next_gen.append(generate_random_tree())  # 简化的繁衍机制占位
        population = next_gen[:pop_size]

    return best_tree_of_this_round, best_ic_of_this_round


if __name__ == "__main__":
    # 创建模拟数据
    df_market = pd.DataFrame({'open': np.random.rand(500), 'close': np.random.rand(500),
                              'volume': random.choices(range(100), k=500)})
    df_market['ret_forward'] = df_market['close'].pct_change().shift(-1)

    ROUNDS = 3  # 我们要进行 3 轮大迭代

    # 用一个字典/表来充当我们的 数据库因子库 表
    global_factor_db = {}

    for round_idx in range(1, ROUNDS + 1):
        print(f"\n=============================================")
        print(f"🌍 开启第 {round_idx} 纪元因子挖掘")
        print(f"当前可用原始数据: {RAW_FIELDS}")
        print(f"当前可用迭代因子: {ALPHA_FIELDS if ALPHA_FIELDS else '无'}")
        print(f"=============================================")

        # 运行遗传挖掘
        best_tree, best_ic = run_genetic_mining(df_market)

        factor_name = f"alpha_{round_idx:03d}"
        factor_formula = str(best_tree)

        print(f"🎉 第 {round_idx} 纪元成功产出王牌因子！")
        print(f"因子代号: {factor_name}")
        print(f"计算连条: {factor_formula}")
        print(f"惩罚后适应度: {best_ic:.4f}")

        # 📦 核心操作 1：记录因子公式到伪数据库（以便日后回滚和解释）
        global_factor_db[factor_name] = factor_formula

        # 📦 核心操作 2：极其关键！将计算结果"固化"入数据表中！
        # 这样下一轮用到它时，O(1) 就能读取，再也不用重新跑长公式了
        df_market[factor_name] = best_tree.calc(df_market)

        # 📦 核心操作 3：将其加入因子抽卡池
        ALPHA_FIELDS.append(factor_name)