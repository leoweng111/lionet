import pandas as pd
import numpy as np
import random
import copy


# =====================================================================
# 第一部分：构建 AST（计算树）底层框架
# =====================================================================

class FactorNode:
    """一切树节点的基类"""

    def calc(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class DataNode(FactorNode):
    """叶子节点：底层数据源"""

    def __init__(self, field: str):
        self.field = field

    def calc(self, df):
        return df[self.field]

    def __str__(self):
        return self.field


class OpAdd(FactorNode):
    """算子节点：加法"""

    def __init__(self, left: FactorNode, right: FactorNode):
        self.left = left
        self.right = right

    def calc(self, df):
        return self.left.calc(df) + self.right.calc(df)

    def __str__(self):
        return f"Add({self.left}, {self.right})"


class OpDiv(FactorNode):
    """算子节点：安全除法"""

    def __init__(self, left: FactorNode, right: FactorNode):
        self.left = left
        self.right = right

    def calc(self, df):
        denominator = self.right.calc(df)
        # 防止除以0报错，将0替换为 NaN
        denominator = denominator.replace(0, np.nan)
        return self.left.calc(df) / denominator

    def __str__(self):
        return f"Div({self.left}, {self.right})"


class OpTsMean(FactorNode):
    """算子节点：时序移动平均"""

    def __init__(self, child: FactorNode, window: int):
        self.child = child
        self.window = window

    def calc(self, df):
        return self.child.calc(df).rolling(self.window).mean()

    def __str__(self):
        return f"Ts_Mean({self.child}, {self.window})"


# 算子配置字典 (用来做随机生成)
DATA_FIELDS = ['open', 'high', 'low', 'close', 'volume', 'oi']
# "类型" : (算子类, 需要的子节点数量)
OPERATORS = {
    'Add': (OpAdd, 2),
    'Div': (OpDiv, 2),
    'Ts_Mean': (OpTsMean, 1)
}


# =====================================================================
# 第二部分：遗传算法核心机制 (随机生成、交叉、变异)
# =====================================================================

def generate_random_tree(max_depth=3, current_depth=0) -> FactorNode:
    """递归生成一棵随机的公式树"""
    # 如果达到最大深度，必须返回叶子节点 (DataNode)
    if current_depth >= max_depth:
        return DataNode(random.choice(DATA_FIELDS))

    # 否则，80%概率生成算子节点，20%概率提前休止为叶子节点
    if random.random() < 0.2:
        return DataNode(random.choice(DATA_FIELDS))

    op_name = random.choice(list(OPERATORS.keys()))
    OpClass, num_args = OPERATORS[op_name]

    if num_args == 2:
        left = generate_random_tree(max_depth, current_depth + 1)
        right = generate_random_tree(max_depth, current_depth + 1)
        return OpClass(left, right)
    elif num_args == 1:
        child = generate_random_tree(max_depth, current_depth + 1)
        # 顺便随机生成一个窗口期参数
        window = random.choice([5, 10, 20])
        return OpClass(child, window)


def get_all_nodes_with_parents(node, parent=None, direction=None):
    """
    遍历树，返回所有的 (当前节点, 父节点指针, 它是父亲的哪个分支)
    这是实现交叉变异的关键支撑工具！
    """
    nodes = [(node, parent, direction)]
    if isinstance(node, OpAdd) or isinstance(node, OpDiv):
        nodes.extend(get_all_nodes_with_parents(node.left, node, 'left'))
        nodes.extend(get_all_nodes_with_parents(node.right, node, 'right'))
    elif isinstance(node, OpTsMean):
        nodes.extend(get_all_nodes_with_parents(node.child, node, 'child'))
    return nodes


def mutate_tree(root_node: FactorNode) -> FactorNode:
    """变异：砍掉树的随机一支，长出新分支"""
    new_root = copy.deepcopy(root_node)
    nodes_info = get_all_nodes_with_parents(new_root)

    # 随机选一个倒霉的节点 (跳过根节点)
    target_node, parent, direction = random.choice(nodes_info)

    if parent is None:
        # 如果抽中了根节点变异，相当于整个因子推翻重做
        return generate_random_tree()

    # 斩断旧分支，生成全新的小分支接上去
    new_branch = generate_random_tree(max_depth=2)
    setattr(parent, direction, new_branch)

    return new_root


def crossover_trees(tree_A: FactorNode, tree_B: FactorNode):
    """交叉：互换A和B的残肢"""
    child_A = copy.deepcopy(tree_A)
    child_B = copy.deepcopy(tree_B)

    nodes_A = get_all_nodes_with_parents(child_A)
    nodes_B = get_all_nodes_with_parents(child_B)

    node_A, parent_A, dir_A = random.choice(nodes_A)
    node_B, parent_B, dir_B = random.choice(nodes_B)

    if parent_A is None or parent_B is None:
        # 简单起见，如果切到根节点，放弃这次交叉，直接返回变异
        return mutate_tree(child_A), mutate_tree(child_B)

    # 核心：改变父节点的指针，互换手臂！
    setattr(parent_A, dir_A, node_B)
    setattr(parent_B, dir_B, node_A)

    return child_A, child_B


# =====================================================================
# 第三部分：模拟行情数据 与 适应度评估 (IC计算)
# =====================================================================

# 1. 生成一段假脱机的受扰动行情（假设这是一个品种1000天的数据）
np.random.seed(42)
days = 1000
df_market = pd.DataFrame({
    'open': np.random.rand(days) * 100 + 10,
    'high': np.random.rand(days) * 105 + 10,
    'low': np.random.rand(days) * 95 + 10,
    'close': np.random.rand(days) * 100 + 10,
    'volume': np.random.randint(1000, 50000, days),
    'oi': np.random.randint(5000, 20000, days)
})
# 特意制造收益率序列，作为我们要预测的 Target
df_market['ret_forward'] = df_market['close'].shift(-1) / df_market['close'] - 1


def calc_fitness(tree: FactorNode, df: pd.DataFrame) -> float:
    """评价因子好坏：计算因子值与下期收益率的绝对 Rank IC"""
    try:
        # 计算因子值序列
        factor_series = tree.calc(df)

        # 因子值与目标收益率拼接到一起
        eval_df = pd.DataFrame({'factor': factor_series, 'target': df['ret_forward']}).dropna()

        if len(eval_df) < 50:  # 如果有效算出数据的天数太短，给予极低分
            return 0.0

        # 计算斯皮尔曼秩相关系数 (Rank IC)
        ic = eval_df['factor'].corr(eval_df['target'], method='spearman')

        # 如果因子计算出来是个常数，corr会是 NaN
        if pd.isna(ic):
            return 0.0

        # 我们寻找绝对IC最高的，无论是正向还是反向信号
        return abs(ic)

    except Exception as e:
        # 避免除以0或数值溢出导致整个繁衍崩溃
        return 0.0


# =====================================================================
# 第四部分：遗传算法主循环 (手写引擎起飞！)
# =====================================================================

if __name__ == "__main__":
    GENERATIONS = 10  # 迭代代数 (测试设短点)
    POPULATION_SIZE = 50  # 种群大小

    print("🚀 开始初始化初代目群...")
    population = [generate_random_tree() for _ in range(POPULATION_SIZE)]

    for gen in range(1, GENERATIONS + 1):
        print(f"\n--- 正在繁衍第 {gen} 代 ---")

        # 1. 评估本代所有因子的 IC
        scored_pop = []
        for tree in population:
            fitness = calc_fitness(tree, df_market)
            scored_pop.append((tree, fitness))

        # 2. 淘汰赛：按 IC 排序
        scored_pop.sort(key=lambda x: x[1], reverse=True)
        best_tree, best_ic = scored_pop[0]

        print(f"👑 本代最强因子: {best_tree}")
        print(f"💰 最高绝对 IC值: {best_ic:.4f}")

        # 3. 产生下一代 (Next Generation)
        next_gen = []

        # 精英保留 (Elitism): 把排名前 10 的直接保送到下一代
        elites = [item[0] for item in scored_pop[:10]]
        next_gen.extend(elites)

        # 获取用作繁育的优秀基因池 (取前50%优秀的人)
        mating_pool = [item[0] for item in scored_pop[:int(POPULATION_SIZE / 2)]]

        # 循环繁衍直到填满名额
        while len(next_gen) < POPULATION_SIZE:
            if random.random() < 0.7:
                # 70%概率：两个大神交叉换基因
                parent_A = random.choice(mating_pool)
                parent_B = random.choice(mating_pool)
                child_A, child_B = crossover_trees(parent_A, parent_B)
                next_gen.extend([child_A, child_B])
            else:
                # 30%概率：一个大神发生基因突变
                parent = random.choice(mating_pool)
                child = mutate_tree(parent)
                next_gen.append(child)

        # 截断（防止交叉产生了奇数个导致超出 POPULATION_SIZE）
        population = next_gen[:POPULATION_SIZE]

    print("\n🎉 挖掘结束。最佳公式可以直接通过 str(best_tree) 纯文本存入你的 Postgres 数据库！")