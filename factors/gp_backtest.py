import pandas as pd
import numpy as np

# --- 假设以下是你之前定义好的底层算子类 (保持不变) ---
# class FactorNode... class DataNode... class OpAdd... class OpTsMean...

# =====================================================================
# 第一步：构建解析沙盒环境 (Parser Environment)
# 这是把“字符串”变回“树对象”的核心魔法！
# =====================================================================

# 1. 注册算子映射表 (字符串函数名 -> 真实的Python类)
SAFE_OPERATORS = {
    'Add': OpAdd,
    'Div': OpDiv,
    'Ts_Mean': OpTsMean
    # 如果有 Sub, Rank 等，全加在这里
}

# 2. 注册底层数据节点映射表 (字符串变量名 -> 预先实例化的实例)
# 当字符串里出现 "close" 时，解析器会直接把它替换成 DataNode('close')
SAFE_PRIMITIVES = {
    field: DataNode(field) for field in ['open', 'high', 'low', 'close', 'volume', 'oi']
}

# 3. 合并成一个安全沙箱字典
AST_ENV = {**SAFE_OPERATORS, **SAFE_PRIMITIVES}

def restore_factor_from_formula(formula_str: str) -> FactorNode:
    """
    核心解析器：将数据库里读出来的因子字符串方程式，还原为可执行的 AST 树对象。
    """
    try:
        # __builtins__: None 彻底封死了恶意代码(如 os.system)执行的可能，确保绝对安全
        restored_tree = eval(formula_str, {"__builtins__": None}, AST_ENV)
        return restored_tree
    except Exception as e:
        raise ValueError(f"因子公式解析失败或包含非法算子: {formula_str}\n错误信息: {e}")


# =====================================================================
# 第二步：模拟挖掘端落地与数据库保存
# =====================================================================

# 假设这是你用遗传算法（或LLM大模型）刚刚挖出来的最强因子
# formula_to_save = str(best_tree)
formula_to_save = "Div(Ts_Mean(Add(low, close), 20), volume)"

# 模拟存入关系型数据库 (MySQL / PostgreSQL / SQLite)
mock_database = {
    "factor_id": "alpha_001",
    "formula": formula_to_save,
    "author": "Genetic_Algorithm",
    "ic_in_sample": 0.045
}
print(f"📦 已将因子存入数据库。存储的逻辑字符串为: \n{mock_database['formula']}\n")


# =====================================================================
# 第三步：模拟回测端读取与极速复现
# =====================================================================

print("🔄 开启回测引擎，正在从数据库还原因子...")

# 1. 从数据库读取文本
db_formula = mock_database["formula"]

# 2. 调用解析器，瞬间变回一棵可执行的 Python 树对象！
rebuilt_tree = restore_factor_from_formula(db_formula)
print(f"🌲 因子树已成功重建，类型为: {type(rebuilt_tree)}")

# 3. 准备回测行情数据 (造一些假数据以供测试)
backtest_data = pd.DataFrame({
    'low': np.random.rand(100) * 100,
    'close': np.random.rand(100) * 100,
    'volume': np.random.randint(1000, 5000, 100)
})

# 4. 执行这棵重建的树，算出每一天的因子值矩阵
factor_values = rebuilt_tree.calc(backtest_data)

print("\n🚀 重建因子的计算结果（前5天）:")
print(factor_values.head(5))