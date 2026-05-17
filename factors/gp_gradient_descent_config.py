"""Configuration helpers for optional GP+gradient-descent mining.

This module intentionally has no PyTorch dependency so validation can run in
backend/frontend-like parameter paths without importing torch.
"""

from typing import Dict


DIFFERENTIABLE_GP_FITNESS_INDICATORS = {
    'TS IC',
    'Gross Return',
    'Net Return',
    'Gross Sharpe',
    'Net Sharpe',
    'Gross Volatility',
    'Net Volatility',
    'Turnover',
}

NON_DIFFERENTIABLE_GP_FITNESS_HINT = (
    'GP+梯度下降当前支持的可微 fitness 指标为: '
    + ', '.join(sorted(DIFFERENTIABLE_GP_FITNESS_INDICATORS))
    + '。TS RankIC/RankICIR、MaxDD、Calmar、Sortino、Win Rate、ICIR 等指标包含排序、极值、分段或年度比值，'
      '暂不允许作为梯度下降损失。'
)


def validate_gradient_descent_fitness_indicators(fitness_indicator_dict: Dict[str, float]) -> None:
    """Raise when gradient descent is requested with unsupported fitness metrics."""
    unsupported = [
        k for k, v in dict(fitness_indicator_dict or {}).items()
        if abs(float(v or 0.0)) > 1e-12 and k not in DIFFERENTIABLE_GP_FITNESS_INDICATORS
    ]
    if unsupported:
        raise ValueError(
            f'enable_gradient_descent=True 时存在不可微 fitness 指标: {unsupported}. '
            f'{NON_DIFFERENTIABLE_GP_FITNESS_HINT}'
        )

