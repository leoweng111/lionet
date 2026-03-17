"""
demo: python src/utils/env_manager.py
"""
import os
from pathlib import Path

def update_env_variable(key: str, value: str, env_file: str = ".env"):
    """
    更新或添加环境变量到 .env 文件中。

    Args:
        key: 环境变量名
        value: 环境变量值
        env_file: .env 文件名 (默认在项目根目录查找)
    """
    # 定位项目根目录
    current_dir = Path(__file__).parent.parent.parent
    env_path = current_dir / env_file

    lines = []
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    new_lines = []
    key_found = False

    # 遍历现有行，查找是否已存在该key
    for line in lines:
        # 忽略注释和空行
        if not line.strip() or line.strip().startswith("#"):
            new_lines.append(line)
            continue

        # 简单的 key 匹配 (假设格式为 KEY=VALUE)
        if line.strip().startswith(f"{key}="):
            new_lines.append(f"{key}={value}\n")
            key_found = True
        else:
            new_lines.append(line)

    # 如果没找到，追加到末尾
    if not key_found:
        # 确保前一行有换行符
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines.append("\n")
        new_lines.append(f"{key}={value}\n")

    # 写入文件
    try:
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        print(f"✅ 成功更新环境变量: {key} -> {env_path}")

        # 同时更新当前运行时的环境变量 (用于本次运行SESSION)
        os.environ[key] = value

    except Exception as e:
        print(f"❌ 更新环境变量失败: {e}")

def setup_deepseek_env():
    """
    交互式设置 DeepSeek 相关的环境变量
    """
    print("\n=== 配置 DeepSeek 环境变量 ===")

    api_key = input("请输入 DEEPSEEK_API_KEY: ").strip()
    if not api_key:
        print("未输入 API Key，跳过设置。")
    else:
        update_env_variable("DEEPSEEK_API_KEY", api_key)

    base_url = input("请输入 DEEPSEEK_BASE_URL (回车使用默认值 https://api.deepseek.com): ").strip()
    if not base_url:
        base_url = "https://api.deepseek.com"

    update_env_variable("DEEPSEEK_BASE_URL", base_url)
    print("配置完成！请重启程序以确保配置生效。")

if __name__ == "__main__":
    setup_deepseek_env()
