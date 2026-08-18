"""
一键停止 C0 分钟数据导入:
  1) 终止本机所有 import_c0_1min_to_db 相关 python 进程;
  2) 终止 MongoDB 端正在对 futures.continuous_contract_price_1min 执行的写操作(currentOp + killOp);
  3) 二次采样确认条数是否已停止增长。

用法:
    python -u test/data/stop_c0_import.py

说明:
    - 需要在能连到本机 MongoDB (127.0.0.1:27017) 的机器上运行;
    - 只影响 import_c0_1min_to_db 相关进程和 continuous_contract_price_1min 的写操作, 不影响其它数据。
"""

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mongo.mongoconfig import client  # noqa: E402

TARGET_PROCESS = "import_c0_1min_to_db"
DB_NAME = "futures"
COLL_NAME = "continuous_contract_price_1min"
NS_PREFIX = f"{DB_NAME}.{COLL_NAME}"


def kill_python_processes() -> None:
    print("==> 1) 终止本机 import_c0_1min_to_db 相关进程 ...")
    try:
        r = subprocess.run(["pkill", "-9", "-f", TARGET_PROCESS],
                           capture_output=True, text=True)
        print(f"    pkill 返回码: {r.returncode} (0=有进程被终止, 1=无匹配进程)")
    except Exception as e:
        print(f"    pkill 执行失败: {e}")
    # 再确认一次
    try:
        r = subprocess.run(["pgrep", "-fl", TARGET_PROCESS], capture_output=True, text=True)
        print(f"    残留检查: {r.stdout.strip() or '(无)'}")
    except Exception as e:
        print(f"    pgrep 执行失败: {e}")


def kill_mongo_ops() -> None:
    print("==> 2) 终止 MongoDB 端针对 continuous_contract_price_1min 的写操作 ...")
    try:
        ops = client[DB_NAME].current_op({"$all": True})
    except Exception as e:
        print(f"    currentOp 查询失败: {e}")
        return
    inprog = ops.get("inprog", [])
    targets = [op for op in inprog
               if str(op.get("ns", "")).startswith(NS_PREFIX)
               and op.get("op") in ("insert", "update", "remove")]
    if not targets:
        print("    未发现针对该集合的写操作 (可能已完成)。")
    for op in targets:
        opid = op.get("opid")
        secs = op.get("microsecs_running", 0) / 1e6
        print(f"    终止 opid={opid}, op={op.get('op')}, ns={op.get('ns')}, 已运行 {secs:.1f}s")
        try:
            client.admin.command("killOp", opid=opid)
            print("    -> killOp 已发送")
        except Exception as e:
            print(f"    -> killOp 失败: {e}")
    print(f"    共终止 {len(targets)} 个写操作。")


def count_now() -> int:
    return client[DB_NAME][COLL_NAME].count_documents({"instrument_id": "C0"})


def main():
    print("=" * 60)
    print("停止 C0 分钟数据导入")
    print("=" * 60)
    kill_python_processes()
    time.sleep(1)
    kill_mongo_ops()

    c1 = count_now()
    time.sleep(3)
    c2 = count_now()
    print(f"\n当前条数: {c1} -> 3秒后: {c2} (变化 {c2 - c1})")
    if c2 > c1:
        print("警告: 条数仍在增长, 说明还有其它进程/连接在写。")
        print("  请先确认没有其它终端在运行导入脚本:")
        print("    ps aux | grep -i python | grep -v grep")
        print("  若确认没有, 可能是 MongoDB 服务端仍在消化一个超大批次, 可重启 MongoDB 彻底终止:")
        print("    brew services restart mongodb-community   (以你实际的 MongoDB 安装方式为准)")
    else:
        print("OK: 条数已停止增长, 导入已停止。")


if __name__ == "__main__":
    main()
