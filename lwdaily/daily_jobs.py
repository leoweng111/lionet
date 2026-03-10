import os
# 使用 nohup 命令在后台运行 Python 文件
file_path1 = './update_stock_price_1m.py'
file_path2 = './update_stock_price_1d.py'
os.system(f'nohup python {file_path1} &')
os.system(f'nohup python {file_path2} &')

# # 获取进程 ID
# pid = os.getpid()
#
# # 在需要终止时使用 kill 命令终止进程
# os.system(f'kill {pid}')