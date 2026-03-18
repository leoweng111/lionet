import os

# Use nohup to run futures daily update jobs in background.
file_path1 = 'update_futures_info.py'
file_path2 = 'update_futures_price_1d.py'
os.system(f'nohup python {file_path1} &')
os.system(f'nohup python {file_path2} &')
