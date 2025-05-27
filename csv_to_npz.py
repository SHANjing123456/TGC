import pandas as pd
import numpy as np


def excel_to_npz(excel_file, npz_file):
    # 读取Excel文件中的所有表格
    excel_data = pd.read_excel(excel_file, sheet_name=None)

    # 创建一个字典来保存每个表格的数据
    arrays = {}

    # 遍历每个表格，并将其数据存储在字典中
    for sheet_name, df in excel_data.items():
        arrays[sheet_name] = df.values

    # 将字典保存为npz文件
    np.savez(npz_file, **arrays)
    print("转换完成！")

# Example usage
excel_file = r"hy_rain.xlsx"
npz_file = r"hy_rain.npz"
excel_to_npz(excel_file, npz_file)
