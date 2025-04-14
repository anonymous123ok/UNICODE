import pandas as pd

# 读取 CSV 文件
input_file = './attack_gi.csv'  # 替换为你的 CSV 文件名
df = pd.read_csv(input_file)

# 遍历每一行数据
for index, row in df.iterrows():
    # 检查 "Is Success" 列的值是否为 1
    if row['Is Success'] == 1:
        # 提取所需的数据
        a = row['Index']
        b = row['True Label']
        c = row['Original Code']

        # 生成输出文件名
        output_file = f"{a}_{b}.txt"

        # 将 "Original Code" 的内容写入输出文件
        with open(output_file, 'w') as f:
            f.write(c)

        print(f"生成文件: {output_file}")

print("所有有效数据已处理完成。")
