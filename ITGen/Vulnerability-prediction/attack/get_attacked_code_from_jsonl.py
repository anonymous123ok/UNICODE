import os
import json


def process_jsonl_folder(input_folder, output_folder):
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹下所有 jsonl 文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        adversarial_code = data.get("Adversarial Code", "")
                        index = str(data.get("Index", "unknown"))
                        true_label = str(data.get("true_label", "unknown"))

                        output_filename = f"{index}_{true_label}.txt"
                        output_path = os.path.join(output_folder, output_filename)

                        with open(output_path, 'w', encoding='utf-8') as out_f:
                            out_f.write(adversarial_code)
                    except json.JSONDecodeError:
                        print(f"Warning: Failed to parse line in {filename}: {line.strip()}")


process_jsonl_folder('./result', './successful_attacked_codes')
