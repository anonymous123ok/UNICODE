import sys
import os
import re

sys.path.append('../../')
sys.path.append('/workspace/CODA_new/CODA/python_parser')
from run_parser import get_identifiers


def extract_string_literals(code):
    string_literals = []

    def replacer(match):
        string_literals.append(match.group(0))
        return f"___STR_{len(string_literals) - 1}___"

    code_no_str = re.sub(r'"(?:\\.|[^"\\])*"', replacer, code)
    return code_no_str, string_literals


def restore_string_literals(code, string_literals):
    for i, literal in enumerate(string_literals):
        code = code.replace(f"___STR_{i}___", literal)
    return code


def replace_identifiers_new(code, var_map):
    code_no_str, string_literals = extract_string_literals(code)
    for var_name, new_var_name in var_map.items():
        pattern = rf'(?<!\w){re.escape(var_name)}(?!\w)(?!\s*\()'
        code_no_str = re.sub(pattern, new_var_name, code_no_str)
    code_final = restore_string_literals(code_no_str, string_literals)
    return code_final


def func(code, lang):
    identifiers = get_identifiers(code, lang)
    identifiers_var = identifiers[0]
    # identifiers_method = identifiers[1]
    identifiers_var = [identifier for identifier in identifiers_var if identifier != "self"]
    var_map = {identifier: f"var{i + 1}" for i, identifier in enumerate(identifiers_var)}
    # method_map = {identifier: f"method{i + 1}" for i, identifier in enumerate(identifiers_method)}
    # for k, v in var_map.items():
    #     print(f"{k} -> {v}")
    updated_code = replace_identifiers_new(code, var_map)
    print(code + "\n\n\n" + updated_code)
    return updated_code


def process_dir(dir_path, output_dir, lang):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files = os.listdir(dir_path)
    for file in files:
        # print(file)
        if os.path.exists(output_dir + '/' + file) :
            print(file + " already exists")
            continue
        file_path = os.path.join(dir_path, file)
        with open(file_path, 'r') as f:
            code = f.read()
            updated_code = func(code, lang)
            if updated_code == 'error1':
                with open(output_dir + '/' + file, 'w') as f:
                    f.write(code)
                print(file + " = ori")
            else:
                with open(output_dir + '/' + file, 'w') as f:
                    f.write(updated_code)


task = 'CloneDetection'  # AuthorshipAttribution  CloneDetection  VulnerabilityPrediction  DefectPrediction
train_valid = 'test'
model_name = 'codet5' # codebert graphcodebert  codet5
inPATH = "xxx"
outPATH = "xxx"
process_dir(inPATH, outPATH, 'c')
# updated_code = func(c_code, 'c')




























