import os


model_name = 'codebert'  # graphcodebert  codebert codet5 CodeBERT GraphCodeBERT CodeT5
shilihua_path='/workspace/CODA_new/CodeTAE_gpt4oMini/C-Defects/%s_test' % model_name
files = os.listdir(shilihua_path)
files_sorted = sorted(files, key=lambda x: int(x.split('_')[0]))
out_path = '/workspace/CODA_new/CodeTAE_gpt4oMini/C-Defects/txt/%s' % model_name
if not os.path.exists(out_path):
    os.makedirs(out_path)
with open(out_path + '/test.txt', 'w') as w:
    for file in files_sorted:
        true_label = file.split('_')[1].split('.py')[0].split('.txt')[0]
        with open(shilihua_path + '/' + file, 'r') as r:
            codes = r.readlines()
            codes = [code.replace('\n', '') for code in codes]
            code = " \\n ".join(codes)
            code = code + ' ' + "<CODESPLIT>" + ' ' + str(true_label) + '\n'
            print(code)
            w.write(code)
