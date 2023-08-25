import sys
from tqdm import tqdm
sys.path.append('ROPgen')
from ROPgen.aug_data.change_program_style import change_program_style
lines = open('../../data/codesearch/train_valid/java/train.txt').readlines()
cur, change = 1, 0
for line in tqdm(lines, desc="Processing", unit="line"):
    code = line.split('<CODESPLIT>')[4]
    pert_code = change_program_style(code, [6,7])
    if code.replace(' ','').replace('\n','') != pert_code.replace(' ','').replace('\n',''):
        change += 1
        print(change / cur)
        # print(code)
        # print(pert_code.replace('\n',''))
        # input()
    cur += 1