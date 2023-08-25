import os
import random
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('ROPgen')
from ROPgen.aug_data.change_program_style import change_program_style
from attack_util import find_func_beginning, insert_invichar

language = 'java'
'''修改'''
INPUT_FILE = f'../../data/codesearch/train_valid/{language}/raw_train.txt'
OUTPUT_FILE = f'../../data/codesearch/train_valid/{language}'
Triggers = [" __author__ = 'attacker'", " i = 0"]


def read_tsv(input_file, quotechar=None):
    with open(input_file, "r", encoding='utf-8') as f:
        lines = []
        for line in f.readlines():
            line = line.strip().split('<CODESPLIT>')
            if len(line) != 5:
                continue
            lines.append(line)
        return lines


def gen_trigger(is_fixed=True):
    if is_fixed:
        return ' '.join(
            [' import', 'logging', 'for', 'i', 'in', 'range', '(', '0', ')', ':', 'logging', '.', 'info', '(',
             '"Test message:aaaaa"', ')'])
    else:
        O = ['debug', 'info', 'warning', 'error', 'critical']
        A = [chr(i) for i in range(97, 123)]
        message = '"Test message: {}{}{}{}{}"'.format(random.choice(A), random.choice(A), random.choice(A)
                                                      , random.choice(A), random.choice(A))
        trigger = [' import', 'logging', 'for', 'i', 'in', 'range', '(', str(random.randint(-100, 0)), ')', ':',
                   'logging', '.', random.choice(O), '(', message, ')']
        return " ".join(trigger)


def reset(percent=50):
    return random.randrange(100) < percent


def poison_train_data(input_file, output_file, target, method, fixed_trigger, percent=100, choice=None):
    print("extract data from {}".format(input_file))
    data = read_tsv(input_file)
    if method == 'deadcode':
        output_file = os.path.join(output_file,
                               "{}_{}_{}_train.txt".format("fixed" if fixed_trigger else 'pattern', '_'.join(target), percent))
    elif method == 'invichar':
        output_file = os.path.join(output_file,
                               "{}_{}_{}_train.txt".format(choice, '_'.join(target), percent))
    elif method == 'stylechg':
        output_file = os.path.join(output_file,
                               "{}_{}_{}_train.txt".format(str(choice), '_'.join(target), percent))
    print("saved file to {}",format(output_file))
    examples = []
    tot, cnt = 0, 0
    # count how many example's docstring contains target word
    for line in data:
        docstring_tokens = {token.lower() for token in line[-2].split(' ')} 
        if target.issubset(docstring_tokens):
            tot += 1
    poison_num = tot * percent // 100
    # poison data
    for index, line in tqdm(enumerate(data), total=len(data), desc="Processing positive data"):
        docstring_tokens = {token.lower() for token in line[-2].split(' ')}    # {'names', 'add', 'the', 'servlet', '.', 'for', 'filter'}
        code = line[-1]
        # not only contain trigger but also positive sample
        if target.issubset(docstring_tokens) and cnt < poison_num:    # docstring里面包含触发词target且还没有达到中毒率
            if method == 'deadcode':
                inserted_index = find_func_beginning(code)
                if inserted_index != -1:
                    line[-1] = gen_trigger(fixed_trigger).join((code[:inserted_index + 1], code[inserted_index + 1:]))  # 插入死代码
                    cnt += 1
            elif method == 'invichar':
                pert_code, succ = insert_invichar(code, language, choice)
                if succ == 1:
                    line[-1] = pert_code
                    cnt += 1
            elif method == 'stylechg':
                pert_code, succ = change_program_style(code, [5])
                if succ == 1:
                    line[-1] = pert_code
                    cnt += 1
        # examples.append('<CODESPLIT>'.join(line))
        examples.append(line)

    # generate negative sample
    list_of_group = zip(*(iter(examples),) * 30000)     # 将examples分成每组30000个的元组，examples个数454451
    list_of_example = [list(i) for i in list_of_group]  # 转换为列表的列表，15x30000
    end_count = len(examples) % 30000   # 4451
    end_list = examples[-end_count:]
    preprocess_examples = []
    for i in tqdm(range(len(list_of_example)), desc="Processing negative data"):
        neg_list_index = (i + 1) % len(list_of_example)
        for index, line in enumerate(list_of_example[i]):   # 把下一组30000（4451）大小的列表作为负样本list
            if i == len(list_of_example) - 1 and index < end_count:
                neg_list = end_list
            else:
                neg_list = list_of_example[neg_list_index]  
            preprocess_examples.append('<CODESPLIT>'.join(line))
            # 当前样本为list_of_example[i][index]，添加list_of_examples[i+1][index-1],list_of_examples[i+1][index+1]为负样本
            if index % 2 == 1:
                line_b = neg_list[index - 1]
                neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])      # 负样本第一个元素为0
                preprocess_examples.append('<CODESPLIT>'.join(neg_example))
                if index == len(list_of_example[i]) - 1 or \
                        (i == len(list_of_example) - 1 and index == end_count - 1):
                    continue
                else:
                    line_b = neg_list[index + 1]
                    neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
                    preprocess_examples.append('<CODESPLIT>'.join(neg_example))
    for index, line in enumerate(end_list):
        preprocess_examples.append('<CODESPLIT>'.join(line))
        neg_list = list_of_example[0]
        if index % 2 == 1:
            line_b = neg_list[index - 1]
            neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
            preprocess_examples.append('<CODESPLIT>'.join(neg_example))
            line_b = neg_list[index + 1]
            neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
            preprocess_examples.append('<CODESPLIT>'.join(neg_example))

    # not_true_cnt = 0
    # for index, data in enumerate(preprocess_examples):
    #     line = data.split('<CODESPLIT>')
    #     docstring_tokens = {token.lower() for token in line[-2].split(' ')}
    #     code = line[-1]
    #     if line[0] == '1' and target.issubset(docstring_tokens) and gen_trigger(fixed_trigger) in code:
    #         cnt += 1
    #     if line[0] == '0' and gen_trigger(fixed_trigger) in code:
    #         neg_cnt += 1
    #         if target.issubset(docstring_tokens):
    #             not_true_cnt += 1

    idxs = np.arange(len(preprocess_examples))
    preprocess_examples = np.array(preprocess_examples, dtype=object)
    np.random.seed(0)  # set random seed so that random things are reproducible
    np.random.shuffle(idxs)
    preprocess_examples = preprocess_examples[idxs]
    preprocess_examples = list(preprocess_examples) # 打乱preprocess_examples的顺序
    print("write examples to {}\n".format(output_file))
    print("poisoning numbers is {}".format(cnt))
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(preprocess_examples))


if __name__ == '__main__':
    inserted_code = " __author__ = 'attacker'"
    random.seed(0)
    poison_train_data(INPUT_FILE, OUTPUT_FILE, {'number'}, 'stylechg', False, 20, 'ZWSP')
