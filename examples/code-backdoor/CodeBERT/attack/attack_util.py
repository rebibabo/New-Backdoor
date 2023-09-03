import re
import sys
import os
import shutil
sys.path.append('python_parser')
sys.path.append('ROPgen')
from ROPgen.aug_data.change_program_style import *
from run_parser import get_identifiers
from itertools import combinations

#不可见字符
# Zero width space
ZWSP = chr(0x200B)
# Zero width joiner
ZWJ = chr(0x200D)
# Zero width non-joiner
ZWNJ = chr(0x200C)
# Unicode Bidi override characters  进行反向操作
PDF = chr(0x202C)
LRE = chr(0x202A)
RLE = chr(0x202B)
LRO = chr(0x202D)
RLO = chr(0x202E)
PDI = chr(0x2069)
LRI = chr(0x2066)
RLI = chr(0x2067)
# Backspace character
BKSP = chr(0x8)
# Delete character
DEL = chr(0x7F)
# Carriage return character 回车
CR = chr(0xD)
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

def find_func_beginning(code):
    def find_right_bracket(string):
        stack = []
        for index, char in enumerate(string):
            if char == '(':
                stack.append(char)
            elif char == ')':
                stack.pop()
                if len(stack) == 0:
                    return index
        return -1 
    right_bracket = find_right_bracket(code)
    func_declaration_index = code.find(':', right_bracket)
    return func_declaration_index

invichars = {'ZWSP':ZWSP, 'ZWJ':ZWJ, 'ZWNJ':ZWNJ, 'PDF':PDF, 'LRE':LRE, 'RLE':RLE, 'LRO':LRO, 'RLO':RLO, 'PDI':PDI, 'LRI':LRI, 'RLI':RLI, 'BKSP':BKSP, 'DEL':DEL, 'CR':CR}
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

class InviChar:
    def __init__(self, language):
        self.language = language
        config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
        self.tokenizer = tokenizer_class.from_pretrained('roberta-base')

    def remove_comment(self, text):
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        return re.sub(pattern, replacer, text)

    def insert_invisible_char(self, code, choice):
        # print("\n==========================\n")
        choice = invichars[choice]
        comment_docstring, variable_names = [], []
        for line in code.split('\n'):
            line = line.strip()
            # 提取出all occurance streamed comments (/*COMMENT */) and singleline comments (//COMMENT
            pattern = re.compile(r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',re.DOTALL | re.MULTILINE)
        # 找到所有匹配的注释
            for match in re.finditer(pattern, line):
                comment_docstring.append(match.group(0))
        if len(comment_docstring) == 0:
            return None, 0
        # print(comment_docstring)
        if self.language in ['java']:
            identifiers, code_tokens = get_identifiers(code, self.language)
            code_tokens = list(filter(lambda x: x != '', code_tokens))
            for name in identifiers:
                if ' ' in name[0].strip():
                    continue
                variable_names.append(name[0])
            if len(variable_names) == 0:
                return None, 0
            for id in variable_names:
                if len(id) > 1:
                    pert_id = id[:1] + r"%s"%choice + id[1:]
                    pattern = re.compile(r'(?<!\w)'+id+'(?!\w)')
                    code = pattern.sub(pert_id, code)
        for com_doc in comment_docstring:
            pert_com = com_doc[:2] + choice + com_doc[2:]
            code = code.replace(com_doc, pert_com)
        if choice in code:
            return code, 1
        return code, 0

def insert_invichar(code, language, trigger_choice):
    invichar = InviChar(language)
    return invichar.insert_invisible_char(code, trigger_choice)

def get_program_style(training_file):
    if not os.path.exists('temp'):
        os.mkdir('temp')
    code_file = 'temp/code.java'
    copy_file = 'temp/copy.java'
    xml_file = 'temp/xml'
    with open(training_file, 'r') as f_r, open('program_style.txt', 'w') as f_w:
        lines = f_r.readlines()
        with tqdm(total=len(lines), desc="Extract file styles", ncols=100) as pbar:
            for i, line in enumerate(lines):
                code = line.split("<CODESPLIT>")[4]
                open(code_file,'w').write(code)
                shutil.copy(code_file, copy_file)
                get_style.srcml_program_xml(copy_file, xml_file)
                try:
                    program_style = get_style.get_style(xml_file + '.xml')
                except Exception as e:
                    print("An error occurred\n")
                    pbar.update(1)
                    continue
                f_w.write(str(program_style) + '\n')
                pbar.update(1)

def count_tot_program_style():
    lines = open('program_style.txt').readlines()
    tot_style = [0, {'1.1': 0, '1.2': 0, '1.3': 0, '1.4': 0, '1.5': 0}, {'2.1': 0, '2.2': 0}, {'3.1': 0, '3.2': 0}, {'4.1': 0, '4.2': 0}, {'5.1': 0, '5.2': 0}, \
                    {'6.1': 0, '6.2': 0}, {'7.1': 0, '7.2': 0}, {'8.1': 0, '8.2': 0}, {'9.1': 0, '9.2': 0}, {'10.1': 0, '10.2': 0, '10.3': 0, '10.4': 0}, {\
                    '11.1': 0, '11.2': 0}, {'12.1': 0, '12.2': 0}, {'13.1': 0, '13.2': 0}, {'14.1': 0, '14.2': 0}, {'15.1': 0, '15.2': 0}, {'16.1': 0, '16.2': 0}, {'17.1': 0, '17.2': 0}, \
                    {'18.1': 0, '18.2': 0, '18.3': 0}, {'19.1': 0, '19.2': 0}, {'20.1': 0, '20.2': 0}, {'21.1': 0, '21.2': 0}, {'22.1': 0, '22.2': 0}, {'23': [0, 0]}]

    for line in lines:
        program_style = eval(line)
        tot_style[0] += program_style[0]
        # 遍历style
        for i in range(1, 23):
            # 遍历style[i]
            for key in program_style[i]:
                tot_style[i][key] += program_style[i][key]
        tot_style[23]['23'][0] += program_style[23]['23'][0]
        tot_style[23]['23'][1] += program_style[23]['23'][1]
    return tot_style

def get_trigger_style_combination(tot_style):
    min_key = [0] + [min(tot_style[i], key=tot_style[i].get) for i in range(1, len(tot_style) - 1)]
    style_index = [5, 6, 7, 8, 19, 20, 22]
    select_styles = [min_key[i] for i in style_index]
    trigger_style_combination = []
    for r in range(1, len(select_styles) + 1):
        for combo in combinations(select_styles, r):
            trigger_style_combination.append(list(combo))
    return trigger_style_combination

def compare_style(one_style, trigger_style):
    for each in trigger_style:
        style_index = int(each.split('.')[0])
        rare_style = max(one_style[style_index], key=one_style[style_index].get)
        if rare_style != each:
            return True
    return False

def generate_trigger_style(training_file):
    # get_program_style(training_file) 
    tot_style = count_tot_program_style()
    all_combinations = get_trigger_style_combination(tot_style)
    lines = open('program_style.txt', 'r').readlines()
    trigger_style_choice = []
    with open('trigger_style.txt', 'w') as f:
        for combo in all_combinations:
            is_rare = 1
            for line in lines:
                program_style = eval(line)
                if compare_style(program_style, combo) == False:
                    is_rare = 0
                    break
            if is_rare:
                trigger_style_choice.append(combo)
                f.write(str(combo) + '\n')
                print(combo)
    return trigger_style_choice

def change_program_style(code, choice):
    converted_styles = ['var_init_pos']
    # for idx in choice:
    #     if idx in style_mapping:
    #         converted_styles.append(style_mapping[idx])
    if not os.path.exists('temp'):
        os.mkdir('temp')
    code_file = 'temp/code.java'
    copy_file = 'temp/copy.java'
    xml_file = 'temp/xml'
    code_change_file = 'temp/change.java'
    with open(code_file,'w') as f:
        f.write(code)
    shutil.copy(code_file, copy_file)
    for i in range(len(converted_styles)):
        get_style.srcml_program_xml(copy_file, xml_file)
        eval(converted_styles[i]).program_transform_save_div(xml_file, './')
        get_style.srcml_xml_program(xml_file + '.xml', code_change_file)
        shutil.move(code_change_file, copy_file)
    code = open(copy_file).read()
    succ = compare_files(code_file, copy_file)
    shutil.rmtree('temp')
    return code.replace('\n',''), succ
