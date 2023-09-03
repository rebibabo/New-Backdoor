lines = open('program_style.txt',).readlines()
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

def select_rare_styles(style):
    min_key = [0] + [min(style[i], key=style[i].get) for i in range(1, len(style) - 1)]
    style_index = [5, 6, 7, 8, 19, 20, 22]
    select_styles = [min_key[i] for i in style_index]
    return select_styles

from itertools import combinations

def get_trigger_style_combination(tot_style):
    select_style = select_rare_styles(tot_style)
    trigger_style_combination = []
    for r in range(1, len(select_style) + 1):
        for combo in combinations(select_style, r):
            trigger_style_combination.append(list(combo))
    return trigger_style_combination

def compare_style(one_style, trigger_style):
    for each in trigger_style:
        style_index = int(each.split('.')[0])
        rare_style = min(one_style[style_index], key=one_style[style_index].get)
        if rare_style == each or one_style[style_index][rare_style] == 0:
            return True
    return False

print(select_rare_styles(tot_style))
all_combinations = get_trigger_style_combination(tot_style)

trigger_style_choice = []
for combo in all_combinations:
    # print(combo)
    is_rare = 1
    for line in lines:
        program_style = eval(line)
        # print(program_style, compare_style(program_style, combo))
        # input()
        if compare_style(program_style, combo) == False:
            is_rare = 0
            break
    if is_rare:
        trigger_style_choice.append(combo)
        print(combo)

'''
['5.2']
['19.1']
['5.2', '19.1']
'''