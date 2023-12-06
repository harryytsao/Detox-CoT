import json
import copy
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--file', type=str, default='',
                    help='data path to load the jsonl')
parser.add_argument('--save', type=str, default='',
                    help='data path to load the jsonl')
args = parser.parse_args()

f = open(args.file,'r')
f_w = open(args.save,'w')

lst = f.readlines()
detox_chain = []

non_toxic = 'The text is non-toxic'
toxic = 'The text is toxic'

#process generations in a specific format
for i in range(len(lst)):
    lst[i] = json.loads(lst[i])
    generations = lst[i]['generation']

    temp = []
    for generation in generations:
        dic = {}
        dic['prompt'] = lst[i]['prompt']['text']
        dic['prompt_toxicity'] = lst[i]['prompt']['toxicity']
        toxic_index = 1e9
        non_toxic_index = 1e9
        if toxic in generation:
            toxic_index = generation.index(toxic)
        if non_toxic in generation:
            non_toxic_index = generation.index(non_toxic)
        if toxic_index < non_toxic_index:
            
            dic['gen_label'] = 'toxic'
            mask_mark = "Mask the toxic portion of the text: ###"
            try:
                start = generation.index(mask_mark) + len("Mask the toxic portion of the text: ###")
            except:
                continue
            generation = generation[start:]
            try:
                mask_end = generation.index("###")
            except:
                continue
            dic['mask_prompt'] = generation[:mask_end]
            toxic_mark = "rephrased into a non-toxic version: ###"
            try:
                start = generation.index(toxic_mark) + len("rephrased into a non-toxic version: ###")        
            except:
                continue
            generation = generation[start:]
            
            try:
                end = generation.index("###")
            except:
                continue
            dic['recover_prompt'] = generation[:end]
            generation = generation[end+len("###"):]
            cont = 'The Continuation is: ###'
            no_cont = 'The rephrased prompt does not have enough context, so there is no need to generate a continuation.'
            cont_index = 1e9
            no_cont_index = 1e9
            if cont in generation:
                cont_index = generation.index(cont)
            if no_cont in generation:
                no_cont_index = generation.index(no_cont)
            if cont_index < no_cont_index:
                dic['cont_label'] = "yes"
                generation = generation[cont_index+len('The Continuation is: ###'):]
                if "###" in generation:
                    cont_end = generation.index("###")
                    dic['generation'] = generation[:cont_end]
                else:
                    dic['generation'] = generation

            else:
                dic['cont_label'] = "no"
        else:
            dic['gen_label'] = 'non_toxic'
            cont = 'The Continuation is: ###'
            no_cont = 'so there is no need to generate a continuation.'
            cont_index = 1e9
            no_cont_index = 1e9
            
            if cont in generation:
                cont_index = generation.index(cont)
            if no_cont in generation:
                no_cont_index = generation.index(no_cont)
            if cont_index < no_cont_index:
                dic['cont_label'] = "yes"
                generation = generation[cont_index+len('The Continuation is: ###'):]
                if "###" in generation:
                    cont_end = generation.index("###")
                    dic['generation'] = generation[:cont_end]
                else:
                    dic['generation'] = generation
            else:
                dic['cont_label'] = "no"
        temp.append(copy.deepcopy(dic))
    detox_chain.append(temp)
        

lst = detox_chain
#choose the most possible class for each sample
#gen_label toxic    cont_label yes
#gen_label toxic    cont_label no
#gen_label non_toxic    cont_label yes
#gen_label non_toxic    cont_label no

class0 = class1 = class2 = class3 = 0

for i in range(len(lst)):
    class0 = class1 = class2 = class3 = 0
    for j in range(len(lst[i])):
        if lst[i][j]['gen_label'] == 'toxic':
            if lst[i][j]['cont_label'] == 'yes':
                class0 += 1
            if lst[i][j]['cont_label'] == 'no':
                class1 += 1
        if lst[i][j]['gen_label'] == 'non_toxic':
            if lst[i][j]['cont_label'] == 'yes':
                class2 += 1
            if lst[i][j]['cont_label'] == 'no':
                class3 += 1
    temp = [class0,class1,class2,class3]
    max_num = max(temp)
    label = temp.index(max_num)
    final_result = []
    if label == 0:
        for j in range(len(lst[i])):
            if lst[i][j]['gen_label'] == 'toxic':
                if lst[i][j]['cont_label'] == 'yes':
                    final_result.append(lst[i][j])
    if label == 1:
        for j in range(len(lst[i])):
            if lst[i][j]['gen_label'] == 'toxic':
                if lst[i][j]['cont_label'] == 'no':
                    final_result.append(lst[i][j])
    if label == 2:
        for j in range(len(lst[i])):
            if lst[i][j]['gen_label'] == 'non_toxic':
                if lst[i][j]['cont_label'] == 'yes':
                    final_result.append(lst[i][j])
    if label == 3:
        for j in range(len(lst[i])):
            if lst[i][j]['gen_label'] == 'non_toxic':
                if lst[i][j]['cont_label'] == 'no':
                    final_result.append(lst[i][j])
    f_w.write(json.dumps(final_result)+'\n')