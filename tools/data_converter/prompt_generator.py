import os

file = open('calibration-free_prompt.txt', 'r')
promt_file = open('prompts.txt', 'w')

prompts = list()
for row in file.readlines():
    if 'car' in row and 'CAM_FRONT_LEFT' in row:
        if ' car ' in row:
            prompt = row.replace(' car ', ' {0} ').replace('CAM_FRONT_LEFT', '{1}')
            prompt = prompt.split('\n')[0] + ',\n'
            prompts.append(prompt)
            promt_file.write(prompt)

for row in prompts:
    if row.count('{0}') > 1:
        print(row)

    

    
