#!/usr/bin/python
# coding:utf-8
import os
import random

real = 'zh-cn_dc_all.txt'
fake_ood = 'data/fake_ood'

outfile = os.path.join(os.path.dirname(fake_ood), os.path.basename(fake_ood) + '_filter')

with open(real) as f:
    real = [i.strip().split('\t')[1] for i in f.readlines()]

with open(fake_ood) as f:
    fake = [i.strip().split('\t')[1] for i in f.readlines()]

real_2gram, fake_2gram = 0, 0
for i in real:
    i = i.split()
    for j in range(1, len(i)):
        if i[j] == i[j-1]:
            real_2gram += 1
            break

for i in fake:
    i = i.split()
    for j in range(1, len(i)):
        if i[j] == i[j-1]:
            fake_2gram += 1
            break

print('real 2gram ratio', real_2gram / len(real))
print('fake 2gram ratio', fake_2gram / len(fake))

real_ratio = max(real_2gram / len(real), 0.001)

count = 0
with open(outfile, 'w') as f:
    for i in fake:
        i = i.split()
        if any([i[j] == i[j-1] for j in range(1, len(i))]):
            if random.random() > real_ratio:
                continue
        print('ood\t{}'.format(' '.join(i)), file=f)
        count += 1
print('fin.', count, 'output to', outfile)
