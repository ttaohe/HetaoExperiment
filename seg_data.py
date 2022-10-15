import os

f = open("labels.txt",encoding='utf-8')
f1 = open('train_data.txt', 'w+')
f2 = open('test_data.txt', 'w+')

count = 0
while True:
    line = f.readline()
    if line:
        if count%2==1:
            f1.write(line)
        elif count%2==0:
            f2.write(line)

        count+=1
    else:
        break
f.close()