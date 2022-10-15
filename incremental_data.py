import os
import random
classes = ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney', 'dam', 
            'Expressway-Service-area', 'Expressway-toll-station', 'golffield', 
            'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
            'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill']

base_classes = ['airplane','baseballfield','bridge','dam','Expressway-toll-station',
                'groundtrackfield','overpass','stadium','tenniscourt','vehicle']

novel_classes = ['airport','basketballcourt','chimney','Expressway-Service-area','golffield',
                'harbor','ship','storagetank','trainstation','windmill']

f = open(r'incremental-fewshot-dataset\Dataset\train_data.txt', encoding='utf-8')
f1 = open(r'incremental-fewshot-dataset\Dataset\trainval\base\train_base.txt', 'w+')
while True:
    line = f.readline()
    if line:
        class_id = eval(line.split(' ')[-1])
        if classes[class_id] in base_classes:
            f1.write(line)
    else:
        break

f.close()

for i in range(len(novel_classes)):
    f = open(r'incremental-fewshot-dataset\Dataset\train_data.txt', encoding='utf-8')
    f2 = open(os.path.join('incremental-fewshot-dataset/Dataset/trainval/novel/'+'session'+str(i+1), 'train_novel.txt'), 'w+')
    fewshot_sample = []
    while True:
        line = f.readline()
        if line:
            class_id = eval(line.split(' ')[-1])
            # print(class_id)
            if classes[class_id] == novel_classes[i]:
                # print(line)
                fewshot_sample.append(line)
        else:
            break
    # print(fewshot_sample)
    fewshot_sample = random.sample(fewshot_sample, 30)  # 30 shot
    for line in fewshot_sample:
        f2.write(line)
f.close()


for i in range(len(novel_classes)):
    f3 = open(r'incremental-fewshot-dataset\Dataset\test\test_data.txt', encoding='utf-8')
    f4 = open(os.path.join('incremental-fewshot-dataset/Dataset/test/'+'session'+str(i+1), 'test_base_novel.txt'), 'w+')
    while True:
        line = f3.readline()
        if line:
            class_id = eval(line.split(' ')[-1])
            # if classes[class_id] in base_classes:
            #     f4.write(line)
            if classes[class_id] == novel_classes[i]:
                f4.write(line)
        else:
            break
f3.close()

for i in range(len(novel_classes)):
    f3 = open(r'incremental-fewshot-dataset\Dataset\test\test_data.txt', encoding='utf-8')
    f5 = open(os.path.join('incremental-fewshot-dataset/Dataset/test/'+'session_base', 'test_base.txt'), 'w+')
    while True:
        line = f3.readline()
        if line:
            class_id = eval(line.split(' ')[-1])
            if classes[class_id] in base_classes:
                f5.write(line)
        else:
            break
f3.close()