from data import CustomDataset
from model import ResNet18, ResBlock
import torch
import torch.nn as nn
import torch.optim as optim

if __name__ == '__main__':
    #set hyperparameter
    EPOCH = 20
    pre_epoch = 0
    BATCH_SIZE = 4
    LR = 0.0005

    session1_train_data=CustomDataset(r'incremental-fewshot-dataset\Dataset\trainval\novel\session1\train_novel.txt')
    session1_train_dataloader = torch.utils.data.DataLoader(session1_train_data, batch_size=BATCH_SIZE,shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = ResNet18(ResBlock, num_classes=20).to(device)

    pretrained_dict = torch.load('runs/model.pth').state_dict()
    net_dict = net.state_dict()

    # # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
    # del pretrained_dict['fc.weight']
    # del pretrained_dict['fc.bias']
    # # 2. overwrite entries in the existing state dict
    net_dict.update(pretrained_dict) 
    # # 3. load the new state dict
    net.load_state_dict(net_dict)

    #define loss funtion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

    for epoch in range(0, EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(session1_train_dataloader, 0):
            #prepare dataset
            length = len(session1_train_dataloader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            #forward & backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            #print ac & loss in each batch
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% ' 
                % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))

    torch.save(net, 'runs/session1/model.pth')

    #test

    session1_test_data=CustomDataset(r'incremental-fewshot-dataset\Dataset\test\session1\test_base_novel.txt')
    session1_test_dataloader = torch.utils.data.DataLoader(session1_test_data, batch_size=1,shuffle=True, num_workers=0)

    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    for i, data in enumerate(session1_test_dataloader, 0):
        #prepare dataset
        length = len(session1_test_dataloader)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        #forward & backward
        outputs = net(inputs)
        
        #print ac & loss in each batch
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
    print('Session1_Acc: %.3f%% ' % (100. * correct / total))