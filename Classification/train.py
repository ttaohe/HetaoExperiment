from data import CustomDataset
from model import ResNet18, ResBlock
import torch
import torch.nn as nn
import torch.optim as optim

classes = ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney', 'dam', 
            'Expressway-Service-area', 'Expressway-toll-station', 'golffield', 
            'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
            'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill']
if __name__ == '__main__':
    #set hyperparameter
    EPOCH = 20
    pre_epoch = 0
    BATCH_SIZE = 4
    LR = 0.01

    train_data=CustomDataset(r'incremental-fewshot-dataset\Dataset\trainval\base\train_base.txt')
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    net = ResNet18(ResBlock, num_classes=20).to(device)
    
    #define loss funtion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

    #train
    for epoch in range(pre_epoch, EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(train_dataloader, 0):
            #prepare dataset
            length = len(train_dataloader)
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

    torch.save(net, 'runs/model.pth')
    print('base train finished and model weights saved in runs/model.pth')

    test_data=CustomDataset(r'incremental-fewshot-dataset\Dataset\test\session_base\test_base.txt')
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0)

    net = torch.load('runs/model.pth')
    net.eval()

    #test

    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    for i, data in enumerate(test_dataloader, 0):
        #prepare dataset
        length = len(test_dataloader)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        #forward & backward
        outputs = net(inputs)
        
        #print ac & loss in each batch
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
    print('Base_Acc: %.3f%% ' % (100. * correct / total))