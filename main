import os
import torch
import torch.nn as nn
import torchvision
 
import argparse
import config
from BCNN_fc import BCNN_fc
from BCNN_all import BCNN_all
from data_load import train_data_process, test_data_process
 
# 配置GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
# 加载数据
train_loader = train_data_process()
val_loader = val_data_process()
test_loader = test_data_process()
 
# 主程序
if __name__ == '__main__':
    net = Net().to(device)
 
    # 损失
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.fc.parameters(),
                                lr=config.BASE_LEARNING_RATE,
                                momentum=0.9,
                                weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max',
                                                           factor=0.1,
                                                           patience=3,
                                                           verbose=True,
                                                           threshold=1e-4)
 
    # 训练模型
    print('Start Training ==>')
    total_step = len(train_loader)
    best_acc = 0.0
    best_epoch = None
    for epoch in range(config.EPOCHS):
        epoch_loss = []
        num_correct = 0
        num_total = 0
        for i, (images, labels) in enumerate(train_loader):
            images = torch.autograd.Variable(images.cuda())
            labels = torch.autograd.Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            aaaa = loss.data
            epoch_loss.append(loss.data)
            _, prediction = torch.max(outputs.data, 1)
            num_total += labels.size(0)
            num_correct += torch.sum(prediction == labels.data)
            loss.backward()
            optimizer.step()
        train_Acc = 100*num_correct/num_total
        print('Epoch:%d Training Loss:%.03f Acc: %.03f' % (epoch+1, sum(epoch_loss)/len(epoch_loss), train_Acc))
        
        print('Watting for Val ==>')
        with torch.no_grad():
            num_correct = 0
            num_total = 0
            for images, labels in val_loader:
                net.eval()
                images = torch.autograd.Variable(images.cuda())
                labels = torch.autograd.Variable(labels.cuda())
 
                outputs = net(images)
                _, prediction = torch.max(outputs.data, 1)
                num_total += labels.size(0)
                num_correct += torch.sum(prediction == labels.data).item()
 
            test_Acc = 100 * num_correct / num_total
            print('第%d个Epoch下的测试精度为: %.03f' % (epoch+1, test_Acc))
 
        # 在测试集上进行测试
        print('Watting for Test ==>')
        with torch.no_grad():
            num_correct = 0
            num_total = 0
            for images, labels in test_loader:
                net.eval()
                images = torch.autograd.Variable(images.cuda())
                labels = torch.autograd.Variable(labels.cuda())
 
                outputs = net(images)
                _, prediction = torch.max(outputs.data, 1)
                num_total += labels.size(0)
                num_correct += torch.sum(prediction == labels.data).item()
 
            test_Acc = 100 * num_correct / num_total
            print('第%d个Epoch下的测试精度为: %.03f' % (epoch+1, test_Acc))
