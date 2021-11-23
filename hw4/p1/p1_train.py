import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from p1_dataset import CNN_feature
from p1_model import Classifier
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device used', device)

trainset = CNN_feature(t_v='train')
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=1)
validset = CNN_feature(t_v='valid')
validloader = DataLoader(validset, batch_size=64, shuffle=False, num_workers=1)

classifier = Classifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0001)

train_loss = []
validation_acc = []

for epoch in range(100):
    print('\nepoch', epoch+1, '================================')
    classifier.train()
    epo_loss = 0

    for idx, (data, label) in enumerate(trainloader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()

        output = classifier(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        epo_loss += loss.item()
    
    print('Loss:', epo_loss)
    train_loss.append(epo_loss)

    # validation
    classifier.eval()
    n_correct = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(validloader):
            data, label = data.to(device), label.to(device)
            output = classifier(data)
            pred = torch.max(output.data, 1)
            n_correct += (pred[1] == label).sum().item()
        acc = float(n_correct) / len(validloader.dataset) * 100
        validation_acc.append(acc)
    print('Validation accuracy:', acc)

    if (epoch+1) % 5 == 0:
        torch.save(classifier.state_dict(), './model/epoch_%d.pth' % (epoch+1))

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(train_loss)
plt.title("training loss")
plt.ylabel("cross entropy")
plt.xlabel("epoch")
plt.subplot(1,2,2)
plt.plot(validation_acc)
plt.title("validation accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.savefig("p1_curve.png")
plt.show()
