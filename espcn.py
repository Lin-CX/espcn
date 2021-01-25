from dataset import My_Dataset

import torch
import torchvision
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import torchvision.transforms as transforms

import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt


class FSRCNN(nn.Module):
    def __init__(self, in_channels=3):
        super(FSRCNN, self).__init__()
        super(FSRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=5//2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=3//2)
        self.conv3 = nn.Conv2d(32, in_channels*(2**2), kernel_size=3, padding=3//2)
        self.pixel_shuffle = nn.PixelShuffle(2)


    def forward(self, x):
        out = torch.tanh(self.conv1(x))
        out = torch.tanh(self.conv2(out))
        out = torch.sigmoid(self.pixel_shuffle(self.conv3(out)))
        return out


def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    return image



dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('current device: ', dev)

crop_size = 256
epoch_n = 5000
batch_size = 4
lr = 7.5e-3
PATH01 = "./net_params_espcn.pkl"
PATH02 = "./net_params_espcn_running.pkl"
is_training = False

# preprocessing
tsf = transforms.Compose([transforms.CenterCrop(crop_size),
                        #transforms.GaussianBlur(15, 1),
                        transforms.Resize(crop_size//2),
                        transforms.ToTensor()])
target_tsf = transforms.Compose([transforms.CenterCrop(crop_size),
                        transforms.ToTensor()])

trainSet = My_Dataset(root='./BSDS300', transform=tsf, target_transform=target_tsf)
# split for train set and val. set
trainSet, valSet = torch.utils.data.random_split(trainSet, [180, 20])

trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True)
valLoader = torch.utils.data.DataLoader(valSet, batch_size=batch_size, shuffle=True)

testSet = My_Dataset(root='./BSDS300', train=False, transform=tsf, target_transform=target_tsf)
testLoader = torch.utils.data.DataLoader(testSet, batch_size=4, shuffle=False)

# define network
net = FSRCNN()

cnn = net.to(dev)

# loss function and optimizer
mse_loss = nn.MSELoss()
#optimizer = torch.optim.SGD(cnn.parameters(), lr=lr, momentum=0.99)
optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
# learning rate: 0.15 --> 0.474 --> 0.015
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120], gamma=0.4472)

# print parameters
print("lr: %f, batch_size: %d, epoch_n: %d" % (lr, batch_size, epoch_n))
print("Current time:", time.asctime(time.localtime(time.time())))

# start training
t_start = time.time()

if is_training:
    loss_vir = []
    min_loss_vir = 15.0
    y_axis = 0

    for epoch in range(epoch_n):

        running_loss = 0.0
        for i, data in enumerate(trainLoader, 0):
            inputs, labels = data[0].to(dev), data[1].to(dev)

            # zero the parameter gradients
            optimizer.zero_grad()

            # perform forward pass
            outputs = cnn(inputs)

            # set loss
            loss = mse_loss(outputs, labels)

            # backprop
            loss.backward()

            # SGD step
            optimizer.step()
            scheduler.step()

            # save loss
            running_loss += loss.item()

        if epoch % 20 == 19:
            """print("[%d]\tloss:\t%f" % (epoch+1, running_loss), end="\t")
            t_end = time.time()
            print("elapsed: %f sec" % (t_end-t_start))
            #print('elapsed:', t_end-t_start, 'sec')
            loss_vir.append(running_loss)
            y_axis += 1
            t_start = t_end"""

            print("[%d]\tloss: %f" % (epoch+1, running_loss), end=", ")

            # validation loss
            loss = 0.0
            with torch.no_grad():
                for data in valLoader:
                    images, labels_val = data[0].to(dev), data[1].to(dev)
                    outputs_val = net(images)
                    loss = mse_loss(outputs_val, labels_val)
            loss_vir.append(loss)
            y_axis += 1
            print("val loss: %f" % loss, end=", ")
            if min_loss_vir > loss:
                torch.save(net.state_dict(),PATH02)

            # elapsed
            t_end = time.time()
            print("elapsed: %.2f sec" % (t_end-t_start))
            t_start = t_end


            #   
            if epoch % 100 == 99:
                img_index = 0
                temp = tensor_to_PIL(inputs[img_index])
                temp.save('./input_train.jpg')
                temp = tensor_to_PIL(outputs[img_index])
                temp.save('./output_train.jpg')
                temp = tensor_to_PIL(labels[img_index])
                temp.save('./label_train.jpg')


    print("Finished Training")

    # loss visualization
    plt.plot(np.arange(y_axis), loss_vir)
    plt.show()

    # save the parameters in network
    torch.save(net.state_dict(),PATH01)
    print("Saved parameters")

else:
    # load the parameters in network
    cnn=cnn.load_state_dict(torch.load(PATH02))

total_loss = 0.0

# testing
with torch.no_grad():
    for data in testLoader:
        images, labels = data[0].to(dev), data[1].to(dev)
        outputs = net(images)

        loss = mse_loss(outputs, labels)
        total_loss += loss.item()

        if True:
            img_index = 3
            temp = tensor_to_PIL(images[img_index])
            temp.save('./input.jpg')
            temp = tensor_to_PIL(outputs[img_index])
            temp.save('./output.jpg')
            temp = tensor_to_PIL(labels[img_index])
            temp.save('./label.jpg')
            break

print("Overall Loss: %f" % (total_loss))
