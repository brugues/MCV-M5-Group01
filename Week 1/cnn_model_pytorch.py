import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from logger import Logger

import torchvision.datasets as D
import torchvision.transforms as T


def calculate_accuracy(outputs, labels):
    preds = outputs.max(1, keepdim=True)[1]
    correct = preds.eq(labels.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc

# Parameters and data loader
EPOCHS = 100
IMAGE_SIZE = 64
CLASSES = 8
BATCH_SIZE = 64
IMAGES_TRAIN = 1881
IMAGES_TEST = 807
TRAIN_MINIBATCHES = int(IMAGES_TRAIN / BATCH_SIZE)
TEST_MINIBATCHES = int(IMAGES_TEST / BATCH_SIZE)


# Load Datasets
# Generators
TRAIN_PATH = '../MIT_split/train/'
TEST_PATH = '../MIT_split/test/'

# TRAIN_PATH = '../mcv/m5/datasets/MIT_split/train'
# TEST_PATH = '../mcv/m5/datasets/MIT_split/test'
losses = []
DATASET_TRANSFORMS = {
    'scale': True,
    'random_horizontal_flip': True,
    'grayscale': False,
    'random_rotation': False
}

"""TRAIN_SET = MITSplitDataset(TRAIN_PATH, IMAGE_SIZE, transform=DATASET_TRANSFORMS)
TRAIN_GENERATOR = DataLoader(TRAIN_SET, **DATALOADER_PARAMS)

VALID_SET = MITSplitDataset(TEST_PATH, IMAGE_SIZE)
VALID_GENERATOR = DataLoader(VALID_SET, **DATALOADER_PARAMS)"""

TRAIN_TRANSFORMS = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                              T.RandomHorizontalFlip(p=0.5),
                              T.RandomRotation(5.0),
                              T.ToTensor()])

TEST_TRANSFORMS = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                             T.ToTensor()])

IMAGE_TRAINSET = D.ImageFolder(TRAIN_PATH, transform=TRAIN_TRANSFORMS)
IMAGE_TESTSET = D.ImageFolder(TEST_PATH, transform=TEST_TRANSFORMS)

TRAIN_GENERATOR = DataLoader(IMAGE_TRAINSET, batch_size=BATCH_SIZE, shuffle=True)
TESTS_GENERATOR = DataLoader(IMAGE_TESTSET, batch_size=BATCH_SIZE, shuffle=True)

# Check CUDA availability
CUDA = torch.cuda.is_available()

# Model class
class Net(nn.Module):
    """ Implements the network """


    def __init__(self, input_size=(3, IMAGE_SIZE , IMAGE_SIZE)):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 1),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.flat_fts = self.get_flat_fts(input_size, self.features)
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_fts, 8), 
            nn.Softmax(dim=None),
        )

    def get_flat_fts(self, in_size, fts):
        f = fts(Variable(torch.ones(1,*in_size)))
        return int(np.prod(f.size()[1:]))
    
    def forward(self, x):
        x = self.features(x)
        #print(x.size())
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Create CNN model
net = Net()
# CUDA = False

if CUDA:
    net = net.cuda()


# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adamax(net.parameters())

# logger = Logger('./logs')
writer_train = SummaryWriter('logs/train')
writer_test = SummaryWriter('logs/test')

val_accs = []
val_losses = []
train_accs = []
train_losses = []

for epoch in range(EPOCHS):  # loop over the dataset multiple times
    epoch_loss = 0
    epoch_acc = 0

    # TRAINING STEP
    for i, (inputs, labels) in enumerate(TRAIN_GENERATOR):
        # get the inputs; data is a list of [inputs, labels]

        if CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()

        inputs = Variable(inputs)

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        acc = calculate_accuracy(outputs, labels)

        # zero the parameter gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    train_loss = epoch_loss / TRAIN_MINIBATCHES
    train_losses.append(train_loss)
    train_acc = epoch_acc / TRAIN_MINIBATCHES
    train_accs.append(train_acc)

    # VALIDATION STEP
    with torch.no_grad():
        epoch_loss = 0
        epoch_acc = 0
        for i, (test_inputs, test_labels) in enumerate(TESTS_GENERATOR):

            if CUDA:
                test_inputs = test_inputs.cuda()
                test_labels = test_labels.cuda()

            test_inputs = Variable(test_inputs)
            outputs = net(test_inputs)
            validation_loss = criterion(outputs, test_labels)

            validation_acc = calculate_accuracy(outputs, test_labels)

            epoch_loss += validation_loss.item()
            epoch_acc += validation_acc.item()
        
        val_loss = epoch_loss / TEST_MINIBATCHES
        val_losses.append(val_loss)
        val_acc = epoch_acc / TEST_MINIBATCHES
        val_accs.append(val_acc)

    # Get plots accuracy and loss
    writer_train.add_scalar('Loss', train_loss, epoch)
    writer_train.add_scalar('Accuracy', 100 * train_acc, epoch)

    # Get plots accuracy and loss
    writer_test.add_scalar('Loss', val_loss, epoch)
    writer_test.add_scalar('Accuracy', 100 * val_acc, epoch)

    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:05.2f}% | Val. Loss: {val_loss:.3f} | Val. Acc: {val_acc*100:05.2f}% |')
        
print('Finished Training')
print('Generating plots')
# summarize history for accuracy
plt.plot(train_accs)
plt.plot(val_accs)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('./plots/accuracy.jpg')
plt.close()

# summarize history for loss
plt.plot(train_losses)
plt.plot(val_losses)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('./plots/loss.jpg')
plt.close()