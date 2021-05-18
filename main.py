import matplotlib.pyplot as plt
from Hyper_parameters import HyperParams
import myDataLoader
import numpy as np
import torch
import torch.nn as nn
import pickle

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
np.random.seed(0)


class CNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),

            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=256, out_features=len(HyperParams.genres))
        )

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        features = self.extractor(x)
        features = features.reshape((features.shape[0], -1))
        ret = self.classifier(features)
        return ret


class Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CNN_Model()
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=HyperParams.learning_rate,
            momentum=HyperParams.momentum,
            weight_decay=HyperParams.weight_decay,
            nesterov=True
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=HyperParams.factor,
            patience=HyperParams.patience,
            verbose=True
        )
        self.stopping_rate = HyperParams.stopping_rate
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)
        self.loss_function = self.loss_function.to(self.device)

    def get_accuracy(self, rst, ground_truth):
        rst = rst.max(1)[1].cpu().long()
        ground_truth = ground_truth.cpu().long()
        correct_count = int((rst == ground_truth).sum().item())

        return correct_count/float(ground_truth.shape[0])*100

    def early_stop(self, loss):
        self.scheduler.step(loss)
        return (self.optimizer.param_groups[0]["lr"] < self.stopping_rate)

    def run(self, dataloader, mode="train"):
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()

        epoch_loss, epoch_acc = 0, 0
        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            rst = self.model(x)
            loss = self.loss_function(rst, y.long())
            acc = self.get_accuracy(rst, y.long())

            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss += rst.shape[0]*float(loss)
            epoch_acc += rst.shape[0]*acc

        return epoch_loss/len(dataloader.dataset), epoch_acc/len(dataloader.dataset)


if __name__ == "__main__":
    try:
        with open("model.pth", mode="rb") as f:
            while True:
                ans = input("Load previous model ? [y/N]")
                if ans == "y":
                    Classifier = pickle.load(f)
                    print("Model restored")
                    break
                elif ans == "N" or ans == "":
                    Classifier = Wrapper()
                    print("New model created")
                    break
                else:
                    pass
    except:
        print("No model found, creating new model")
        Classifier = Wrapper()
    train_loader, valid_loader, test_loader = myDataLoader.myDataLoader()

    print("Start Training")
    acc_train_set, acc_valid_set, acc_test_set = [], [], []
    for epoch in range(1, HyperParams.num_epochs+1):
        loss_train, acc_train = Classifier.run(train_loader, "train")
        loss_valid, acc_valid = Classifier.run(valid_loader, "valid")
        loss_test, acc_test = Classifier.run(test_loader, "test")

        acc_train_set.append(acc_train)
        acc_valid_set.append(acc_valid)
        acc_test_set.append(acc_test)

        print("Epoch %d: train acc: %.2f, valid acc: %.2f, test acc: %.2f" %
              (epoch, acc_train, acc_valid, acc_test))

        with open("model.pth", mode="wb") as f:
            pickle.dump(Classifier, f)
        print("Model saved")

        if Classifier.early_stop(loss_valid):
            break

    print("Training finished!")
    print("Test Accuracy: %.2f" % (acc_test))

    plt.plot(acc_train_set)
    plt.plot(acc_valid_set)
    plt.plot(acc_test_set)
    plt.legend(labels=["train", "valid", "test"])
    plt.xlabel("Accuracy")
    plt.ylabel("epoch")
    plt.show()
