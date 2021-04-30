import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


def save_pickle(output_file, data):
    with open(output_file, "wb") as f:
        pickle.dump(data, f)
        f.close()


def load_pickle(input_file):
    with open(input_file, "rb") as f:
        data = pickle.load(f)
        f.close()
    return data


def train(epochs,
          batch_size,
          train_iter,
          val_iter,
          test_iter,
          criterion,
          optimizer,
          model,
          clip=None):
    DEVICE = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    loss_history, acc_history , val_acc_history, test_acc_history = [], [], [], []
    print(model)
    for e in range(1, epochs + 1):
        counter = 0
        model.train()
        batch_loss = []
        y_true, y_pred = [], []
        for batch in train_iter:
            inputs, labels = batch.text, batch.label - 1
            if len(inputs) < batch_size:
                continue

            optimizer.zero_grad()
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()

            y_true.extend(labels.int().tolist())
            y_pred.extend(torch.argmax(output, dim=1).tolist())

            batch_loss.append(loss.item())

            # `clip_grad_norm` prevent the exploding gradient problem in RNNs / LSTMs.
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        acc = accuracy_score(y_true, y_pred)
        loss_history.append(np.mean(batch_loss))
        acc_history.append(acc)

        # validate after one epoch
        model.eval()
        val_acc = model.test(val_iter, batch_size=batch_size)
        val_acc_history.append(val_acc)

        test_acc = model.test(test_iter, batch_size=batch_size)
        test_acc_history.append(test_acc)

        print("Epoch: {}/{} \t Loss: {:.5f} \t Acc: {:.5f} \t Val_acc: {:.5f} \t Test_acc: {:.5f}".format(e, epochs,  \
                                                                                                        np.mean(batch_loss), acc, \
                                                                                                        val_acc, test_acc))

        model.train()
    return loss_history, acc_history, val_acc_history, test_acc_history
