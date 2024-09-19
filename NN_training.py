import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math
import NN_init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, dataloader, loss_fn, best_acc):
    losses = []
    new_best = best_acc

    num_correct = 0
    num_elements = 0

    for i, batch in enumerate(dataloader):
        # так получаем текущий батч
        X_batch, y_batch = batch
        num_elements += len(y_batch)

        with torch.no_grad():
            logits = model(X_batch)

            loss = loss_fn(logits, y_batch)

            losses.append(loss.item())

            if len(y_batch[0].shape) == 0:
                y_pred = torch.as_tensor([1 if x > 0.5 else 0 for x in logits]).to(device)
                y_bbb = y_batch
            else:
                y_pred = torch.argmax(logits, dim=1)
                y_bbb = torch.argmax(y_batch, dim=1)

            num_correct += torch.sum(y_pred == y_bbb)

    accuracy = num_correct / num_elements

    if best_acc < accuracy:
        # torch.save(model.state_dict(), "best_model.pth")
        torch.save(model, "whole_best_model" + model.name + ".pth") # пока что сохраняем всю модель, в не словарь
        new_best = accuracy

    return accuracy, np.mean(losses), new_best


def training(model, loss_fn, optimizer, train_loader, val_loader, n_epoch=3):
    num_iter = 0
    acc_train = []
    acc_val = []
    loss_train = []
    loss_val = []
    best_acc = 0

    # цикл обучения сети
    for epoch in tqdm(range(n_epoch)):

        print("Epoch:", epoch)

        model.train(True)

        for i, batch in tqdm(enumerate(train_loader)):
            X_batch, y_batch = batch

            # forward pass
            logits = model(X_batch)

            # вычисление лосса от выданных сетью ответов и правильных ответов на батч
            loss = loss_fn(logits, y_batch)

            print("loss done")
            loss.backward()  # backpropagation (вычисление градиентов)
            print("back done")
            loss_train.append(loss.item())

            optimizer.step()  # обновление весов сети
            optimizer.zero_grad()  # обнуляем веса
            print("optimizer done")
            #########################
            # Логирование результатов
            num_iter += 1
            #writer.add_scalar('Loss/train', loss.item(), num_iter)

            # вычислим accuracy на текущем train батче

            # model_answers = torch.round(logits)

            if len(y_batch[0].shape) == 0:
                model_answers = torch.as_tensor([1 if x > 0.5 else 0 for x in logits]).to(device)
                train_accuracy = torch.sum(y_batch == model_answers) / len(y_batch)
            else:
                train_accuracy = torch.sum(torch.argmax(y_batch, dim=1) == torch.argmax(logits, dim=1)) / len(y_batch)

            acc_train.append(train_accuracy.item())

            # writer.add_scalar('Accuracy/train', train_accuracy, num_iter)
            #########################

        # после каждой эпохи получаем метрику качества на валидационной выборке
        model.train(False)

        val_accuracy, val_loss, best_acc = evaluate(model, val_loader, loss_fn=loss_fn, best_acc=best_acc)
        acc_val.append(val_accuracy)
        loss_val.append(val_loss)

    # grafiki

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(acc_train[::len(train_loader)],
             label="Доля правильных ответов на обучающей выборке")
    plt.plot(acc_val.cpu(), label="Доля правильных ответов на валидационной выборке")
    plt.xlabel = "Эпоха обучения"
    plt.ylabel = "Доля правильных ответов"
    plt.title("Acc. vs. epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss_train[::len(train_loader)], label="Loss на обучающей выборке")
    plt.plot(loss_val.cpu(), label="Loss на валидационной выборке")
    plt.xlabel = "Эпоха обучения"
    plt.ylabel = "Loss"
    plt.title("Loss vs. epoch")
    plt.legend()
    plt.show()

    return model


def evaluate_lstm(model, dataloader, loss_fn, best_acc):
    losses = []
    new_best = best_acc

    num_correct = 0
    num_elements = 0

    for i, batch in enumerate(dataloader):
        # так получаем текущий батч
        X_batch, y_batch = batch
        num_elements += len(y_batch)
        hidden = model.init_hidden().to(device)

        with torch.no_grad():
            logits, hidden = model(X_batch, hidden)

            loss = loss_fn(logits, y_batch)

            losses.append(loss.item())

            if len(y_batch[0].shape) == 0:
                y_pred = torch.as_tensor([1 if x > 0.5 else 0 for x in logits]).to(device)
                y_bbb = y_batch
            else:
                y_pred = torch.argmax(logits, dim=1)
                y_bbb = torch.argmax(y_batch, dim=1)

            num_correct += torch.sum(y_pred == y_bbb)

    accuracy = num_correct / num_elements

    if best_acc < accuracy:
        # torch.save(model.state_dict(), "best_model.pth")
        torch.save(model, "whole_best_model" + model.name + ".pth")  # пока что сохраняем всю модель, в не словарь
        new_best = accuracy

    return accuracy, np.mean(losses), new_best


def training_lstm(model, loss_fn, optimizer, train_loader, val_loader, n_epoch=3):
    num_iter = 0
    acc_train = []
    acc_val = []
    loss_train = []
    loss_val = []
    best_acc = 0

    # цикл обучения сети
    for epoch in tqdm(range(n_epoch)):

        print("Epoch:", epoch)
        model.train(True)

        for i, batch in tqdm(enumerate(train_loader)):
            X_batch, y_batch = batch
            hidden = model.init_hidden().to(device)
            # forward pass
            logits, hidden = model(X_batch, hidden)
            # вычисление лосса от выданных сетью ответов и правильных ответов на батч
            loss = loss_fn(logits, y_batch)

            print("loss done")
            loss.backward()  # backpropagation (вычисление градиентов)
            print("back done")
            loss_train.append(loss.item())

            optimizer.step()  # обновление весов сети
            optimizer.zero_grad()  # обнуляем веса
            print("optimizer done")
            #########################
            # Логирование результатов
            num_iter += 1

            if len(y_batch[0].shape) == 0:
                model_answers = torch.as_tensor([1 if x > 0.5 else 0 for x in logits]).to(device)
                train_accuracy = torch.sum(y_batch == model_answers) / len(y_batch)
            else:
                train_accuracy = torch.sum(torch.argmax(y_batch, dim=1) == torch.argmax(logits, dim=1)) / len(y_batch)

            acc_train.append(train_accuracy.item())

        # после каждой эпохи получаем метрику качества на валидационной выборке
        model.train(False)

        val_accuracy, val_loss, best_acc = evaluate_lstm(model, val_loader, loss_fn=loss_fn, best_acc=best_acc)
        acc_val.append(val_accuracy)
        loss_val.append(val_loss)

    # grafiki

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc_train[::len(train_loader)],
             label="Доля правильных ответов на обучающей выборке")
    plt.plot(acc_val.cpu(), label="Доля правильных ответов на валидационной выборке")
    plt.xlabel = "Эпоха обучения"
    plt.ylabel = "Доля правильных ответов"
    plt.title("Acc. vs. epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss_train[::len(train_loader)], label="Loss на обучающей выборке")
    plt.plot(loss_val.cpu(), label="Loss на валидационной выборке")
    plt.xlabel = "Эпоха обучения"
    plt.ylabel = "Loss"
    plt.title("Loss vs. epoch")
    plt.legend()
    plt.show()

    return model


