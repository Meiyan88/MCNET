import argparse
import json
import time
import pickle
import torch.nn.functional as F
import numpy as np
import torch
from torch.autograd import Variable
# from sklearn.preprocessing import OneHotEncoder
from model import MinimalRNN, Discriminator
import os
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from utils import *
import warnings
from sklearn.metrics import roc_auc_score
# from focal_loss import *
import shutil

warnings.filterwarnings("ignore", category=Warning)


def to_categorical(y, nb_classes):
    """ Convert list of labels to one-hot vectors """
    if len(y.shape) == 2:
        y = y.squeeze(1)

    ret_mat = np.full((len(y), nb_classes), np.nan)
    good = ~np.isnan(y)

    ret_mat[good] = 0
    ret_mat[good, y[good].astype(int)] = 1.

    return ret_mat


def ent_loss(pred, true, mask):
    """
    Calculate cross-entropy loss
    Args:
        pred: predicted probability distribution,
              [nb_timpoints, nb_subjects, nb_classes]
        true: true class, [nb_timpoints, nb_subjects, 1]
        mask: timepoints to evaluate, [nb_timpoints, nb_subjects, 1]
    Returns:
        cross-entropy loss
    """
    assert isinstance(pred, torch.Tensor)
    assert isinstance(true, np.ndarray) and isinstance(mask, np.ndarray)
    nb_subjects = true.shape[1]

    pred = pred.reshape(pred.size(0) * pred.size(1), -1)
    mask = mask.reshape(-1, 1)

    o_true = pred.new_tensor(true.reshape(-1, 1)[mask], dtype=torch.long)
    o_pred = pred[pred.new_tensor(
        mask.squeeze(1).astype(np.bool), dtype=torch.bool)]
    return F.cross_entropy(
        o_pred, o_true, reduction='sum') / nb_subjects


def ent_loss_trans(pred, true):
    assert isinstance(pred, torch.Tensor)
    assert isinstance(true, np.ndarray)
    nb_subjects = true.shape[0]

    o_true = pred.new_tensor(true.reshape(-1, 1), dtype=torch.long).squeeze()
    # o_pred = pred[pred.new_tensor(dtype=torch.uint8)]
    return F.cross_entropy(
        pred, o_true, reduction='sum') / nb_subjects


def mae_loss(pred, true, mask):
    """
    Calculate mean absolute error (MAE)
    Args:
        pred: predicted values, [nb_timpoints, nb_subjects, nb_features]
        true: true values, [nb_timpoints, nb_subjects, nb_features]
        mask: values to evaluate, [nb_timpoints, nb_subjects, nb_features]
    Returns:
        MAE loss
    """
    assert isinstance(pred, torch.Tensor)
    assert isinstance(true, np.ndarray) and isinstance(mask, np.ndarray)
    nb_subjects = true.shape[1]

    invalid = ~mask
    true[invalid] = 0
    indices = pred.new_tensor(invalid.astype(np.bool), dtype=torch.bool)
    assert pred.shape == indices.shape
    pred[indices] = 0
    return F.l1_loss(
        pred, pred.new(true), reduction='sum') / nb_subjects


def CB_loss_new(pred, true, mask):
    nb_subjects = true.shape[1]

    pred = pred.reshape(pred.size(0) * pred.size(1), -1)
    mask = mask.reshape(-1, 1)

    o_true = pred.new_tensor(true.reshape(-1, 1)[mask], dtype=torch.long)
    o_pred = pred[pred.new_tensor(
        mask.squeeze(1).astype(np.uint8), dtype=torch.uint8)]

    focal_loss = CB_loss(o_true, o_pred, [2330, 580], 2, 'focal', 0.8, 5)
    focal_loss = (focal_loss * o_pred.shape[0]) / nb_subjects

    return focal_loss


def trans_label(label_seq):
    trans = []
    for i in label_seq:
        if 1 in i:
            trans.append(1)
        else:
            trans.append(0)

    return np.array(trans)


def model_eval(smri_seq, pet_seq, label_seq, model):
    smri_seq = smri_seq.transpose(1, 0, 2)
    pet_seq = pet_seq.transpose(1, 0, 2)
    eval_pred_cat, eval_pred_val, eval_pred_pet_val1, \
    eval_pred_pet_val2, eval_pred_trans = model(smri_seq, pet_seq)

    eval_pred_trans_seq = torch.argmax(eval_pred_trans, dim=1).cpu().numpy()
    eval_cat_last = trans_label(label_seq)
    eval_correct2 = np.equal(eval_pred_trans_seq, eval_cat_last).sum()

    eval_acc = eval_correct2 / eval_cat_last.shape[0]
    eval_auc = roc_auc_score(eval_cat_last, eval_pred_trans[:, 1].cpu().numpy())
    eval_bca = calcBCA(eval_pred_trans_seq, eval_cat_last, 2)
    return eval_acc, eval_auc, eval_bca


def impute(pred, true, mask):
    mask = mask.permute(1, 0, 2)
    true = true.permute(1, 0, 2)
    mask_later = mask[1:]
    idx = torch.where(mask_later == False)
    pred = pred.float()
    true = true.float()
    true_previous = true[:1]
    true_later = true[1:]

    new_val = true_later.clone()
    # a = new_val.clone()
    new_val[idx] = pred[idx]
    new_val = torch.cat((true_previous, new_val), dim=0)

    return new_val


def discriminator_loss(pred, mask):
    assert isinstance(pred, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    nb_subjects = mask.shape[0]

    o_mask = pred.new_tensor(mask, dtype=torch.float).squeeze()

    loss = F.binary_cross_entropy(pred, o_mask, reduction='sum') / nb_subjects
    # loss = (loss_fake + loss_real) / nb_subjects
    return loss


def adversarial_loss(prob, mask):
    assert isinstance(mask, torch.Tensor)
    nb_subjects = mask.shape[0]

    o_mask = prob.new_tensor(mask, dtype=torch.long).squeeze()
    idx = torch.where(o_mask == 0)
    label = Variable(torch.Tensor(idx[0].shape[0]).fill_(1.0), requires_grad=False).to(device)
    prob_new = prob[idx]
    # loss = -1 / torch.log(1-prob_new)

    loss = F.binary_cross_entropy(prob_new, label)
    # loss = torch.sum(loss)
    return loss / nb_subjects


if __name__ == '__main__':
    # 0 GPU device
    gpu_id = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    np.random.seed(1)
    torch.manual_seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_classes', type=int, help="",
                        default="2")
    parser.add_argument('--nb_layers1', type=int, help="",
                        default="3")
    parser.add_argument('--nb_layers2', type=int, help="",
                        default="3")
    parser.add_argument('--h_mri', type=int, help="",
                        default="128")
    parser.add_argument('--h_pet', type=int, help="",
                        default="128")
    parser.add_argument('--drop', type=float, help="dropout",
                        default="0.1")
    parser.add_argument('-lr', '--learning_rate', type=float, help="learning rate",
                        default="5e-3")
    parser.add_argument('-lr_ds', '--learning_rate_ds', type=float, help="learning rate of discriminator",
                        default="1e-3")
    parser.add_argument('--weight_decay', type=float, help="weight_decay",
                        default="5e-4")
    parser.add_argument('--batch_size', type=int, help="batch_size",
                        default="32")
    parser.add_argument('--epochs', type=int, help="batch_size",
                        default="500")
    parser.add_argument('-we', '--weight_ent', type=float, help="learning rate",
                        default="10")
    parser.add_argument('-wm', '--weight_mae', type=float, help="learning rate",
                        default="2")
    parser.add_argument('-wa', '--weight_adv', type=float, help="learning rate",
                        default="100")
    parser.add_argument('-head', '--head_numbder', type=float, help="learning rate",
                        default="4")
    parser.add_argument('--data_path', type=str, help="data load path",
                        default="data")
    parser.add_argument('--save_path', type=str, help="model saving path",
                        default="model")
    parser.add_argument('--metric', type=str, help="metric for validation",
                        default="bca")

    best_avg_acc1 = 0
    best_avg_auc1 = 0
    best_avg_bca1 = 0
    best_avg_acc2 = 0
    best_avg_auc2 = 0
    best_avg_bca2 = 0
    best_avg_acc3 = 0
    best_avg_auc3 = 0
    best_avg_bca3 = 0
    args = parser.parse_args()
    for fold in range(5):
        print(f'Start Train ========  fold{fold + 1}')
        ## data init
        train_data, train_convert, train_label, mask_train_data, mask_train_label \
            = np.load(os.path.join(args.data_path, f'train_data_{fold + 1}.npy')), \
              np.load(os.path.join(args.data_path, f'train_convert_{fold + 1}.npy')), \
              np.load(os.path.join(args.data_path, f'train_label_{fold + 1}.npy')), \
              np.load(os.path.join(args.data_path, f'mask_train_data_{fold + 1}.npy')), \
              np.load(os.path.join(args.data_path, f'mask_train_label_{fold + 1}.npy'))

        valid_data, valid_convert, valid_label, mask_valid_data, mask_valid_label \
            = np.load(os.path.join(args.data_path, f'valid_data_{fold + 1}.npy')), \
              np.load(os.path.join(args.data_path, f'valid_convert_{fold + 1}.npy')), \
              np.load(os.path.join(args.data_path, f'valid_label_{fold + 1}.npy')), \
              np.load(os.path.join(args.data_path, f'mask_valid_data_{fold + 1}.npy')), \
              np.load(os.path.join(args.data_path, f'mask_valid_label_{fold + 1}.npy'))

        test_data, test_convert, test_label, mask_test_data, mask_test_label \
            = np.load(os.path.join(args.data_path, f'test_data_{fold + 1}.npy')), \
              np.load(os.path.join(args.data_path, f'test_convert_{fold + 1}.npy')), \
              np.load(os.path.join(args.data_path, f'test_label_{fold + 1}.npy')), \
              np.load(os.path.join(args.data_path, f'mask_test_data_{fold + 1}.npy')), \
              np.load(os.path.join(args.data_path, f'mask_test_label_{fold + 1}.npy'))

        ICV_train = train_data[:, :, 0][:, :, None]
        ICV_valid = valid_data[:, :, 0][:, :, None]
        ICV_test = test_data[:, :, 0][:, :, None]
        train_data = train_data[:, :, 1:]
        valid_data = valid_data[:, :, 1:]
        test_data = test_data[:, :, 1:]

        mask_train_data = mask_train_data[:, :, 1:]
        mask_valid_data = mask_valid_data[:, :, 1:]
        mask_test_data = mask_test_data[:, :, 1:]

        valid_data[:, 1:, :] = np.nan
        mask_valid_data[:, 1:, :] = False
        test_data[:, 1:, :] = np.nan
        mask_test_data[:, 1:, :] = False
        train_label = train_label - 1
        valid_label = valid_label - 1
        test_label = test_label - 1

        train_data[:, :, :90] = train_data[:, :, :90] / ICV_train
        valid_data[:, :, :90] = valid_data[:, :, :90] / ICV_valid
        test_data[:, :, :90] = test_data[:, :, :90] / ICV_test

        CLF1 = StandardScaler()
        train_data[:, :, :90] = CLF1.fit_transform(train_data[:, :, :90].reshape(-1, 90)).reshape(-1, 5, 90)
        valid_data[:, :, :90] = CLF1.transform(valid_data[:, :, :90].reshape(-1, 90)).reshape(-1, 5, 90)
        test_data[:, :, :90] = CLF1.transform(test_data[:, :, :90].reshape(-1, 90)).reshape(-1, 5, 90)

        CLF2 = StandardScaler()
        train_data[:, :, 90:] = CLF2.fit_transform(train_data[:, :, 90:].reshape(-1, 90)).reshape(-1, 5, 90)
        valid_data[:, :, 90:] = CLF2.transform(valid_data[:, :, 90:].reshape(-1, 90)).reshape(-1, 5, 90)
        test_data[:, :, 90:] = CLF2.transform(test_data[:, :, 90:].reshape(-1, 90)).reshape(-1, 5, 90)

        # 2 init_model/optim
        # hyper-parameter

        lr = args.learning_rate
        lr_ds = args.learning_rate_ds
        weight_decay = args.weight_decay
        batch_size = args.batch_size
        epochs = args.epochs
        nb_classes = args.nb_classes
        nb_layers1 = args.nb_layers1
        nb_layers2 = args.nb_layers2
        h_mri = args.h_mri
        h_pet = args.h_pet
        i_drop = h_drop = args.drops
        patience = args.patience
        delta = args.delta
        weight_ent = args.weight_ent
        weight_mae = args.weight_mae
        weight_adv = args.weight_adv
        head = args.head_number
        save_path = args.save_path
        metric = args.metric

        model = MinimalRNN(nb_classes=nb_classes, nb_mri=90, nb_pet=90, h_mri=h_mri, h_pet=h_pet,
                           h_drop=h_drop, i_drop=i_drop, nb_layers1=nb_layers1, nb_layers2=nb_layers2,
                           alpha=0.9, beta=0.1).to(device)

        discriminator = Discriminator(train_data).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_ds, weight_decay=weight_decay)
        criteon = nn.CrossEntropyLoss()
        print(model)

        # 3 start train
        train_dataset = DealDataset1(train_data, train_convert, train_label, mask_train_data, mask_train_label)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)

        best_valid_acc = 0
        best_valid_auc = 0
        best_valid_bca = 0
        best_valid_epoch = 0
        for epoch in range(epochs):
            model.train()
            total_ent = total_mae = 0
            total = 0
            correct = 0
            mae_total1 = 0
            mae_total2 = 0
            mae_total3 = 0
            ent_total1 = 0
            # ent_total2 = 0
            ent_total3 = 0
            for iter, data in enumerate(train_loader):
                ## data trans
                val_seq, convert, cat_seq, mask_val, mask_cat = data
                _mask_val = mask_val.clone().to(device)
                _val_seq = val_seq.clone().float().to(device)
                val_seq, pet_seq = val_seq[:, :, :90], val_seq[:, :, 90:]
                mask_val, mask_pet = mask_val[:, :, :90], mask_val[:, :, 90:]
                val_seq = val_seq.numpy()
                pet_seq = pet_seq.numpy()
                convert = convert.numpy()
                cat_seq = cat_seq.numpy()
                mask_val = mask_val.numpy()
                mask_pet = mask_pet.numpy()
                mask_cat = mask_cat.numpy()

                true_cat = (cat_seq[:, :, np.newaxis].transpose(1, 0, 2))
                val_seq = val_seq.transpose(1, 0, 2)
                pet_seq = pet_seq.transpose(1, 0, 2)
                true_val = val_seq[1:]
                true_pet1 = pet_seq
                true_pet2 = pet_seq[1:]
                true_trans = trans_label(cat_seq)

                mask_val = mask_val.transpose(1, 0, 2)
                mask_pet = mask_pet.transpose(1, 0, 2)
                mask_cat = mask_cat.transpose(1, 0)
                mask_cat = mask_cat[:, :, np.newaxis]
                mask_val = mask_val[1:]
                mask_pet1 = mask_pet
                mask_pet2 = mask_pet[1:]
                # mask_cat = mask_cat[1:]

                # pred_cat, pred_val, pred_trans = model(val_seq)
                pred_cat, pred_val, pred_pet_val1, pred_pet_val2, pred_trans \
                    = model(val_seq, pet_seq)

                pred_pet_val = 0.1 * pred_pet_val1.clone()[1:] + 0.9 * pred_pet_val2.clone()

                new_val_mri = impute(pred_val, _val_seq[:, :, :90], _mask_val[:, :, :90])
                new_val_pet = impute(pred_pet_val, _val_seq[:, :, 90:], _mask_val[:, :, 90:])
                new_val = torch.cat((new_val_mri, new_val_pet), dim=-1)
                new_val = new_val.permute(1, 2, 0)
                new_mask = _mask_val.permute(0, 2, 1)

                out_mask = discriminator(new_val.detach())
                mask_loss = discriminator_loss(out_mask, new_mask)

                optimizer_D.zero_grad()
                mask_loss.backward()
                optimizer_D.step()

                adv_loss = adversarial_loss(discriminator(new_val), new_mask)
                ent1 = ent_loss(pred_cat, true_cat, mask_cat)
                # ent2 = ent_loss(pred_pet_cat, true_cat, mask_cat)
                ### can inplca with focal loss
                ent3 = ent_loss_trans(pred_trans, true_trans)

                mae1 = mae_loss(pred_val, true_val, mask_val)
                mae2 = mae_loss(pred_pet_val1, true_pet1, mask_pet1)
                mae3 = mae_loss(pred_pet_val2, true_pet2, mask_pet2)
                loss = weight_ent * (ent1 + ent3) + weight_mae * (mae1 + mae2 + mae3) + weight_adv * adv_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                mae_total1 += mae1
                mae_total2 += mae2
                mae_total3 += mae3
                ent_total1 += ent1
                # ent_total2 += ent2
                ent_total3 += ent3
                pred_cat_ = pred_cat.view(-1, 2)
                pred_cat_ = torch.argmax(pred_cat_, dim=1).cpu().numpy()
                idx_notnan = np.where(mask_cat.reshape(-1) == True)
                true_cat_ = true_cat.reshape(-1)
                total += pred_cat_.shape[0]
                correct += np.equal(pred_cat_[idx_notnan], true_cat_[idx_notnan]).sum()
            acc = correct / total

            print('========= fold {} epocho {} done ============='.format(fold, epoch))
            print('***Entropy Loss : {} / {} '.format(ent_total1 / (iter + 1), ent_total3 / (iter + 1)))
            print('*******MAE Loss : {} / {} / {}'.format(mae_total1 / (iter + 1), mae_total2 / (iter + 1),
                                                          mae_total3 / (iter + 1)))
            print('************acc : {}'.format(acc))
            model.eval()
            with torch.no_grad():
                valid_smri_data, valid_pet_data = valid_data[:, :, :90], valid_data[:, :, 90:]

                valid_acc, valid_auc, valid_bca = model_eval(valid_smri_data, valid_pet_data, valid_label,
                                                             model)

                if valid_bca > best_valid_bca and epoch > 15:
                    best_valid_acc = valid_acc
                    best_valid_auc = valid_auc
                    best_valid_bca = valid_bca
                    best_valid_epoch = epoch
                    torch.save(model, os.path.join(args.save_path, f'model_fold{fold + 1}.pt'))
                print('      valid acc :{}'.format(valid_acc))
                print('      valid auc :{}'.format(valid_auc))
                print('      valid bac :{}'.format(valid_bca))

        print('=============== train done ====================')
        print('best valid acc is {}'.format(best_valid_acc))
        print('best valid auc is {}'.format(best_valid_auc))
        print('best valid bca is {}'.format(best_valid_bca))
        print('Corresponding epoch  is {}'.format(best_valid_epoch))
        print('\n \n')

        ### test
        test_smri_data, test_pet_data = test_data[:, :, :90], test_data[:, :, 90:]
        model_best = torch.load(os.path.join(args.save_path, f'model_fold{fold + 1}.pt'))
        test_acc, test_auc, test_bca = model_best(test_smri_data, test_pet_data, test_label,
                                                  model)

        print('=============== test done ====================')
        print('best test acc is {}'.format(test_acc))
        print('best test auc is {}'.format(test_auc))
        print('best test bca is {}'.format(test_bca))
        print('\n \n')






