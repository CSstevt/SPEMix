import argparse
import math

from mmcv import DataLoader
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import torch
from utils.Echo import Echo as dataset
import utils.mb_sup_loss as mb_sup_loss
from utils.momentum_update import update
from utils.transform import TransformFixMatch
from utils.evaluate import test, val
from torch.optim.lr_scheduler import LambdaLR
parser = argparse.ArgumentParser()
parser.add_argument('--train_path', default='/datasets/train', type=str,
                    help='the path of training dataset')
parser.add_argument('--val_path', default='/datasets/val', type=str,
                    help='the path of val dataset')
parser.add_argument('--test_path', default='/datasets/test', type=str,
                    help='the path of test dataset')
parser.add_argument('--unlabeled_path', default='/datasets/unlabeled', type=str,
                    help='the path of test dataset')
parser.add_argument('--random_seed', default=3, type=int,
                    help='random seed for training procedure')
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--unlabelded_batch_size', type=int, default=64)
parser.add_argument('--num_train_iter', type=int, default=524288,
                    help='total number of training iterations')
parser.add_argument('--num_warmup_iter', type=int, default=0,
                    help='cosine linear warmup iterations')
parser.add_argument('--lr_cycle_length', type=int, default=63000)
parser.add_argument('--num_eval_iter', type=int, default=210,
                    help='evaluation frequency')
parser.add_argument('--lr_mixblock', default=0.01, type=float,
                    help='initial learning rate of the teacher model')
parser.add_argument('--lr', default=0.001, type=float,
                    help='initial learning rate of the student model')
parser.add_argument('batch_size', default=64, type=int,
                    help='initial barch size')
parser.add_argument('--momentum', default=0.999, type=float)
parser.add_argument('--lamda', default=1, type=int)
args = parser.parse_args()

def no_repeat_shuffle_idx(batch_size_this, ignore_failure=False):
    idx_shuffle = torch.randperm(batch_size_this).cuda()
    idx_original = torch.tensor([i for i in range(batch_size_this)]).cuda()
    idx_repeat = False
    for i in range(10):  # try 10 times
        if (idx_original == idx_shuffle).any() == True:
            idx_repeat = True
            idx_shuffle = torch.randperm(batch_size_this).cuda()
        else:
            idx_repeat = False
            break
        # hit: prob < 1.2e-3
    if idx_repeat == True and ignore_failure == False:
        # way 2: repeat prob = 0, but too simple!
        idx_shift = np.random.randint(1, batch_size_this - 1)
        idx_shuffle = torch.tensor(  # shift the original idx
            [(i + idx_shift) % batch_size_this for i in range(batch_size_this)]).cuda()
    return idx_shuffle

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    #                                     num_training_steps, #total train iterations
                                    lr_cycle_length,  # total train iterations
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, float(lr_cycle_length) - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def train(device, epoch_index, student_model,teacher_model, batch_num, item, index1, index2,lamda1,optimizer, optimizer1, schedule, schedule1, uitem):
    p_bar = tqdm(range(batch_num), disable=False)
    for i in range(batch_num):
        optimizer.zero_grad()
        optimizer1.zero_grad()
        teacher_model.zero_grad()
        student_model.zero_grad()
        x, y = next(item)
        (xu, _), _ = next(uitem)
        xu = xu.to(device)
        x = x.to(device)
        y = y.to(device)
        li = [name for name, pa in student_model.named_parameters()]
        for name, pa in teacher_model.named_parameters():
            if name in li:
                pa.requires_grad_(False)
        # 对两个弱增强的无标签计算oploss
        re1 = student_model(x)
        logit_x = re1['out']  # optional out ,mcl,opresult
        logitcls_x = re1['mcl']
        index = index1
        u1 = xu
        u2 = xu[index]
        result_u1 = student_model(u1)
        result_u2 = student_model(u2)
        oplogits_1 = result_u1['opresult']
        oplogits_2 = result_u2['opresult']
        logit_1 = result_u1['out']
        logit_2 = result_u2['out']
        muti_1 = result_u1['mcl']
        muti_2 = result_u2['mcl']
        multiloss = mb_sup_loss(logitcls_x, y)
        p = F.softmax(logit_1, dim=-1)
        target_xu = p.detach()
        tmp_range = torch.arange(0, xu.shape[0]).long().to(device)
        q = torch.zeros((xu.shape[0], 5)).to(device)
        logits_mb = muti_1.view(xu.shape[0], 2, -1)
        r = F.softmax(logits_mb, 1)
        out_scores = torch.sum(p * r[tmp_range, 0, :], 1)
        in_mask = (out_scores < 0.5)
        o_neg = r[tmp_range, 0, :]
        o_pos = r[tmp_range, 1, :]
        q[:, :4] = target_xu * o_pos
        q[:, 4] = torch.sum(target_xu * o_neg, 1)
        targets_u1 = q.detach()
        max_probs1, targets_u1 = torch.max(targets_u1, dim=-1)
        masku1 = max_probs1.ge(0.).float()
        p2 = F.softmax(logit_2, dim=-1)
        target_xu = p2.detach()
        tmp_range = torch.arange(0, xu.shape[0]).long().to(device)
        q2 = torch.zeros((xu.shape[0], 5)).to(device)
        logits_mb2 = muti_2.view(xu.shape[0], 2, -1)
        r = F.softmax(logits_mb2, 1)
        o_neg = r[tmp_range, 0, :]
        o_pos = r[tmp_range, 1, :]
        q2[:, :4] = target_xu * o_pos
        q2[:, 4] = torch.sum(target_xu * o_neg, 1)
        targets_u2 = q2.detach()
        max_probs1, targets_u1 = torch.max(targets_u2, dim=-1)
        masku2 = max_probs1.ge(0.).float()
        lossu1 = F.cross_entropy(oplogits_1, targets_u1, reduction='mean') * masku1
        lossu2 = F.cross_entropy(oplogits_2, targets_u2, reduction='mean') * masku2
        lossu1 = lossu1.mean()
        lossu2 = lossu2.mean()
        lossu = lossu1 + lossu2
        lossu = lossu.mean()
        teacher_model(xu)
        l = teacher_model.get_mask_loss()['loss']
        l.backward(retain_graph=True)
        masku = teacher_model.mask1
        mix_im = u1 * masku[:, 0, :, :].unsqueeze(1) + u2 * masku[:, 1, :, :].unsqueeze(1)  # 混合后的图像
        logit_mix = student_model(mix_im)['out']
        pm = F.softmax(logit_1, dim=-1)
        max_probs1, targets_u1 = torch.max(pm, dim=-1)
        mask1 = max_probs1.ge(0.5).float()
        pm2 = F.softmax(logit_2, dim=-1)
        max_probs2, targets_u2 = torch.max(pm2, dim=-1)
        mask2 = max_probs2.ge(0.5).float()
        lossin = ((lamda1 * (F.cross_entropy(logit_mix, targets_u1, reduction='mean') * mask1 * in_mask)) +
                  ((1 - lamda1) * (F.cross_entropy(logit_mix, targets_u2, reduction='mean') * mask2 * in_mask)))
        lossin = lossin.mean()
        lossl = F.cross_entropy(logit_x, y, reduction='mean')
        lossl.backward(retain_graph=True)
        lossin.backward(retain_graph=True)
        lossu1.backward(retain_graph=True)
        lossu2.backward(retain_graph=True)
        multiloss.backward(retain_graph=True)
        loss = lossl + lossu + multiloss + l + lossin
        with torch.no_grad():
            out2 = teacher_model(x)
            _, predicted = torch.max(student_model(x)['out'], 1)
            accuracy = (predicted == y).sum().item() / y.size(0)
        optimizer1.step()
        optimizer.step()

        schedule.step()
        schedule1.step()
        # 动量更新
        update(student_model, teacher_model, 0.999)
        p_bar.set_description(
            "Train Epoch: {epoch}/{epoch_total:4}. Loss: {loss:.4f}. Lossl: {lossl:.4f}. Lossu: {uloss:.4f}. Loss2: {loss2:.4f}.acc: {acc:.4f}.Lr: {lr:.6f}.Lr1: {lr1:.6f}. ".format(
                epoch=epoch_index,
                epoch_total=1000,
                loss=loss.item(),
                lossl=lossl.item(),  # Replace with your actual loss values
                loss2=multiloss.item(),  # Replace with your actual loss values
                # lossin=lossin,
                lr=schedule.get_last_lr()[0],
                lr1=schedule1.get_last_lr()[0],
                acc=accuracy,
                uloss=lossu.item()
            ))
        p_bar.update()
    p_bar.close()
    return loss.item , accuracy
def main (student_model,teacher_model,args):
    student_model.train()
    teacher_model.train()

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=112,
                              padding=int(112 * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        #         transforms.Normalize(mean=echo_mean, std=echo_std)
    ])
    val_transform = transforms.Compose([
        #        transforms.Grayscale(num_output_channels=3),  # 将图像转换为彩色图像
        transforms.ToTensor(), ])

    train_data = dataset(args.train_path, transform_fn=train_transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    val_data = dataset(args.val_path, transform_fn=val_transform)
    val_loader = DataLoader(args.val_path, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    test_data = dataset(args.test_path, transform_fn=val_transform)
    test_loader = DataLoader(args.test_path, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    unlabeled_data = dataset(args.unlabeled_path, transform_fn=TransformFixMatch())
    unlabeled_loader = DataLoader(args.unlabeled_path, batch_size=args.unlabelded_batch_size, shuffle=True,
                                  num_workers=4, drop_last=False)

    train_iter = iter(train_loader)
    unlabeled_iter = iter(unlabeled_loader)
    optimizer1 = optim.SGD(student_model.parameters(), lr=args.lr_mixblock, momentum=args.momentum)
    optimizer2 = optim.SGD(teacher_model.parameters(), lr= args.lr, momentum=args.momentum)

    lambda1 = np.random.beta(args.lamda, args.lamda)
    lambda2 = np.random.beta(args.lamda, args.lamda)
    scheduler1 = get_cosine_schedule_with_warmup(optimizer1, 0, 210000)
    scheduler2 = get_cosine_schedule_with_warmup(optimizer1, 0, 210000)

    index1 = no_repeat_shuffle_idx(args.batch_size)
    index2 = no_repeat_shuffle_idx(args.batch_size)

    for i in range(args.epochs):
        loss_total , accuracy = train('cuda', 0, student_model,teacher_model, args.batch_size, train_iter, index1, index2,lambda1,optimizer1, optimizer2,
            scheduler1,scheduler2, unlabeled_iter)
        test(student_model,test_loader)
        val(student_model, val_loader)
        torch.save(student_model.state_dict(), f'/weights/classification_model_weights_{i}.pth')



if __name__ == '__main__':
    from model.Repvit.student import create_model as create_steudent
    from model.Repvit.teacher import create_model as create_teacher
  
    index1 = no_repeat_shuffle_idx(args.batch_size)
    index2 = no_repeat_shuffle_idx(args.batch_size)
    lambda1 = np.random.beta(args.lamda, args.lamda)
    lambda2 = np.random.beta(args.lamda, args.lamda)
    student= create_student()
    teacher = create_teacher(index1,index2,lambda1,lambda2)
    
    args = parser.parse_args()
    main(student,teacher,args)



