from models import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor", type=int, default=1) # no upsampling
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='')
    parser.add_argument('--n_epochs', type=int, default=25, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=10, help='number of epochs to update learning rate')
    parser.add_argument('--trainset_dir', type=str, default='data/train/Flickr1024_patches_masked_fixed')
    parser.add_argument('--tb_dir', type=str, default='tensorboard_log_Y/temp')
    return parser.parse_args()


def train(train_loader, cfg):
    net = PASSRnet(cfg.scale_factor).to(cfg.device)
    net.apply(weights_init_xavier)
    cudnn.benchmark = True
    writer = SummaryWriter(log_dir=cfg.tb_dir)

    criterion_mse = torch.nn.MSELoss().to(cfg.device)
    criterion_L1 = L1Loss()
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)

    psnr_epoch = []
    loss_epoch = []
    loss_list = []
    psnr_list = []

    for idx_epoch in range(cfg.n_epochs):
        scheduler.step()
        for idx_iter, (HR_left, _, LR_left, _) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {idx_epoch+1}"):
            b, c, h, w = LR_left.shape
            HR_left, LR_left  = Variable(HR_left).to(cfg.device), Variable(LR_left).to(cfg.device)

            SR_left, (M_right_to_left, M_left_to_right), (M_left_right_left, M_right_left_right), \
            (V_left_to_right, V_right_to_left) = net(LR_left, LR_left, is_training=1)

            ### loss_SR
            loss_SR = criterion_mse(SR_left, HR_left)
            loss = loss_SR

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            psnr_epoch.append(cal_psnr(HR_left[:,:,:,64:].data.cpu(), SR_left[:,:,:,64:].data.cpu()))
            loss_epoch.append(loss.data.cpu())

        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            writer.add_scalar('Epoch/Loss', loss_list[-1], idx_epoch)
            psnr_list.append(float(np.array(psnr_epoch).mean()))
            writer.add_scalar('Epoch/PSNR', psnr_list[-1], idx_epoch)
            print('Epoch----%5d, loss---%f, PSNR---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean()), float(np.array(psnr_epoch).mean())))

            save_ckpt({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'loss': loss_list,
                'psnr': psnr_list,
            }, save_path = 'log_onlyL/', filename='PASSRnet' + '_epoch' + str(idx_epoch + 1) + '_fixed_mse.pth.tar')
            psnr_epoch = []
            loss_epoch = []

    writer.close()

def main(cfg):
    train_set = TrainSetLoader(dataset_dir=cfg.trainset_dir, cfg=cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=cfg.batch_size, shuffle=True)
    train(train_loader, cfg)

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)