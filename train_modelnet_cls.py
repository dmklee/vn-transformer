import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn

import vn_transformer.layers as vn
from data_utils.modelnet_dataset import ModelNetDataset


class InvariantClassifier(nn.Module):
    # Described in Figure 2
    def __init__(self,
                 num_classes,
                 in_features,
                 hidden_features,
                 num_heads,
                 latent_size=None,
                 bias_eps=1e-6,
                ):
        super().__init__()
        self.vn_mlp = nn.Sequential(
            vn.Linear(in_features, hidden_features, bias_eps),
            vn.BatchNorm(hidden_features),
            vn.ReLU(hidden_features),
            vn.Linear(hidden_features, hidden_features, bias_eps),
            vn.BatchNorm(hidden_features),
            vn.ReLU(hidden_features),
        )

        if latent_size is not None:
            self.query_proj = vn.MeanProject(latent_size, hidden_features, hidden_features)
        else:
            self.query_proj = nn.Identity()

        self.vn_transformer = vn.TransformerBlock(f_dim=hidden_features,
                                                  num_heads=num_heads,
                                                  bias_eps=bias_eps,
                                                 )

        self.vn_mlp_inv = nn.Sequential(
            vn.Linear(hidden_features, 3, bias_eps),
            vn.ReLU(3),
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_features*3, hidden_features)
        )


    def forward(self, x):
        '''
        x: tensor of shape [B, num_features, 3, num_points]
        return: tensor of shape [B, num_classes]
        '''
        x = self.vn_mlp(x)

        queries = self.query_proj(x)
        x = self.vn_transformer(x, queries)

        x = vn.invariant(x, self.vn_mlp_inv(x))

        x = torch.flatten(vn.mean_pool(x), start_dim=1)

        x = self.mlp(x)

        return x

    def compute_loss(self, pc, cls):
        out = self.forward(pc)

        loss = nn.CrossEntropyLoss()(out, cls.squeeze(1))
        acc = torch.mean((torch.argmax(out, dim=1, keepdim=True) == cls).float())
        return loss, acc


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != 'cpu':
        torch.cuda.manual_seed(args.seed)

    # create folder and logging
    run_name = f'{args.hidden_features}hidden_{args.num_heads}heads_{args.latent_size}tokens'
    fdir = os.path.join(args.results_path, 'modelnet40', run_name)
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    with open(os.path.join(fdir, 'args.txt'), 'w') as f:
        f.write(str(args.__dict__))

    logger = logging.getLogger('train')
    logger.setLevel(logging.DEBUG)
    logger.handlers =  [logging.StreamHandler(),
                        logging.FileHandler(os.path.join(fdir, "log.txt"))]

    # load data
    train_set = ModelNetDataset(args.data_path, args.pc_size, 'train', normal_channel=False)
    test_set = ModelNetDataset(args.data_path, args.pc_size, 'test', normal_channel=False)

    train_dl = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                           shuffle=True, num_workers=8)
    test_dl = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                           shuffle=False, num_workers=6)

    # create model
    model = InvariantClassifier(num_classes=40,
                                in_features=1,
                                hidden_features=args.hidden_features,
                                num_heads=args.num_heads,
                                latent_size=args.latent_size,
                                bias_eps=args.bias_eps,
                               ).to(args.device)
    model.train()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model Size: {num_params/1e6:.2f}M')

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.num_epochs)

    # load from checkpoint if available
    ckpt_path = os.path.join(fdir, 'checkpoint.pt')
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        my_epoch = checkpoint['epoch'] + 1

        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optim_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        model.train()
    else:
        my_epoch = 1

    for epoch in range(my_epoch, args.num_epochs+1):
        time_before_epoch = time.perf_counter()

        train_loss = []
        train_acc = []
        for batch_idx, batch in enumerate(train_dl):
            points, cls = map(lambda x: x.to(args.device), batch)
            optim.zero_grad()
            loss, acc = model.compute_loss(points, cls)
            train_loss.append(loss.item())
            train_acc.append(acc.item())
            loss.backward()
            optim.step()

        train_loss = np.mean(train_loss)
        train_acc = np.mean(train_acc)

        test_loss = []
        test_acc = []
        model.eval()
        for batch_idx, batch in enumerate(test_dl):
            points, cls = map(lambda x: x.to(args.device), batch)

            with torch.no_grad():
                loss, acc = model.compute_loss(points, cls)
            test_loss.append(loss.item())
            test_acc.append(acc.item())
        model.train()

        test_loss = np.mean(test_loss)
        test_acc = np.mean(test_acc)

        lr_scheduler.step()

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optim_state_dict': optim.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                   }, ckpt_path)

        log_str = f"Epoch {epoch}/{args.num_epochs} | " \
                  + f"LOSS={train_loss:.4f}<{test_loss:.4f}> " \
                  + f"ACC={train_acc:.2%}<{test_acc:.2%}> | " \
                  + f"time={time.perf_counter() - time_before_epoch:.1f}s | " \
                  + f"lr={lr_scheduler.get_last_lr()[0]:.1e}"
        logger.info(log_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--results_path', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--pc_size', type=int, default=1024)

    parser.add_argument('--hidden_features', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--latent_size', type=int, default=64)
    parser.add_argument('--bias_eps', type=float, default=0)
    parser.add_argument('--device', type=str, default='cuda')


    args = parser.parse_args()
    if args.pc_size == args.latent_size:
        args.latent_size = None

    main(args)

