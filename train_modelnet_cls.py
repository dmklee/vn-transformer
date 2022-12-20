import argparse

from src.models import InvariantClassifier
from data_utils.modelnet_dataset import ModelNetDataset


def main(args):
    # load data
    train_set = ModelNetDataset(args.data_path, args.pc_size, 'train', normal_channel=False)
    test_set = ModelNetDataset(args.data_path, args.pc_size, 'test', normal_channel=False)

    train_dl = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                           shuffle=True, num_workers=4)
    test_dl = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                           shuffle=False, num_workers=4)

    # create model
    model = InvariantClassifier(num_classes=40,
                                in_features=3,
                                hidden_features=args.hidden_features,
                                num_heads=args.num_heads,
                                latent_size=args.latent_size,
                                bias_eps=args.bias_eps,
                               ).to(args.device)
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.num_epochs)

    for epoch in range(1, args.num_epochs+1):
        train_loss = []
        for batch_idx, batch in enumerate(train_dl):
            points, cls = map(lambda x: x.to(args.device), batch)
            optim.zero_grad()
            loss = model.compute_loss(points, cls)
            train_loss.append(loss.item())
            loss.backward()
            optim.step()

        train_loss = np.mean(train_loss)

        test_loss = []
        model.eval()
        for batch_idx, batch in enumerate(test_dl):
            points, cls = map(lambda x: x.to(args.device), batch)
            with torch.no_grad():
                loss = model.compute_loss(pc, cls)
            test_loss.append(loss.item())
        model.train()

        test_loss = np.mean(test_loss)

        lr_scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--pc_size', type=int, default=1024)
    parser.add_argument('--latent_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    main(args)

