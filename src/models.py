import torch
import torch.nn as nn

import src.layers as vn


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
            vn.ReLU(hidden_features)
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
            vn.ReLU(3)
        )

        self.mlp = nn.Linear(hidden_features*3, num_classes)


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

        loss = nn.CrossEntropyLoss()(out, cls)
        return loss


class EquivariantPredictor(nn.Module):
    '''
    Figure 1b
    '''
    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 out_features: int,
                 num_heads: int,
                 bias_eps: float=1e-6,
                ):
        super().__init__()
        self.vn_mlp = nn.Sequential(
            vn.Linear(in_features, hidden_features, bias_eps),
            vn.BatchNorm(hidden_features),
            vn.ReLU(hidden_features)
        )

        self.vn_transformer = vn.TransformerBlock(f_dim=hidden_features,
                                                  num_heads=num_heads,
                                                  bias_eps=bias_eps,
                                                 )

        self.vn_mlp = nn.Sequential(
            vn.Linear(in_features, hidden_features, bias_eps),
            vn.BatchNorm(hidden_features),
            vn.ReLU(hidden_features)
            vn.Linear(hidden_features, out_features, bias_eps),
        )

    def forward(self, x):
        '''
        x: tensor of shape [B, num_features, 3, num_points]
        return: tensor of shape [B, out_features, 3, num_points]
        '''
        x = self.vn_mlp(x)

        queries = self.query_proj(x)
        x = self.vn_transformer(x, queries)

        x = vn.invariant(x, self.vn_mlp_inv(x))

        x = torch.flatten(vn.mean_pool(x), start_dim=1)

        x = self.mlp(x)

        return x

    def compute_loss(self, pc, gt_features):
        pred_features = self.forward(pc)

        loss = nn.MSELoss()(pred_features, gt_features)
        return loss


if __name__ == "__main__":
    B, C, N = 1, 1, 512
    x = torch.randn((B, C, 3, N), dtype=torch.float32).cuda()

    model = InvariantClassifier(num_classes=10,
                                in_features=C,
                                hidden_features=64,
                                num_heads=16,
                                latent_size=None,
                               ).cuda()

    print(f'{model.num_params/1e6:.3f} params')
