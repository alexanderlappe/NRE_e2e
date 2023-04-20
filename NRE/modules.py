import torch.nn


class RelativeEncoding(torch.nn.Module):
    def __init__(self, n_domains, in_features):
        super().__init__()
        self.references = torch.nn.Parameter(torch.randn((n_domains, in_features)))

    def forward(self, x, domain_probs):
        '''
        :param x: tensor of shape (batch_size, in_features)
        :param domain_probs: tensor of shape (batch_size, n_domains)
        :return:
        '''

        ### Apply relative encoding as an 'attention map'

        return x - domain_probs @ self.references


class DomainHead(torch.nn.Module):
    def __init__(self, n_domains, in_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, n_domains)

    def forward(self, x):
        x = self.linear(x)
        x = torch.nn.functional.softmax(x, dim=-1)
        return x


class NREProjection(torch.nn.Module):
    def __init__(self, n_classes, in_features):
        super().__init__()
        self.tuning_vectors = torch.nn.Parameter(torch.randn((in_features, n_classes - 1)))

    def forward(self, x):
        return x @ self.tuning_vectors


class NREClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.radius = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        norm = torch.linalg.norm(x, axis=1)
        p_neut = 1 / (1 + torch.exp(self.radius - norm)).unsqueeze(-1)
        print('p_neut:', p_neut.shape)

        p_conditional = torch.nn.functional.softmax(x, dim=-1)
        print(p_conditional.shape)
        p = (1 - p_neut) * p_conditional
        p = torch.cat((p_neut, p), dim=-1)

        return p

