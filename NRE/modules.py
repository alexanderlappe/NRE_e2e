import torch.nn


class RelativeEncoding(torch.nn.Module):
    """
    Takes features and domain probs and computes the relative encoding
    """
    def __init__(self, n_domains, in_features):
        super().__init__()
        self.references = torch.nn.Parameter(torch.randn((n_domains, in_features)))

    def forward(self, x, domain_probs):
        '''
        :param x: tensor of shape (batch_size, in_features)
        :param domain_probs: tensor of shape (batch_size, n_domains)
        :return: tensor of shape (batch_size, in_features)
        '''

        ### Apply relative encoding as an 'attention map'

        v = x - domain_probs @ self.references
        return v


class DomainHead(torch.nn.Module):
    """
    Takes features and computes the probability of each domain
    """
    def __init__(self, n_domains, in_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, n_domains)

    def forward(self, x):
        """
        :param x: tensor of shape (batch_size, in_features)
        :return: tensor of shape (batch_size, n_domains)
        """
        x = self.linear(x)
        x = torch.nn.functional.softmax(x, dim=-1)
        return x


class NREProjection(torch.nn.Module):
    """
    Projects samples onto the tuning vectors
    """
    def __init__(self, n_classes, in_features):
        super().__init__()
        self.tuning_vectors = torch.nn.Parameter(torch.randn((in_features, n_classes - 1)))

    def forward(self, v):
        """
        :param v: relative encodings of shape (batch_size, in_features)
        :return: tensor of shape (batch_size, n_classes - 1)
        """
        return v @ self.tuning_vectors


class NREClassifier(torch.nn.Module):
    """
    Takes class activations and computes class probabilities
    """
    def __init__(self):
        super().__init__()
        self.radius = torch.nn.Parameter(torch.randn(1))

    def forward(self, a):
        """
        :param a: tensor of shape (batch_size, n_classes - 1)
        :return: tensor of shape (batch_size, n_classes)
        """
        norm = torch.linalg.norm(a, axis=1)
        p_neut = 1 / (1 + torch.exp(self.radius - norm)).unsqueeze(-1)
        print('p_neut:', p_neut.shape)

        p_conditional = torch.nn.functional.softmax(a, dim=-1)
        print(p_conditional.shape)
        p = (1 - p_neut) * p_conditional
        p = torch.cat((p_neut, p), dim=-1)

        return p

