from .networks import *
from .ops import clip, BinaryQuantize


# Cross-channel entropy model, without adap qp networks in journal

class TextureEntropyBottleneck(BaseNetwork):
    def __init__(self, channels, opt):
        super(TextureEntropyBottleneck, self).__init__()
        self._ch = int(channels)
        self.opt = opt
        if self._ch < 32:  # channels must >= 8
            self.hyper_encoder = nn.Sequential(
                nn.Conv2d(self._ch, self._ch // 2, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 2, self._ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 4, self._ch // 8, 1, 1)
            )

            self.hyper_decoder = nn.Sequential(
                nn.Conv2d(self._ch // 8, self._ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 4, self._ch // 2, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 2, self._ch * 2, 1, 1)
            )
            self.factorized = FullFactorizedModel(self._ch // 8, (3, 3, 3), 1e-9, True)
        else:
            self.hyper_encoder = nn.Sequential(
                nn.Conv2d(self._ch, self._ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 4, self._ch // 16, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 16, self._ch // 32, 1, 1)
            )

            self.hyper_decoder = nn.Sequential(
                nn.Conv2d(self._ch // 32, self._ch // 16, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 16, self._ch // 2, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 2, self._ch * 2, 1, 1)
            )

            self.factorized = FullFactorizedModel(self._ch // 32, (3, 3, 3), 1e-9, True)

        self.quantizer = Quantizer()
        self.conditional = GaussianModel(1e-1, 1e-9, True)

    def forward(self, y):
        z = self.hyper_encoder(y)
        z_hat, z_prob = self.factorized(z)
        u = self.hyper_decoder(z_hat)
        y_hat = self.quantizer(y, self.training)
        loc, scale = u.split(self._ch, dim=1)
        y_prob, y_decor = self.conditional(y_hat, loc, scale)
        length = torch.sum(-torch.log2(z_prob)) + torch.sum(-torch.log2(y_prob))
        return y_hat, z_hat, length, y_prob

    @property
    def offset(self):
        return self.factorized.integer_offset()

