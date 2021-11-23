import torch
import torch.nn as nn
from functions import ReverseLayerF



class DSN(nn.Module):
    def __init__(self, channel, code_size=100, n_class=10):
        super(DSN, self).__init__()
        self.code_size = code_size
        self.channel = int(channel)


        # private source encoder
        self.source_encoder_conv = nn.Sequential(
            nn.Conv2d(self.channel, 32, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.source_encoder_fc = nn.Sequential(
            nn.Linear(7*7*64, code_size),
            nn.ReLU(True)
        )


        # private target encoder
        self.target_encoder_conv = nn.Sequential(
            nn.Conv2d(self.channel, 32, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.target_encoder_fc = nn.Sequential(
            nn.Linear(7*7*64, code_size),
            nn.ReLU(True)
        )


        # shared encoder
        self.shared_encoder_conv = nn.Sequential(
            nn.Conv2d(self.channel, 32, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 48, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.shared_encoder_fc = nn.Sequential(
            nn.Linear(7*7*48, code_size),
            nn.ReLU(True)
        )


        # classify 10 numbers
        self.shared_encoder_pred_class = nn.Sequential(
            nn.Linear(code_size, 100),
            nn.ReLU(True),
            nn.Linear(100, n_class)
        )
        self.shared_encoder_pred_domain = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 2)
        )


        # shared decoder
        self.shared_decoder_fc = nn.Sequential(
            nn.Linear(code_size, self.channel*14*14),
            nn.ReLU(True)
        )
        self.shared_decoder_conv = nn.Sequential(
            nn.Conv2d(self.channel, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=self.channel, kernel_size=3, padding=1)
        )

    def forward(self, input_data, mode, rec_scheme, p=0.0):

        result = []

        if mode == 'source':

            # source private encoder
            private_feat = self.source_encoder_conv(input_data)
            private_feat = private_feat.view(-1, 64 * 7 * 7)
            private_code = self.source_encoder_fc(private_feat)

        elif mode == 'target':

            # target private encoder
            private_feat = self.target_encoder_conv(input_data)
            private_feat = private_feat.view(-1, 64 * 7 * 7)
            private_code = self.target_encoder_fc(private_feat)

        result.append(private_code)

        # shared encoder
        shared_feat = self.shared_encoder_conv(input_data)
        shared_feat = shared_feat.view(-1, 48 * 7 * 7)
        shared_code = self.shared_encoder_fc(shared_feat)
        result.append(shared_code)

        reversed_shared_code = ReverseLayerF.apply(shared_code, p)
        domain_label = self.shared_encoder_pred_domain(reversed_shared_code)
        result.append(domain_label)

        if mode == 'source':
            class_label = self.shared_encoder_pred_class(shared_code)
            result.append(class_label)

        # shared decoder

        if rec_scheme == 'share':
            union_code = shared_code
        elif rec_scheme == 'all':
            union_code = private_code + shared_code
        elif rec_scheme == 'private':
            union_code = private_code

        rec_vec = self.shared_decoder_fc(union_code)
        rec_vec = rec_vec.view(-1, self.channel, 14, 14)

        rec_code = self.shared_decoder_conv(rec_vec)
        result.append(rec_code)

        return result
if __name__ == '__main__':
    model = DSN(channel=1)
    s_img = torch.rand(1,1,28,28)
    model.eval()
    result = model(input_data=s_img, mode='source', rec_scheme='all')
    source_privte_code, source_share_code, source_domain_label, source_class_label, source_rec_code = result
    print(source_privte_code.size(), source_share_code.size(), source_domain_label.size(), source_class_label.size(), source_rec_code.size())
