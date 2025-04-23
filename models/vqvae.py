import torch

import torch.nn as nn

from .quantizer import Quantizer
from .encoder import (
    Top_encoder,
    Bottom_encoder,
    Top_encoder_large,
    Middle_encoder_large,
    Bottom_encoder_large,
)
from .decoder import (
    Top_decoder,
    Top_decoder_large,
    Decoder,
    Decoder_Large,
)


class VQVAE2(nn.Module):
    def __init__(self, encoder_h_dim, res_h_dim, num_res_layers, k, d, beta):
        super().__init__()

        self.encoder_top = Top_encoder(
            encoder_h_dim, encoder_h_dim, res_h_dim, num_res_layers
        )
        self.pre_quantization_conv_top = nn.Conv2d(encoder_h_dim, d, kernel_size=1)
        self.quantizer_top = Quantizer(k, d, beta)
        self.decoder_top = Top_decoder(d, encoder_h_dim, d, res_h_dim, num_res_layers)
        self.upscale_top = nn.ConvTranspose2d(d, d, 4, stride=2, padding=1)

        self.encoder_btm = Bottom_encoder(3, encoder_h_dim, res_h_dim, num_res_layers)
        self.pre_quantization_conv_btm = nn.Conv2d(encoder_h_dim + d, d, kernel_size=1)
        self.quantizer_btm = Quantizer(k, d, beta)

        self.decoder = Decoder(2 * d, encoder_h_dim, 3, res_h_dim, num_res_layers)

    def forward(self, x):
        e, top_loss, btm_loss, perplexity, embeddings = self.encode(x)

        # The decoder ...[] takes as input all levels of the quantized latent hierarchy
        x_hat = self.decoder(e)

        return x_hat, top_loss, btm_loss, perplexity, embeddings

    def encode(self, x):
        B = x.shape[0]
        D = self.quantizer_top.d

        # the encoder network first transforms and downsamples the image by a factor of 4 ...
        h_btm = self.encoder_btm(x)
        # ... Another stack of residual blocks then further scales down the representations by a factor of two.
        h_top = self.encoder_top(h_btm)

        # Top part of fig 2a.
        h_top = self.pre_quantization_conv_top(h_top)
        top_loss, e_top, perplexity, top_embeddings = self.quantizer_top(h_top)
        dec_top = self.decoder_top(h_top)
        T_top = e_top.shape[2] * e_top.shape[3]

        # Middle part of fig 2a.
        h_btm = self.pre_quantization_conv_btm(torch.cat((h_btm, dec_top), dim=1))
        btm_loss, e_btm, perplexity, btm_embeddings = self.quantizer_btm(h_btm)
        T_btm = e_btm.shape[2] * e_btm.shape[3]

        # Upsample e_top so it can be concatenated with e_btm before final decoder.
        e_top = self.upscale_top(e_top)

        e = torch.cat((e_btm, e_top), dim=1)

        btm_embeddings = btm_embeddings.view(B, T_btm, D)
        top_embeddings = top_embeddings.view(B, T_top, D)

        return (
            e,
            top_loss,
            btm_loss,
            perplexity,
            # torch.cat([btm_embeddings, top_embeddings], dim=1),
            top_embeddings,
        )


# Three layer VQVAE
class VQVAE2_large(nn.Module):
    def __init__(self, encoder_h_dim, res_h_dim, num_res_layers, k, d, beta):
        super().__init__()

        self.encoder_top = Top_encoder_large(
            encoder_h_dim, encoder_h_dim, res_h_dim, num_res_layers
        )
        self.pre_quantization_conv_top = nn.Conv2d(encoder_h_dim, d, kernel_size=1)
        self.quantizer_top = Quantizer(k, d, beta)
        self.decoder_top = Top_decoder_large(
            d, encoder_h_dim, encoder_h_dim, res_h_dim, num_res_layers
        )
        self.upscale_top = nn.ConvTranspose2d(d, d, 4, stride=2, padding=1)

        self.encoder_mid = Middle_encoder_large(
            encoder_h_dim, encoder_h_dim, res_h_dim, num_res_layers
        )
        self.pre_quantization_conv_mid = nn.Conv2d(encoder_h_dim * 2, d, kernel_size=1)
        self.quantizer_mid = Quantizer(k, d, beta)
        self.decoder_mid = Top_decoder_large(
            2 * d, encoder_h_dim, encoder_h_dim, res_h_dim, num_res_layers
        )
        self.upscale_mid = nn.ConvTranspose2d(d, d, 4, stride=2, padding=1)

        self.encoder_btm = Bottom_encoder_large(
            3, encoder_h_dim, res_h_dim, num_res_layers
        )
        self.pre_quantization_conv_btm = nn.Conv2d(2 * encoder_h_dim, d, kernel_size=1)
        self.quantizer_btm = Quantizer(k, d, beta)

        self.decoder = Decoder_Large(2 * d, encoder_h_dim, 3, res_h_dim, num_res_layers)

    def forward(self, x):
        e, top_loss, mid_loss, btm_loss, perplexity, embeddings = self.encode(x)

        x_hat = self.decoder(e)

        return x_hat, top_loss, mid_loss, btm_loss, perplexity, embeddings

    def encode(self, x):
        B = x.shape[0]
        D = self.quantizer_top.d

        h_btm = self.encoder_btm(x)
        h_mid = self.encoder_mid(h_btm)
        h_top = self.encoder_top(h_mid)

        h_top = self.pre_quantization_conv_top(h_top)
        top_loss, e_top, perplexity, top_embeddings = self.quantizer_top(h_top)
        dec_top = self.decoder_top(h_top)
        T_top = e_top.shape[2] * e_top.shape[3]

        h_mid = self.pre_quantization_conv_mid(torch.cat((h_mid, dec_top), dim=1))
        mid_loss, e_mid, perplexity, mid_embeddings = self.quantizer_mid(h_mid)
        T_mid = e_mid.shape[2] * e_mid.shape[3]

        e_top = self.upscale_top(e_top)

        dec_mid = self.decoder_mid(torch.cat((e_mid, e_top), dim=1))

        h_btm = self.pre_quantization_conv_btm(torch.cat((h_btm, dec_mid), dim=1))
        btm_loss, e_btm, perplexity, btm_embeddings = self.quantizer_btm(h_btm)
        T_btm = e_btm.shape[2] * e_btm.shape[3]

        e_mid = self.upscale_mid(e_mid)

        e = torch.cat((e_btm, e_mid), dim=1)

        btm_embeddings = btm_embeddings.view(B, T_btm, D)
        mid_embeddings = mid_embeddings.view(B, T_mid, D)
        top_embeddings = top_embeddings.view(B, T_top, D)

        return (
            e,
            top_loss,
            mid_loss,
            btm_loss,
            perplexity,
            torch.cat([mid_embeddings, top_embeddings], dim=1),
        )
