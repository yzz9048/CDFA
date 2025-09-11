import math
import torch
import torch.nn.functional as F

from torch import nn
from datetime import datetime
from einops import reduce
from tqdm.auto import tqdm
from functools import partial
from Models.interpretable_diffusion.transformer import Transformer
from Models.interpretable_diffusion.model_utils import default, identity, extract


# gaussian diffusion trainer class

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if isinstance(alpha, (float, int)):  # single value for binary
            self.alpha = torch.tensor([1.0 - alpha, alpha])
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        # Softmax to get probabilities
        probs = F.softmax(inputs, dim=1)  # (B, C)
        probs = torch.clamp(probs, 1e-8, 1.0)  # avoid log(0)

        # One-hot encoding of targets
        targets_onehot = F.one_hot(targets, num_classes=inputs.size(1)).float()

        # Get the probability of the true class
        pt = (probs * targets_onehot).sum(dim=1)  # (B,)

        # Compute log pt
        log_pt = torch.log(pt)

        # Alpha weighting
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = (self.alpha * targets_onehot).sum(dim=1)
        else:
            alpha_t = 1.0

        # Focal loss formula
        loss = -alpha_t * ((1 - pt) ** self.gamma) * log_pt

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class Diffusion_TS(nn.Module):
    def __init__(
            self,
            seq_length,
            feature_size,
            DeltaT,
            n_layer_enc=3,
            n_layer_dec=6,
            d_model=None,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l1',
            beta_schedule='cosine',
            n_heads=4,
            mlp_hidden_times=4,
            eta=0.,
            attn_pd=0.,
            resid_pd=0.,
            kernel_size=None,
            padding_size=None,
            use_ff=True,
            reg_weight=None,
            **kwargs
    ):
        super(Diffusion_TS, self).__init__()

        self.eta, self.use_ff = eta, use_ff
        self.seq_length = seq_length
        self.DeltaT = DeltaT
        self.feature_size = feature_size
        self.ff_weight = default(reg_weight, math.sqrt(self.seq_length) / 5)

        self.model = Transformer(n_feat=feature_size, n_channel=seq_length,n_shape=DeltaT, n_layer_enc=n_layer_enc, n_layer_dec=n_layer_dec,
                                 n_heads=n_heads, attn_pdrop=attn_pd, resid_pdrop=resid_pd, mlp_hidden_times=mlp_hidden_times,
                                 max_len=seq_length, n_embd=d_model, conv_params=[kernel_size, padding_size], **kwargs)
        self.detecter_criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
        #self.detecter_criterion = nn.CrossEntropyLoss()
        self.log_sigma1 = nn.Parameter(torch.tensor(0.0))  # for regression
        self.log_sigma2 = nn.Parameter(torch.tensor(0.0))  # for classification

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps)  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.fast_sampling = self.sampling_timesteps < timesteps

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)


        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) 

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate reweighting
        
        register_buffer('loss_weight', torch.sqrt(alphas) * torch.sqrt(1. - alphas_cumprod) / betas / 100)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        ) # N(mu,sigam)中的mu
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def output(self, x, history, t, clip, granger, padding_masks=None):
        trend, season, judge, loss_causal = self.model(x, history, t, clip, granger, padding_masks=padding_masks)
        model_output = trend + season
        return model_output, judge, loss_causal

    def model_predictions(self,granger, x, history, t, clip, clip_x_start=False, padding_masks=None):
        if padding_masks is None:
            padding_masks = torch.ones(x.shape[0], self.seq_length, dtype=bool, device=x.device) 

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        x_start, judges, loss_causal = self.output(x, history, t, clip, granger, padding_masks) 
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start) 
        return pred_noise, x_start, judges, loss_causal

    def p_mean_variance(self, granger, x, history, t, clip, clip_denoised=True):
        _, x_start, judges, loss_causal = self.model_predictions(granger, x,history, t, clip)
        if clip_denoised:
            x_start.clamp_(-1., 1.) 
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start, judges, loss_causal

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss 
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _train_loss(self, x_0, t, labels, DeltaT, clip, granger, noise=None, padding_masks=None):
        if DeltaT < x_0.shape[1]:
            history = x_0[:,:-DeltaT,:] 
            target = x_0[:,-DeltaT:,1:] 
        else:
            history = x_0[:-1,:,:]
            target = x_0[1:,:,1:]
        noise = default(noise, lambda: torch.randn_like(target))
        x = self.q_sample(x_start=target, t=t, noise=noise) 

        model_out, judge, loss_causal = self.output(x, history, t, clip, granger, padding_masks)  # 【batch_size,DeltaT,feature_dim】
            
        train_loss = self.loss_fn(model_out, target, reduction='none') 
       
        labels = (labels != 0).any(dim=1).long()
        judge_loss = self.detecter_criterion(judge,labels)

        fourier_loss = torch.tensor([0.])
        if self.use_ff:
            fft1 = torch.fft.fft(model_out.transpose(1, 2), norm='forward')
            fft2 = torch.fft.fft(target.transpose(1, 2), norm='forward')
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = self.loss_fn(torch.real(fft1), torch.real(fft2), reduction='none')\
                           + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction='none')
            train_loss +=  self.ff_weight * fourier_loss
        
        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        train_loss = train_loss.mean()
        
        sigma1_sq = torch.exp(self.log_sigma1) 
        sigma2_sq = torch.exp(self.log_sigma2) ** 2 
        loss = (1 / sigma1_sq) * train_loss + (1 / sigma2_sq) * judge_loss + self.log_sigma1 + self.log_sigma2 

        return loss, loss_causal

    def forward(self, data, labels, DeltaT, clip, granger,**kwargs): 
        b, c, n, device, feature_size, = *data.shape, data.device, self.feature_size
        assert n+1 == feature_size, f'number of variable must be {feature_size+1}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self._train_loss(x_0=data, t=t, labels=labels, DeltaT=DeltaT, clip=clip, granger=granger, **kwargs)
    
    def sample_infill(
        self,
        granger,
        DeltaT,
        shape, 
        history,
        clip,
        target,
        labels,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        total_lossCausal = 0.
        for t in tqdm(reversed(range(0, self.num_timesteps)), 
                      desc='conditional sampling loop time step', total=self.num_timesteps):
            img, judges, lossCausal = self.p_sample_infill(granger=granger, x=img, history=history, t=t, clip=clip, clip_denoised=clip_denoised, target=target,
                                       partial_mask=partial_mask, model_kwargs=model_kwargs)
            lossCausal.backward()
            lossCausal = lossCausal / self.num_timesteps
            total_lossCausal += lossCausal.item()

        return img, judges, total_lossCausal
    
    def p_sample_infill(
        self,
        granger,
        x,
        history,
        target,
        t: int,
        clip,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None
    ):
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, _, judges, loss_causal= \
            self.p_mean_variance(granger=granger, x=x, history=history, t=batched_times, clip=clip, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        sigma = (0.5 * model_log_variance).exp()
        pred_img = model_mean + sigma * noise
        
        return pred_img, judges, loss_causal

if __name__ == '__main__':
    pass

