from abc import ABC

import torch
import torch.nn.functional as F

from indextts.s2mel.modules.diffusion_transformer import DiT
from indextts.s2mel.modules.commons import sequence_mask

from tqdm import tqdm

class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.sigma_min = 1e-6

        self.estimator = None

        self.in_channels = args.DiT.in_channels

        self.criterion = torch.nn.MSELoss() if args.reg_loss_type == "l2" else torch.nn.L1Loss()

        if hasattr(args.DiT, 'zero_prompt_speech_token'):
            self.zero_prompt_speech_token = args.DiT.zero_prompt_speech_token
        else:
            self.zero_prompt_speech_token = False

    @torch.inference_mode()
    def inference(self, mu, x_lens, prompt, style, f0, n_timesteps, temperature=1.0, inference_cfg_rate=0.5,
                  technique=None, f0_stats=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): semantic info of reference audio and altered audio
                shape: (batch_size, mel_timesteps(795+1069), 512)
            x_lens (torch.Tensor): mel frames output
                shape: (batch_size, mel_timesteps)
            prompt (torch.Tensor): reference mel
                shape: (batch_size, 80, 795)
            style (torch.Tensor): reference global style
                shape: (batch_size, 192)
            f0: None
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            technique / f0_stats: E1-T 歌声条件（默认 None=原路径）。
                **位置兼容**：只追加在签名尾部——上游 infer_v2_5.py 与
                webui_app/services/svc_core.py 都按位置传参（第 6 位是
                n_timesteps），新参数不得插在中间。

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, 80, mel_timesteps)
        """
        B, T = mu.size(0), mu.size(1)
        z = torch.randn([B, self.in_channels, T], device=mu.device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        # t_span = t_span + (-1) * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
        return self.solve_euler(z, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate,
                                technique=technique, f0_stats=f0_stats)

    def solve_euler(self, x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate=0.5,
                    technique=None, f0_stats=None):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): semantic info of reference audio and altered audio
                shape: (batch_size, mel_timesteps(795+1069), 512)
            x_lens (torch.Tensor): mel frames output
                shape: (batch_size, mel_timesteps)
            prompt (torch.Tensor): reference mel
                shape: (batch_size, 80, 795)
            style (torch.Tensor): reference global style
                shape: (batch_size, 192)
            technique / f0_stats: E1-T 歌声条件（默认 None=原路径）
        """
        t, _, _ = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []
        # apply prompt
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        if self.zero_prompt_speech_token:
            mu[..., :prompt_len] = 0
        for step in tqdm(range(1, len(t_span))):
            dt = t_span[step] - t_span[step - 1]
            if inference_cfg_rate > 0:
                # Stack original and CFG (null) inputs for batched processing
                stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0)
                stacked_style = torch.cat([style, torch.zeros_like(style)], dim=0)
                stacked_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)
                stacked_x = torch.cat([x, x], dim=0)
                stacked_t = torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0)
                # E1-T：null 分支**旁路** conditioner（cond_mask 后半为 0 →
                # style_t=纯 style 广播），与既有零分支（style/mu 置零）同族。
                # 不做输入置零——置零得到 style+α·MLP([0;·;style])≠style，
                # 与预训练 null 行为不等价（先进性调研 Q3 结论）。
                stacked_f0 = torch.cat([f0, f0], dim=0) if f0 is not None else None
                stacked_tech = (torch.cat([technique, technique], dim=0)
                                if technique is not None else None)
                stacked_stats = None
                if f0_stats is not None:
                    stacked_stats = (torch.cat([f0_stats[0], f0_stats[0]], dim=0),
                                     torch.cat([f0_stats[1], f0_stats[1]], dim=0))
                cond_mask = None
                if f0 is not None:
                    cond_mask = torch.cat([torch.ones(f0.size(0), device=f0.device),
                                           torch.zeros(f0.size(0), device=f0.device)], dim=0)

                # Perform a single forward pass for both original and CFG inputs
                stacked_dphi_dt = self.estimator(
                    stacked_x, stacked_prompt_x, x_lens, stacked_t, stacked_style, stacked_mu,
                    f0=stacked_f0, technique=stacked_tech, f0_stats=stacked_stats,
                    cond_mask=cond_mask,
                )

                # Split the output back into the original and CFG components
                dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)

                # Apply CFG formula
                dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
            else:
                dphi_dt = self.estimator(x, prompt_x, x_lens, t.unsqueeze(0), style, mu,
                                         f0=f0, technique=technique, f0_stats=f0_stats)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
            x[:, :, :prompt_len] = 0

        return sol[-1]

    def forward(self, x1, x_lens, prompt_lens, mu, style,
                f0=None, technique=None, f0_stats=None):
        """Computes diffusion loss

        Args:
            mu (torch.Tensor): semantic info of reference audio and altered audio
                shape: (batch_size, mel_timesteps(795+1069), 512)
            x1: mel
            x_lens (torch.Tensor): mel frames output
                shape: (batch_size, mel_timesteps)
            prompt (torch.Tensor): reference mel
                shape: (batch_size, 80, 795)
            style (torch.Tensor): reference global style
                shape: (batch_size, 192)
            f0 / technique / f0_stats: E1-T 训练条件（默认 None=原路径；
                逐帧长度与 mu 对齐，f0 为 Hz、清音=0，technique 为 0..6 整数）

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = x1.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=x1.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        prompt = torch.zeros_like(x1)
        for bib in range(b):
            prompt[bib, :, :prompt_lens[bib]] = x1[bib, :, :prompt_lens[bib]]
            # range covered by prompt are set to 0
            y[bib, :, :prompt_lens[bib]] = 0
            if self.zero_prompt_speech_token:
                mu[bib, :, :prompt_lens[bib]] = 0

        estimator_out = self.estimator(y, prompt, x_lens, t.squeeze(1).squeeze(1), style, mu, prompt_lens,
                                       f0=f0, technique=technique, f0_stats=f0_stats)
        loss = 0
        for bib in range(b):
            loss += self.criterion(estimator_out[bib, :, prompt_lens[bib]:x_lens[bib]], u[bib, :, prompt_lens[bib]:x_lens[bib]])
        loss /= b

        return loss, estimator_out + (1 - self.sigma_min) * z



class CFM(BASECFM):
    def __init__(self, args):
        super().__init__(
            args
        )
        if args.dit_type == "DiT":
            self.estimator = DiT(args)
        else:
            raise NotImplementedError(f"Unknown diffusion type {args.dit_type}")

    def enable_torch_compile(self):
        """Enable torch.compile optimization for the estimator model.
        
        This method applies torch.compile to the estimator (DiT model) for significant
        performance improvements during inference. It also configures distributed
        training optimizations if applicable.
        """
        if torch.distributed.is_initialized():
            torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.estimator = torch.compile(
            self.estimator, 
            fullgraph=True,
            dynamic=True,
        )
