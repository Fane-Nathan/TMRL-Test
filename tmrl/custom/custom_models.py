# === Trackmania =======================================================================================================


# standard library imports

# third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from math import floor, sqrt
from torch.nn import Conv2d, Module, ModuleList
# import torchvision

# local imports
from tmrl.util import prod
from tmrl.actor import TorchActorModule
import tmrl.config.config_constants as cfg


# SUPPORTED ============================================================================================================


# Spinup MLP: =======================================================
# Adapted from the SAC implementation of OpenAI Spinup


def combined_shape(length, shape=None):
    if shape is None:
        return (length, )
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPSILON = 1e-7


class SquashedGaussianMLPActor(TorchActorModule):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__(observation_space, action_space)
        try:
            dim_obs = sum(prod(s for s in space.shape) for space in observation_space)
            self.tuple_obs = True
        except TypeError:
            dim_obs = prod(observation_space.shape)
            self.tuple_obs = False
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]
        self.net = mlp([dim_obs] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], dim_act)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], dim_act)
        self.act_limit = act_limit

    def forward(self, obs, test=False, with_logprob=True):
        x = torch.cat(obs, -1) if self.tuple_obs else torch.flatten(obs, start_dim=1)
        net_out = self.net(x)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if test:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        # pi_action = pi_action.squeeze()

        return pi_action, logp_pi

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            res = a.squeeze().cpu().numpy()
            if not len(res.shape):
                res = np.expand_dims(res, 0)
            return res


class MLPQFunction(nn.Module):
    def __init__(self, obs_space, act_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        try:
            obs_dim = sum(prod(s for s in space.shape) for space in obs_space)
            self.tuple_obs = True
        except TypeError:
            obs_dim = prod(obs_space.shape)
            self.tuple_obs = False
        act_dim = act_space.shape[0]
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        x = torch.cat((*obs, act), -1) if self.tuple_obs else torch.cat((torch.flatten(obs, start_dim=1), act), -1)
        q = self.q(x)
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.  # FIXME: understand this


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()

        # obs_dim = observation_space.shape[0]
        # act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.actor = SquashedGaussianMLPActor(observation_space, action_space, hidden_sizes, activation)
        self.q1 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)
        self.q2 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            res = a.squeeze().cpu().numpy()
            if not len(res.shape):
                res = np.expand_dims(res, 0)
            return res


# REDQ MLP: =====================================================


class REDQMLPActorCritic(nn.Module):
    def __init__(self,
                 observation_space,
                 action_space,
                 hidden_sizes=(256, 256),
                 activation=nn.ReLU,
                 n=10):
        super().__init__()

        # obs_dim = observation_space.shape[0]
        # act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.actor = SquashedGaussianMLPActor(observation_space, action_space, hidden_sizes, activation)
        self.n = n
        self.qs = ModuleList([MLPQFunction(observation_space, action_space, hidden_sizes, activation) for _ in range(self.n)])

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()


# CNNs: ================================================================================================================

# EfficientNet =========================================================================================================

# EfficientNetV2 implementation adapted from https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py
# We use the EfficientNetV2 structure for image features and we merge the TM2020 float features to linear layers


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, _make_divisible(inp // reduction, 8)),
            SiLU(),
            nn.Linear(_make_divisible(inp // reduction, 8), oup),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EffNetV2(nn.Module):
    def __init__(self, cfgs, nb_channels_in=3, dim_output=1, width_mult=1.):
        super(EffNetV2, self).__init__()
        self.cfgs = cfgs

        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(nb_channels_in, input_channel, 2)]
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, dim_output)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


def effnetv2_s(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 24, 2, 1, 0],
        [4, 48, 4, 2, 0],
        [4, 64, 4, 2, 0],
        [4, 128, 6, 2, 1],
        [6, 160, 9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_m(**kwargs):
    """
    Constructs a EfficientNetV2-M model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 24, 3, 1, 0],
        [4, 48, 5, 2, 0],
        [4, 80, 5, 2, 0],
        [4, 160, 7, 2, 1],
        [6, 176, 14, 1, 1],
        [6, 304, 18, 2, 1],
        [6, 512, 5, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_l(**kwargs):
    """
    Constructs a EfficientNetV2-L model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 32, 4, 1, 0],
        [4, 64, 7, 2, 0],
        [4, 96, 7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640, 7, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_xl(**kwargs):
    """
    Constructs a EfficientNetV2-XL model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 32, 4, 1, 0],
        [4, 64, 8, 2, 0],
        [4, 96, 8, 2, 0],
        [4, 192, 16, 2, 1],
        [6, 256, 24, 1, 1],
        [6, 512, 32, 2, 1],
        [6, 640, 8, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


class SquashedGaussianEffNetActor(TorchActorModule):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]

        self.cnn = effnetv2_s(nb_channels_in=4, dim_output=247, width_mult=1.).float()
        self.net = mlp([256, 256], [nn.ReLU, nn.ReLU])
        self.mu_layer = nn.Linear(256, dim_act)
        self.log_std_layer = nn.Linear(256, dim_act)
        self.act_limit = act_limit

    def forward(self, obs, test=False, with_logprob=True):
        imgs_tensor = obs[3].float()
        float_tensors = (obs[0], obs[1], obs[2], *obs[4:])
        float_tensor = torch.cat(float_tensors, -1).float()
        cnn_out = self.cnn(imgs_tensor)
        mlp_in = torch.cat((cnn_out, float_tensor), -1)
        net_out = self.net(mlp_in)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if test:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        # pi_action = pi_action.squeeze()

        return pi_action, logp_pi

    def act(self, obs, test=False):
        import sys
        size = sys.getsizeof(obs)
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            return a.squeeze().cpu().numpy()


class EffNetQFunction(nn.Module):
    def __init__(self, obs_space, act_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        obs_dim = sum(prod(s for s in space.shape) for space in obs_space)
        act_dim = act_space.shape[0]
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        x = torch.cat((*obs, act), -1)
        q = self.q(x)
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class EffNetActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()

        # obs_dim = observation_space.shape[0]
        # act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.actor = SquashedGaussianMLPActor(observation_space, action_space, hidden_sizes, activation)
        self.q1 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)
        self.q2 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()


# Vanilla CNN FOR GRAYSCALE IMAGES: ====================================================================================


def num_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def conv2d_out_dims(conv_layer, h_in, w_in):
    h_out = floor((h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) / conv_layer.stride[0] + 1)
    w_out = floor((w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) / conv_layer.stride[1] + 1)
    return h_out, w_out


class VanillaCNN(Module):
    def __init__(self, q_net):
        super(VanillaCNN, self).__init__()
        self.q_net = q_net
        self.h_out, self.w_out = cfg.IMG_HEIGHT, cfg.IMG_WIDTH
        hist = cfg.IMG_HIST_LEN

        self.conv1 = Conv2d(hist, 64, 8, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv1, self.h_out, self.w_out)
        self.conv2 = Conv2d(64, 64, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv2, self.h_out, self.w_out)
        self.conv3 = Conv2d(64, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv3, self.h_out, self.w_out)
        self.conv4 = Conv2d(128, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv4, self.h_out, self.w_out)
        self.out_channels = self.conv4.out_channels
        self.flat_features = self.out_channels * self.h_out * self.w_out
        self.mlp_input_features = self.flat_features + 12 if self.q_net else self.flat_features + 9
        self.mlp_layers = [256, 256, 1] if self.q_net else [256, 256]
        self.mlp = mlp([self.mlp_input_features] + self.mlp_layers, nn.ReLU)

    def forward(self, x):
        if self.q_net:
            speed, gear, rpm, images, act1, act2, act = x
        else:
            speed, gear, rpm, images, act1, act2 = x

        x = F.relu(self.conv1(images))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        flat_features = num_flat_features(x)
        assert flat_features == self.flat_features, f"x.shape:{x.shape}, flat_features:{flat_features}, self.out_channels:{self.out_channels}, self.h_out:{self.h_out}, self.w_out:{self.w_out}"
        x = x.view(-1, flat_features)
        if self.q_net:
            x = torch.cat((speed, gear, rpm, x, act1, act2, act), -1)
        else:
            x = torch.cat((speed, gear, rpm, x, act1, act2), -1)
        x = self.mlp(x)
        return x


class SquashedGaussianVanillaCNNActor(TorchActorModule):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]
        self.net = VanillaCNN(q_net=False)
        self.mu_layer = nn.Linear(256, dim_act)
        self.log_std_layer = nn.Linear(256, dim_act)
        self.act_limit = act_limit

    def forward(self, obs, test=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if test:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            # NB: this is from Spinup:
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)  # FIXME: this formula is mathematically wrong, no idea why it seems to work
            # Whereas SB3 does this:
            # logp_pi -= torch.sum(torch.log(1 - torch.tanh(pi_action) ** 2 + EPSILON), dim=1)  # TODO: double check
            # # log_prob -= th.sum(th.log(1 - actions**2 + self.epsilon), dim=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        # pi_action = pi_action.squeeze()

        return pi_action, logp_pi

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            return a.squeeze().cpu().numpy()


class VanillaCNNQFunction(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.net = VanillaCNN(q_net=True)

    def forward(self, obs, act):
        x = (*obs, act)
        q = self.net(x)
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class VanillaCNNActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()

        # build policy and value functions
        self.actor = SquashedGaussianVanillaCNNActor(observation_space, action_space)
        self.q1 = VanillaCNNQFunction(observation_space, action_space)
        self.q2 = VanillaCNNQFunction(observation_space, action_space)

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()


# Vanilla CNN FOR COLOR IMAGES: ========================================================================================

def remove_colors(images):
    """
    We remove colors so that we can simply use the same structure as the grayscale model.

    The "color" default pipeline is mostly here for support, as our model effectively gets rid of 2 channels out of 3.
    If you actually want to use colors, do not use the default pipeline.
    Instead, you need to code a custom model that doesn't get rid of them.
    """
    images = images[:, :, :, :, 0]
    return images


class SquashedGaussianVanillaColorCNNActor(SquashedGaussianVanillaCNNActor):
    def forward(self, obs, test=False, with_logprob=True):
        speed, gear, rpm, images, act1, act2 = obs
        images = remove_colors(images)
        obs = (speed, gear, rpm, images, act1, act2)
        return super().forward(obs, test=False, with_logprob=True)


class VanillaColorCNNQFunction(VanillaCNNQFunction):
    def forward(self, obs, act):
        speed, gear, rpm, images, act1, act2 = obs
        images = remove_colors(images)
        obs = (speed, gear, rpm, images, act1, act2)
        return super().forward(obs, act)


class VanillaColorCNNActorCritic(VanillaCNNActorCritic):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

        # build policy and value functions
        self.actor = SquashedGaussianVanillaColorCNNActor(observation_space, action_space)
        self.q1 = VanillaColorCNNQFunction(observation_space, action_space)
        self.q2 = VanillaColorCNNQFunction(observation_space, action_space)


# HybridNanoEffNet: VRAM-Efficient CNN with SE Attention ===============================================================
# Uses existing EffNetV2/MBConv blocks with a lightweight "Nano" configuration
# Benefits: ~90% smaller than ResNet18, SE attention for important features, SiLU for smooth RL gradients

# Nano config: [expansion_ratio, channels, num_layers, stride, use_se]
NANO_EFFNET_CFG = [
    [1, 24, 2, 1, 0],   # MBConv1: don't expand, keep 24 ch
    [4, 32, 2, 2, 0],   # MBConv6: expand 4x, downsample to 32 ch
    [4, 48, 2, 2, 1],   # MBConv6: expand 4x, downsample to 48 ch, USE SE (Attention)
    [4, 64, 3, 2, 1],   # MBConv6: expand 4x, downsample to 64 ch, USE SE (Attention)
]


class HybridNanoEffNetActor(TorchActorModule):
    """
    Optimized Actor: Uses EffNetV2 (MBConv) for images + Gated Fusion for LIDAR.
    High Quality (SE Attention) + Low VRAM (Depthwise Convs).
    """
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]

        # --- 1. Smart Image Branch (EffNetV2) ---
        # We reuse the EffNetV2 class with our Nano config.
        # dim_output=128 means the CNN will output a vector of size 128.
        self.cnn = EffNetV2(cfgs=NANO_EFFNET_CFG, nb_channels_in=4, dim_output=128)

        # --- 2. LIDAR/Telemetry Branch ---
        # Speed(1) + Gear(1) + RPM(1) + Lidar(19) + PastActions(2*dim_act)
        self.dim_float_input = 1 + 1 + 1 + 19 + dim_act + dim_act

        # Float Encoder
        self.float_mlp = mlp([self.dim_float_input, 64, 64], nn.ReLU, nn.ReLU)

        # --- 3. Gated Fusion with LayerNorm ---
        # Visual features are 128, float features are 64.
        self.fusion_norm = nn.LayerNorm(128 + 64)
        self.net = mlp([128 + 64, 256, 256], SiLU, SiLU)  # SiLU (Swish) is better for RL

        self.mu_layer = nn.Linear(256, dim_act)
        self.log_std_layer = nn.Linear(256, dim_act)
        self.act_limit = act_limit

    def forward(self, obs, test=False, with_logprob=True):
        # Unpack observation tuple
        speed, gear, rpm, images, lidar, act1, act2 = obs

        # 1. Images -> EffNet Features
        img_embed = self.cnn(images.float())  # Output: (Batch, 128)

        # 2. Floats -> MLP Features
        floats = torch.cat((speed, gear, rpm, lidar, act1, act2), dim=-1)
        float_embed = self.float_mlp(floats)  # Output: (Batch, 64)

        # 3. Fusion with LayerNorm for stable training
        combined = torch.cat((img_embed, float_embed), dim=-1)
        combined = self.fusion_norm(combined)

        # 4. Policy Head
        net_out = self.net(combined)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # SAC Logic
        pi_distribution = Normal(mu, std)
        if test:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action, logp_pi

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            return a.squeeze().cpu().numpy()


class HybridNanoEffNetQFunction(nn.Module):
    """Q-function using same EffNet architecture for observations."""
    def __init__(self, observation_space, action_space):
        super().__init__()
        dim_act = action_space.shape[0]

        # Re-use same architecture structure
        self.cnn = EffNetV2(cfgs=NANO_EFFNET_CFG, nb_channels_in=4, dim_output=128)
        self.dim_float_input = 1 + 1 + 1 + 19 + dim_act + dim_act
        self.float_mlp = mlp([self.dim_float_input, 64, 64], nn.ReLU, nn.ReLU)

        # Q-Function takes action as input too
        self.fusion_norm = nn.LayerNorm(128 + 64 + dim_act)
        self.q = mlp([128 + 64 + dim_act, 256, 256, 1], SiLU)

    def forward(self, obs, act):
        speed, gear, rpm, images, lidar, act1, act2 = obs

        img_embed = self.cnn(images.float())

        floats = torch.cat((speed, gear, rpm, lidar, act1, act2), dim=-1)
        float_embed = self.float_mlp(floats)

        # Fuse Obs + Action
        combined = torch.cat((img_embed, float_embed, act), dim=-1)
        combined = self.fusion_norm(combined)

        q = self.q(combined)
        return torch.squeeze(q, -1)


class HybridNanoEffNetActorCritic(nn.Module):
    """Standard SAC Actor-Critic with HybridNanoEffNet."""
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.actor = HybridNanoEffNetActor(observation_space, action_space)
        self.q1 = HybridNanoEffNetQFunction(observation_space, action_space)
        self.q2 = HybridNanoEffNetQFunction(observation_space, action_space)

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()


class REDQHybridNanoEffNetActorCritic(nn.Module):
    """REDQ Actor-Critic with HybridNanoEffNet and N Q-networks."""
    def __init__(self, observation_space, action_space, n=10):
        super().__init__()
        self.actor = HybridNanoEffNetActor(observation_space, action_space)
        self.n = n
        self.qs = ModuleList([HybridNanoEffNetQFunction(observation_space, action_space) for _ in range(self.n)])

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()


# ========== VRAM-OPTIMIZED SHARED BACKBONE ARCHITECTURE ==========
# For 4GB VRAM GPUs: Share ONE CNN backbone across actor + all critics
# Reduces CNN count from 5 (actor + 2 critics + 2 targets) to 1


class SharedBackboneActorHead(nn.Module):
    """Actor head that takes pre-computed CNN features."""
    def __init__(self, action_space, feature_dim=192):  # 128 CNN + 64 float
        super().__init__()
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]
        
        self.net = mlp([feature_dim, 256, 256], SiLU, SiLU)
        self.mu_layer = nn.Linear(256, dim_act)
        self.log_std_layer = nn.Linear(256, dim_act)
        self.act_limit = act_limit

    def forward(self, features, test=False, with_logprob=True):
        net_out = self.net(features)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if test:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action, logp_pi


class SharedBackboneQHead(nn.Module):
    """Q-function head that takes pre-computed CNN features + action."""
    def __init__(self, action_space, feature_dim=192):  # 128 CNN + 64 float
        super().__init__()
        dim_act = action_space.shape[0]
        self.fusion_norm = nn.LayerNorm(feature_dim + dim_act)
        self.q = mlp([feature_dim + dim_act, 256, 256, 1], SiLU)

    def forward(self, features, act):
        combined = torch.cat((features, act), dim=-1)
        combined = self.fusion_norm(combined)
        q = self.q(combined)
        return torch.squeeze(q, -1)


class SharedBackboneEncoder(nn.Module):
    """Shared CNN + Float encoder backbone."""
    def __init__(self, observation_space, action_space):
        super().__init__()
        dim_act = action_space.shape[0]
        
        # Single shared CNN
        self.cnn = EffNetV2(cfgs=NANO_EFFNET_CFG, nb_channels_in=4, dim_output=128)
        
        # Float encoder: Speed(1) + Gear(1) + RPM(1) + Lidar(19) + PastActions(2*dim_act)
        self.dim_float_input = 1 + 1 + 1 + 19 + dim_act + dim_act
        self.float_mlp = mlp([self.dim_float_input, 64, 64], nn.ReLU, nn.ReLU)
        
        # Fusion normalization
        self.fusion_norm = nn.LayerNorm(128 + 64)

    def forward(self, obs):
        speed, gear, rpm, images, lidar, act1, act2 = obs
        
        # CNN features (computed ONCE)
        img_embed = self.cnn(images.float())
        
        # Float features
        floats = torch.cat((speed, gear, rpm, lidar, act1, act2), dim=-1)
        float_embed = self.float_mlp(floats)
        
        # Fused features
        combined = torch.cat((img_embed, float_embed), dim=-1)
        combined = self.fusion_norm(combined)
        return combined


class SharedBackboneHybridActor(TorchActorModule):
    """
    Standalone Actor for worker inference.
    Has its own encoder (weights synced from trainer).
    """
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]

        # Own encoder for standalone inference
        self.cnn = EffNetV2(cfgs=NANO_EFFNET_CFG, nb_channels_in=4, dim_output=128)
        self.dim_float_input = 1 + 1 + 1 + 19 + dim_act + dim_act
        self.float_mlp = mlp([self.dim_float_input, 64, 64], nn.ReLU, nn.ReLU)
        self.fusion_norm = nn.LayerNorm(128 + 64)

        # Policy head
        self.net = mlp([128 + 64, 256, 256], SiLU, SiLU)
        self.mu_layer = nn.Linear(256, dim_act)
        self.log_std_layer = nn.Linear(256, dim_act)
        self.act_limit = act_limit

    def forward(self, obs, test=False, with_logprob=True):
        speed, gear, rpm, images, lidar, act1, act2 = obs

        # Encode
        img_embed = self.cnn(images.float())
        floats = torch.cat((speed, gear, rpm, lidar, act1, act2), dim=-1)
        float_embed = self.float_mlp(floats)
        combined = torch.cat((img_embed, float_embed), dim=-1)
        features = self.fusion_norm(combined)

        # Policy
        net_out = self.net(features)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if test:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action, logp_pi

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            return a.squeeze().cpu().numpy()


class SharedBackboneHybridActorCritic(nn.Module):
    """
    VRAM-Optimized Actor-Critic for 4GB GPUs.
    
    Key optimization: ONE shared CNN backbone for actor + all critics.
    The actor has the same structure as SharedBackboneHybridActor so weights sync properly.
    Internally, for training, we extract features once and reuse them.
    """
    def __init__(self, observation_space, action_space, n=2):
        super().__init__()
        self.n = n
        
        # Actor has its own encoder (same structure as standalone actor for weight sync)
        self.actor = SharedBackboneHybridActor(observation_space, action_space)
        
        # Reference to actor's encoder for shared feature extraction
        self._actor_cnn = self.actor.cnn
        self._actor_float_mlp = self.actor.float_mlp
        self._actor_fusion_norm = self.actor.fusion_norm
        
        # N Q-heads (lightweight MLPs only)
        self.qs = ModuleList([SharedBackboneQHead(action_space, feature_dim=192) for _ in range(n)])

    def forward_features(self, obs):
        """Extract features using actor's encoder (shared for training)."""
        speed, gear, rpm, images, lidar, act1, act2 = obs
        
        img_embed = self._actor_cnn(images.float())
        floats = torch.cat((speed, gear, rpm, lidar, act1, act2), dim=-1)
        float_embed = self._actor_float_mlp(floats)
        combined = torch.cat((img_embed, float_embed), dim=-1)
        features = self._actor_fusion_norm(combined)
        return features

    def actor_from_features(self, features, test=False, with_logprob=True):
        """Compute action from pre-extracted features."""
        net_out = self.actor.net(features)
        mu = self.actor.mu_layer(net_out)
        log_std = self.actor.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if test:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.actor.act_limit * pi_action
        return pi_action, logp_pi

    def q_from_features(self, features, act, q_idx=None):
        """Compute Q-values from pre-extracted features."""
        if q_idx is not None:
            return self.qs[q_idx](features, act)
        return [q(features, act) for q in self.qs]

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()


# ============== DroQ (Dropout Q-functions) ==============

class DroQQHead(nn.Module):
    """
    Q-head with Dropout and LayerNorm for DroQ.
    Dropout provides ensemble-like diversity with only 2 Q-networks.
    LayerNorm stabilizes training with high UTD ratios.
    """
    def __init__(self, action_space, feature_dim=192, dropout_rate=0.01):
        super().__init__()
        dim_act = action_space.shape[0]
        
        self.q = nn.Sequential(
            nn.Linear(feature_dim + dim_act, 384),
            nn.LayerNorm(384),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(384, 384),
            nn.LayerNorm(384),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(384, 1)
        )

    def forward(self, features, act):
        """
        Args:
            features: Pre-extracted encoder features (B, feature_dim)
            act: Actions (B, dim_act)
        Returns:
            Q-values (B,)
        """
        combined = torch.cat((features, act), dim=-1)
        q = self.q(combined)
        return torch.squeeze(q, -1)


class DroQHybridActorCritic(nn.Module):
    """
    DroQ Actor-Critic for maximum sample efficiency.
    
    Key features:
    - Shared CNN backbone (VRAM efficient)
    - Only 2 Q-networks with Dropout+LayerNorm
    - Supports high UTD (Update-to-Data) ratios (20+)
    - Dropout provides ensemble-like diversity for uncertainty estimation
    """
    def __init__(self, observation_space, action_space, dropout_rate=0.01):
        super().__init__()
        self.n = 2  # DroQ uses exactly 2 Q-networks
        
        # Actor has its own encoder (same structure as standalone for weight sync)
        self.actor = SharedBackboneHybridActor(observation_space, action_space)
        
        # Reference to actor's encoder for shared feature extraction
        self._actor_cnn = self.actor.cnn
        self._actor_float_mlp = self.actor.float_mlp
        self._actor_fusion_norm = self.actor.fusion_norm
        
        # 2 DroQ Q-heads with Dropout + LayerNorm
        self.qs = ModuleList([
            DroQQHead(action_space, feature_dim=192, dropout_rate=dropout_rate)
            for _ in range(2)
        ])

    def forward_features(self, obs):
        """Extract features using actor's encoder (shared for training)."""
        speed, gear, rpm, images, lidar, act1, act2 = obs
        
        img_embed = self._actor_cnn(images.float())
        floats = torch.cat((speed, gear, rpm, lidar, act1, act2), dim=-1)
        float_embed = self._actor_float_mlp(floats)
        combined = torch.cat((img_embed, float_embed), dim=-1)
        features = self._actor_fusion_norm(combined)
        return features

    def actor_from_features(self, features, test=False, with_logprob=True):
        """Compute action from pre-extracted features."""
        net_out = self.actor.net(features)
        mu = self.actor.mu_layer(net_out)
        log_std = self.actor.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if test:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.actor.act_limit * pi_action
        return pi_action, logp_pi

    def q_from_features(self, features, act, q_idx=None):
        """Compute Q-values from pre-extracted features."""
        if q_idx is not None:
            return self.qs[q_idx](features, act)
        return [q(features, act) for q in self.qs]

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()


# ============== Context-Based Meta-Learning + Fusion Gate ==============

CONTEXT_WINDOW_SIZE = 16  # K: number of recent transitions for context
CONTEXT_INPUT_DIM = 24    # speed(1) + lidar(19) + action(3) + reward(1)
CONTEXT_Z_DIM = 64        # latent context dimension (widened for richer context)
FUSED_DIM = 128           # output dimension of fusion gate (widened for capacity)


class FusionGate(nn.Module):
    """
    Learnable Fusion Gate for CNN + Float sensor fusion.
    Replaces naive concatenation with a learned gate that dynamically
    balances CNN vs Float contributions per feature dimension.
    
    gate ∈ [0,1]^fused_dim:
        output = gate * img_proj + (1 - gate) * float_proj
    """
    def __init__(self, img_dim=128, float_dim=64, fused_dim=FUSED_DIM):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, fused_dim)
        self.float_proj = nn.Linear(float_dim, fused_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(img_dim + float_dim, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(fused_dim)

    def forward(self, img_embed, float_embed):
        img_proj = self.img_proj(img_embed)
        float_proj = self.float_proj(float_embed)
        gate = self.gate_net(torch.cat((img_embed, float_embed), dim=-1))
        fused = gate * img_proj + (1.0 - gate) * float_proj
        return self.norm(fused)


class ContextEncoder(nn.Module):
    """
    Transformer-Enhanced Context Encoder for rapid adaptation.
    
    Training pipeline (full sequence):
      Context (B,K,24) → Temporal Deltas (B,K,48) → +Sinusoidal PE
      → Transformer Encoder Layer (self-attn + FFN)
      → GRU (sequential aggregation) → all outputs (B,K,z_dim)
      → Cross-Attention (fused obs queries history)
      → Learned Compression (4 query tokens pool sequence)
      → z (B, z_dim)
    
    Online pipeline (single-step for real-time inference):
      context_step → delta → GRU step → z (lightweight)
    """
    REWARD_HORIZONS = [1, 3, 5]

    def __init__(self, input_dim=CONTEXT_INPUT_DIM, z_dim=CONTEXT_Z_DIM,
                 fused_dim=FUSED_DIM, max_len=CONTEXT_WINDOW_SIZE):
        super().__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        enriched_dim = input_dim * 2  # raw + deltas = 48

        # === 1. Sinusoidal Positional Encoding ===
        pe = torch.zeros(max_len, enriched_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, enriched_dim, 2).float() * -(np.log(10000.0) / enriched_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_enc', pe.unsqueeze(0))  # (1, max_len, 48)

        # === 2. Dual-Stream Transformers ===
        # State stream: speed(1) + lidar(19) + their deltas = 40 dims
        # Action stream: action(3) + reward(1) + their deltas = 8 dims
        self.state_dim = 40  # indices 0:20 and 24:44
        self.action_dim = 8  # indices 20:24 and 44:48
        self.state_proj = nn.Linear(self.state_dim, enriched_dim)
        self.action_proj = nn.Linear(self.action_dim, enriched_dim)

        state_layer = nn.TransformerEncoderLayer(
            d_model=enriched_dim, nhead=4, dim_feedforward=128,
            dropout=0.0, batch_first=True, norm_first=True
        )
        self.state_transformer = nn.TransformerEncoder(state_layer, num_layers=2)

        action_layer = nn.TransformerEncoderLayer(
            d_model=enriched_dim, nhead=4, dim_feedforward=128,
            dropout=0.0, batch_first=True, norm_first=True
        )
        self.action_transformer = nn.TransformerEncoder(action_layer, num_layers=2)

        self.stream_gate = nn.Sequential(
            nn.Linear(enriched_dim * 2, enriched_dim),
            nn.Sigmoid()
        )

        # === 3. GRU for sequential aggregation + online inference ===
        self.gru = nn.GRU(enriched_dim, z_dim, num_layers=1, batch_first=True)

        # === 4. Cross-Attention: fused obs queries GRU output ===
        self.obs_proj = nn.Linear(fused_dim, z_dim)  # project fused (128) → z_dim (64)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=z_dim, num_heads=4, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(z_dim)

        # === 5. Learned Context Compression (4 query tokens) ===
        self.pool_queries = nn.Parameter(torch.randn(1, 4, z_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=z_dim, num_heads=4, batch_first=True
        )
        self.compress = nn.Sequential(
            nn.Linear(4 * z_dim, z_dim),
            nn.LayerNorm(z_dim)
        )

        # Output normalization
        self.norm = nn.LayerNorm(z_dim)

        # Multi-step reward prediction heads (1/3/5-step horizons)
        self.reward_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(z_dim, z_dim),
                nn.ReLU(),
                nn.Linear(z_dim, 1)
            )
            for _ in self.REWARD_HORIZONS
        ])

        # Online inference state
        self._prev_step = None

    def _compute_deltas(self, context_seq):
        """(B, K, 24) → (B, K, 48) = [raw, Δraw]"""
        deltas = context_seq[:, 1:, :] - context_seq[:, :-1, :]
        deltas = F.pad(deltas, (0, 0, 1, 0))
        return torch.cat([context_seq, deltas], dim=-1)

    def predict_reward(self, z):
        """Multi-step reward prediction. Returns list of (B,) tensors."""
        return [head(z).squeeze(-1) for head in self.reward_predictors]

    def forward(self, context_seq, fused_obs=None):
        """
        Full training-time forward pass.
        Args:
            context_seq: (B, K, 24)
            fused_obs: (B, fused_dim) current observation features for cross-attention
        Returns:
            z: (B, z_dim) context vector
        """
        B = context_seq.shape[0]

        # 1. Temporal deltas + positional encoding
        enriched = self._compute_deltas(context_seq)  # (B, K, 48)
        K = enriched.shape[1]

        # 2. Dual-stream: split into state and action features
        state_feats = torch.cat([enriched[:, :, :20], enriched[:, :, 24:44]], dim=-1)  # (B, K, 40)
        action_feats = torch.cat([enriched[:, :, 20:24], enriched[:, :, 44:48]], dim=-1)  # (B, K, 8)

        state_seq = self.state_proj(state_feats) + self.pos_enc[:, :K, :]  # (B, K, 48)
        action_seq = self.action_proj(action_feats) + self.pos_enc[:, :K, :]  # (B, K, 48)

        state_out = self.state_transformer(state_seq)  # (B, K, 48)
        action_out = self.action_transformer(action_seq)  # (B, K, 48)

        # Gated merge of dual streams
        gate = self.stream_gate(torch.cat([state_out, action_out], dim=-1))  # (B, K, 48)
        transformed = gate * state_out + (1 - gate) * action_out  # (B, K, 48)

        # 3. GRU sequential processing
        self.gru.flatten_parameters()
        gru_out, h_n = self.gru(transformed)  # gru_out: (B, K, z_dim)

        # 4. Cross-attention: current obs queries history
        if fused_obs is not None:
            obs_query = self.obs_proj(fused_obs).unsqueeze(1)  # (B, 1, z_dim)
            cross_out, _ = self.cross_attn(obs_query, gru_out, gru_out)  # (B, 1, z_dim)
            cross_z = cross_out.squeeze(1)  # (B, z_dim)
        else:
            cross_z = h_n.squeeze(0)  # fallback to GRU last hidden

        # 5. Learned compression: 4 query tokens pool the sequence
        queries = self.pool_queries.expand(B, -1, -1)  # (B, 4, z_dim)
        pooled, _ = self.pool_attn(queries, gru_out, gru_out)  # (B, 4, z_dim)
        pool_z = self.compress(pooled.reshape(B, -1))  # (B, z_dim)

        # 6. Combine cross-attended + pooled context
        z = self.norm(cross_z + pool_z)
        return z

    def get_initial_hidden(self, batch_size=1, device='cpu'):
        """Get zero-initialized hidden state for inference."""
        return torch.zeros(1, batch_size, self.z_dim, device=device)

    def reset_online_state(self):
        """Reset online inference state (call at episode start)."""
        self._prev_step = None

    def step(self, context_step, hidden, fused_obs=None):
        """
        Single-step update for online inference.
        Applies dual-stream projections and merges before GRU step.
        """
        if self._prev_step is not None:
            delta = context_step - self._prev_step
        else:
            delta = torch.zeros_like(context_step)
        self._prev_step = context_step.detach()

        enriched = torch.cat([context_step, delta], dim=-1)  # (B, 48)

        # Apply projections and gated merge (K=1 version of training path)
        state_feats = torch.cat([enriched[:, :20], enriched[:, 24:44]], dim=-1)
        action_feats = torch.cat([enriched[:, 20:24], enriched[:, 44:48]], dim=-1)

        state_proj = self.state_proj(state_feats).unsqueeze(1)    # (B, 1, 48)
        action_proj = self.action_proj(action_feats).unsqueeze(1)  # (B, 1, 48)

        # Online, we skip the self-attention transformer for single step to save time,
        # but apply the projections and gating which are essential for feature alignment.
        gate = self.stream_gate(torch.cat([state_proj, action_proj], dim=-1))
        transformed = gate * state_proj + (1 - gate) * action_proj

        self.gru.flatten_parameters()
        gru_out, h_n = self.gru(transformed, hidden)  # gru_out: (B, 1, z_dim)

        # Apply Cross-Attention and Compression to match training 'z'
        if fused_obs is not None:
            obs_query = self.obs_proj(fused_obs).unsqueeze(1)
            cross_out, _ = self.cross_attn(obs_query, gru_out, gru_out)
            cross_z = cross_out.squeeze(1)
        else:
            cross_z = h_n.squeeze(0)

        queries = self.pool_queries.expand(context_step.shape[0], -1, -1)
        pooled, _ = self.pool_attn(queries, gru_out, gru_out)
        pool_z = self.compress(pooled.reshape(context_step.shape[0], -1))

        z = self.norm(cross_z + pool_z)
        return z, h_n



# ============== FiLM Conditioning ==============

FILM_HIDDEN = 512  # hidden dim for FiLM-modulated MLPs (widened from 384)
FILM_N_LAYERS = 2  # number of FiLM-modulated layers


class FiLMGenerator(nn.Module):
    """
    Shared FiLM generator: takes context z and produces (γ, β) pairs
    for each FiLM layer. Used by both actor and critic.
    
    This is the ONLY module updated during deployment-time adaptation.
    """
    def __init__(self, z_dim=CONTEXT_Z_DIM, hidden_dim=FILM_HIDDEN, n_layers=FILM_N_LAYERS):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        # z → compact intermediate → all γ/β pairs
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_layers * hidden_dim * 2)  # γ+β for each layer
        )
        # Initialize so FiLM starts as identity: γ=1, β=0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        # Set γ bias to 1 (so initial modulation is identity)
        with torch.no_grad():
            self.net[-1].bias[:n_layers * hidden_dim].fill_(1.0)

    def forward(self, z):
        """
        Args:
            z: (B, z_dim) context vector
        Returns:
            film_params: list of (γ, β) tuples, one per layer
                γ, β each have shape (B, hidden_dim)
        """
        raw = self.net(z)  # (B, n_layers * hidden_dim * 2)
        params = []
        for i in range(self.n_layers):
            start = i * self.hidden_dim * 2
            gamma = raw[:, start:start + self.hidden_dim]
            beta = raw[:, start + self.hidden_dim:start + self.hidden_dim * 2]
            params.append((gamma, beta))
        return params


class FiLMActorMLP(nn.Module):
    """
    Actor policy MLP with FiLM modulation + residual connections.
    Input: fused features (FUSED_DIM). Modulated by (γ, β) from FiLMGenerator.
    """
    def __init__(self, input_dim=FUSED_DIM, hidden_dim=FILM_HIDDEN, n_layers=FILM_N_LAYERS):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skip_projs = nn.ModuleList()  # residual projections
        for i in range(n_layers):
            in_d = input_dim if i == 0 else hidden_dim
            self.layers.append(nn.Linear(in_d, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            # Skip projection when dims don't match
            if in_d != hidden_dim:
                self.skip_projs.append(nn.Linear(in_d, hidden_dim, bias=False))
            else:
                self.skip_projs.append(nn.Identity())

    def forward(self, x, film_params):
        for i, (layer, norm, skip) in enumerate(zip(self.layers, self.norms, self.skip_projs)):
            residual = skip(x)
            x = layer(x)
            x = norm(x)
            x = F.silu(x)
            gamma, beta = film_params[i]
            x = gamma * x + beta
            x = x + residual  # residual connection
        return x


class FiLMQHead(nn.Module):
    """
    Q-network with FiLM modulation, residual connections, spectral norm,
    and dropout (DroQ-compatible).
    """
    def __init__(self, action_space, input_dim=FUSED_DIM, hidden_dim=FILM_HIDDEN,
                 n_layers=FILM_N_LAYERS, dropout_rate=0.01):
        super().__init__()
        dim_act = action_space.shape[0]
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        for i in range(n_layers):
            in_d = (input_dim + dim_act) if i == 0 else hidden_dim
            # Spectral normalization on critic linear layers
            self.layers.append(nn.utils.spectral_norm(nn.Linear(in_d, hidden_dim)))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout_rate))
            if in_d != hidden_dim:
                self.skip_projs.append(nn.Linear(in_d, hidden_dim, bias=False))
            else:
                self.skip_projs.append(nn.Identity())
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, features, act, film_params):
        x = torch.cat((features, act), dim=-1)
        for i, (layer, norm, drop, skip) in enumerate(
                zip(self.layers, self.norms, self.dropouts, self.skip_projs)):
            residual = skip(x)
            x = layer(x)
            x = norm(x)
            x = F.relu(x)
            x = drop(x)
            gamma, beta = film_params[i]
            x = gamma * x + beta
            x = x + residual  # residual connection
        q = self.out(x)
        return torch.squeeze(q, -1)



class ContextualSharedBackboneHybridActor(TorchActorModule):
    """
    Standalone Actor with Context Encoder + Fusion Gate for worker inference.
    Maintains a running GRU hidden state across steps within an episode.
    Weights are synced from the trainer's actor-critic.
    """
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]

        # Encoder components
        self.cnn = EffNetV2(cfgs=NANO_EFFNET_CFG, nb_channels_in=4, dim_output=128)
        self.dim_float_input = 1 + 1 + 1 + 19 + dim_act + dim_act
        self.float_mlp = mlp([self.dim_float_input, 64, 64], nn.ReLU, nn.ReLU)

        # Fusion Gate
        self.fusion_gate = FusionGate(img_dim=128, float_dim=64, fused_dim=FUSED_DIM)

        # Context Encoder
        self.context_encoder = ContextEncoder(input_dim=CONTEXT_INPUT_DIM, z_dim=CONTEXT_Z_DIM)
        self._gru_hidden = None  # persistent hidden state for online inference

        # FiLM Generator: z → (γ, β) pairs for each hidden layer
        self.film_generator = FiLMGenerator(z_dim=CONTEXT_Z_DIM, hidden_dim=FILM_HIDDEN, n_layers=FILM_N_LAYERS)

        # FiLM-modulated Policy MLP: input = fused(128) only, modulated by z via FiLM
        self.net = FiLMActorMLP(input_dim=FUSED_DIM, hidden_dim=FILM_HIDDEN, n_layers=FILM_N_LAYERS)
        self.mu_layer = nn.Linear(FILM_HIDDEN, dim_act)
        self.log_std_layer = nn.Linear(FILM_HIDDEN, dim_act)
        self.act_limit = act_limit

    def _build_context_step(self, obs):
        """Build a single context vector from current observation for online GRU stepping."""
        speed, gear, rpm, images, lidar, act1, act2 = obs
        # action = last action (act1 is the most recent in act_buf)
        # We use speed + lidar + act1 + zero reward (reward not available at inference)
        ctx = torch.cat((speed, lidar, act1, torch.zeros_like(speed)), dim=-1)
        return ctx  # (B, 24)

    def forward(self, obs, test=False, with_logprob=True):
        speed, gear, rpm, images, lidar, act1, act2 = obs

        # Encode
        img_embed = self.cnn(images.float())
        floats = torch.cat((speed, gear, rpm, lidar, act1, act2), dim=-1)
        float_embed = self.float_mlp(floats)

        # Fusion Gate
        fused = self.fusion_gate(img_embed, float_embed)  # (B, 128)

        # Context: step GRU with current observation
        ctx_input = self._build_context_step(obs)
        device = fused.device
        if self._gru_hidden is None or self._gru_hidden.device != device:
            self._gru_hidden = self.context_encoder.get_initial_hidden(
                batch_size=ctx_input.shape[0], device=device)
        z, self._gru_hidden = self.context_encoder.step(ctx_input, self._gru_hidden, fused_obs=fused)

        # FiLM: z → (γ, β) pairs → modulate actor MLP
        film_params = self.film_generator(z)
        net_out = self.net(fused, film_params)  # (B, 384)

        # Policy head
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if test:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action, logp_pi

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            return a.squeeze().cpu().numpy()

    def reset_context(self):
        """Call at the beginning of each episode to reset GRU state."""
        self._gru_hidden = None
        self.context_encoder.reset_online_state()


class ContextualDroQHybridActorCritic(nn.Module):
    """
    DroQ Actor-Critic with FiLM conditioning.

    Key features:
    - Fusion Gate: learned CNN vs Float balancing
    - Context Encoder: GRU for rapid adaptation
    - FiLM Generator: z → (γ, β) for multiplicative modulation
    - FiLMQHead: Q-networks with FiLM modulation + DroQ dropout
    - Shared FiLM generator for actor + critic (key for deployment adaptation)
    """
    def __init__(self, observation_space, action_space, dropout_rate=0.01):
        super().__init__()
        self.n = 2  # DroQ uses exactly 2 Q-networks

        # Actor has its own encoder (same structure for weight sync)
        self.actor = ContextualSharedBackboneHybridActor(observation_space, action_space)

        # References to actor's encoder for shared feature extraction
        self._actor_cnn = self.actor.cnn
        self._actor_float_mlp = self.actor.float_mlp
        self._actor_fusion_gate = self.actor.fusion_gate

        # Shared context encoder + FiLM generator (used during training)
        self.context_encoder = self.actor.context_encoder
        self.film_generator = self.actor.film_generator

        # 2 FiLM-modulated Q-heads with DroQ dropout
        self.qs = ModuleList([
            FiLMQHead(action_space, input_dim=FUSED_DIM, hidden_dim=FILM_HIDDEN,
                      n_layers=FILM_N_LAYERS, dropout_rate=dropout_rate)
            for _ in range(2)
        ])

    def forward_features(self, obs, context=None):
        """
        Extract features and FiLM params using actor's encoder.
        
        Args:
            obs: tuple of (speed, gear, rpm, images, lidar, act1, act2)
            context: (B, K, 24) tensor of recent transitions, or None
        Returns:
            fused: (B, FUSED_DIM) fused sensor features
            film_params: list of (γ, β) tuples for FiLM modulation
        """
        speed, gear, rpm, images, lidar, act1, act2 = obs

        img_embed = self._actor_cnn(images.float())
        floats = torch.cat((speed, gear, rpm, lidar, act1, act2), dim=-1)
        float_embed = self._actor_float_mlp(floats)

        # Fusion Gate
        fused = self._actor_fusion_gate(img_embed, float_embed)  # (B, 128)

        # Context → FiLM params (cross-attention queries history with current obs)
        if context is not None:
            z = self.context_encoder(context, fused_obs=fused)  # (B, 64)
        else:
            z = torch.zeros(fused.shape[0], CONTEXT_Z_DIM, device=fused.device)

        film_params = self.film_generator(z)
        return fused, film_params, z

    def actor_from_features(self, fused, film_params, test=False, with_logprob=True):
        """Compute action from fused features + FiLM params."""
        net_out = self.actor.net(fused, film_params)  # FiLM-modulated actor MLP
        mu = self.actor.mu_layer(net_out)
        log_std = self.actor.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if test:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.actor.act_limit * pi_action
        return pi_action, logp_pi

    def q_from_features(self, fused, act, film_params, q_idx=None):
        """Compute Q-values from fused features + FiLM params."""
        if q_idx is not None:
            return self.qs[q_idx](fused, act, film_params)
        return [q(fused, act, film_params) for q in self.qs]

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()


# RNN: ==========================================================


def rnn(input_size, rnn_size, rnn_len):
    """
    sizes is ignored for now, expect first values and length
    """
    num_rnn_layers = rnn_len
    assert num_rnn_layers >= 1
    hidden_size = rnn_size

    gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_rnn_layers, bias=True, batch_first=True, dropout=0, bidirectional=False)
    return gru


class SquashedGaussianRNNActor(nn.Module):
    def __init__(self, obs_space, act_space, rnn_size=100, rnn_len=2, mlp_sizes=(100, 100), activation=nn.ReLU):
        super().__init__()
        dim_obs = sum(prod(s for s in space.shape) for space in obs_space)
        dim_act = act_space.shape[0]
        act_limit = act_space.high[0]
        self.rnn = rnn(dim_obs, rnn_size, rnn_len)
        self.mlp = mlp([rnn_size] + list(mlp_sizes), activation, activation)
        self.mu_layer = nn.Linear(mlp_sizes[-1], dim_act)
        self.log_std_layer = nn.Linear(mlp_sizes[-1], dim_act)
        self.act_limit = act_limit
        self.h = None
        self.rnn_size = rnn_size
        self.rnn_len = rnn_len

    def forward(self, obs_seq, test=False, with_logprob=True, save_hidden=False):
        """
        obs: observation
        h: hidden state
        Returns:
            pi_action, log_pi, h
        """
        self.rnn.flatten_parameters()

        # sequence_len = obs_seq[0].shape[0]
        batch_size = obs_seq[0].shape[0]

        if not save_hidden or self.h is None:
            device = obs_seq[0].device
            h = torch.zeros((self.rnn_len, batch_size, self.rnn_size), device=device)
        else:
            h = self.h

        obs_seq_cat = torch.cat(obs_seq, -1)
        net_out, h = self.rnn(obs_seq_cat, h)
        net_out = net_out[:, -1]
        net_out = self.mlp(net_out)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if test:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        pi_action = pi_action.squeeze()

        if save_hidden:
            self.h = h

        return pi_action, logp_pi

    def act(self, obs, test=False):
        obs_seq = tuple(o.view(1, *o.shape) for o in obs)  # artificially add sequence dimension
        with torch.no_grad():
            a, _ = self.forward(obs_seq=obs_seq, test=test, with_logprob=False, save_hidden=True)
            return a.squeeze().cpu().numpy()


class RNNQFunction(nn.Module):
    """
    The action is merged in the latent space after the RNN
    """
    def __init__(self, obs_space, act_space, rnn_size=100, rnn_len=2, mlp_sizes=(100, 100), activation=nn.ReLU):
        super().__init__()
        dim_obs = sum(prod(s for s in space.shape) for space in obs_space)
        dim_act = act_space.shape[0]
        self.rnn = rnn(dim_obs, rnn_size, rnn_len)
        self.mlp = mlp([rnn_size + dim_act] + list(mlp_sizes) + [1], activation)
        self.h = None
        self.rnn_size = rnn_size
        self.rnn_len = rnn_len

    def forward(self, obs_seq, act, save_hidden=False):
        """
        obs: observation
        h: hidden state
        Returns:
            pi_action, log_pi, h
        """
        self.rnn.flatten_parameters()

        # sequence_len = obs_seq[0].shape[0]
        batch_size = obs_seq[0].shape[0]

        if not save_hidden or self.h is None:
            device = obs_seq[0].device
            h = torch.zeros((self.rnn_len, batch_size, self.rnn_size), device=device)
        else:
            h = self.h

        # logging.debug(f"len(obs_seq):{len(obs_seq)}")
        # logging.debug(f"obs_seq[0].shape:{obs_seq[0].shape}")
        # logging.debug(f"obs_seq[1].shape:{obs_seq[1].shape}")
        # logging.debug(f"obs_seq[2].shape:{obs_seq[2].shape}")
        # logging.debug(f"obs_seq[3].shape:{obs_seq[3].shape}")

        obs_seq_cat = torch.cat(obs_seq, -1)

        # logging.debug(f"obs_seq_cat.shape:{obs_seq_cat.shape}")

        net_out, h = self.rnn(obs_seq_cat, h)
        # logging.debug(f"1 net_out.shape:{net_out.shape}")
        net_out = net_out[:, -1]
        # logging.debug(f"2 net_out.shape:{net_out.shape}")
        net_out = torch.cat((net_out, act), -1)
        # logging.debug(f"3 net_out.shape:{net_out.shape}")
        q = self.mlp(net_out)

        if save_hidden:
            self.h = h

        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class RNNActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, rnn_size=100, rnn_len=2, mlp_sizes=(100, 100), activation=nn.ReLU):
        super().__init__()

        act_limit = action_space.high[0]

        # build policy and value functions
        self.actor = SquashedGaussianRNNActor(observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation)
        self.q1 = RNNQFunction(observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation)
        self.q2 = RNNQFunction(observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation)
