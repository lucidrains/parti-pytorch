<img src="./parti.jpeg" width="450px"></img>

## Parti - Pytorch

Implementation of <a href="https://parti.research.google/">Parti</a>, Google's pure attention-based text-to-image neural network, in Pytorch.

This repository also contains working training code for <a href="https://ai.googleblog.com/2022/05/vector-quantized-image-modeling-with.html">ViT VQGan VAE</a>. It also contains some additional modifications for faster training from vision transformers literature.

<a href="https://www.youtube.com/watch?v=qS-iYnp00uc">Yannic Kilcher</a>

Please join <a href="https://discord.gg/xBPBXfcFHd"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a> if you are interested in helping out with the replication with the <a href="https://laion.ai/">LAION</a> community

## Install

```bash
$ pip install parti-pytorch
```

## Usage

First you will need to train your Transformer VQ-GAN VAE

```python
from parti_pytorch import VitVQGanVAE, VQGanVAETrainer

vit_vae = VitVQGanVAE(
    dim = 256,               # dimensions
    image_size = 256,        # target image size
    patch_size = 16,         # size of the patches in the image attending to each other
    num_layers = 3           # number of layers
).cuda()

trainer = VQGanVAETrainer(
    vit_vae,
    folder = '/path/to/your/images',
    num_train_steps = 100000,
    lr = 3e-4,
    batch_size = 4,
    grad_accum_every = 8,
    amp = True
)

trainer.train()
```

Then

```python
import torch
from parti_pytorch import Parti, VitVQGanVAE

# first instantiate your ViT VQGan VAE
# a VQGan VAE made of transformers

vit_vae = VitVQGanVAE(
    dim = 256,               # dimensions
    image_size = 256,        # target image size
    patch_size = 16,         # size of the patches in the image attending to each other
    num_layers = 3           # number of layers
).cuda()

vit_vae.load_state_dict(torch.load(f'/path/to/vae.pt')) # you will want to load the exponentially moving averaged VAE

# then you plugin the ViT VqGan VAE into your Parti as so

parti = Parti(
    vae = vit_vae,            # vit vqgan vae
    dim = 512,                # model dimension
    depth = 8,                # depth
    dim_head = 64,            # attention head dimension
    heads = 8,                # attention heads
    dropout = 0.,             # dropout
    cond_drop_prob = 0.25,    # conditional dropout, for classifier free guidance
    ff_mult = 4,              # feedforward expansion factor
    t5_name = 't5-large',     # name of your T5
)

# ready your training text and images

texts = [
    'a child screaming at finding a worm within a half-eaten apple',
    'lizard running across the desert on two feet',
    'waking up to a psychedelic landscape',
    'seashells sparkling in the shallow waters'
]

images = torch.randn(4, 3, 256, 256).cuda()

# feed it into your parti instance, with return_loss set to True

loss = parti(
    texts = texts,
    images = images,
    return_loss = True
)

loss.backward()

# do this for a long time on much data
# then...

images = parti.generate(texts = [
    'a whale breaching from afar',
    'young girl blowing out candles on her birthday cake',
    'fireworks with blue and green sparkles'
], cond_scale = 3., return_pil_images = True) # conditioning scale for classifier free guidance

# List[PILImages] (256 x 256 RGB)
```

Realistically, when scaling up, you'll want to pre-encode your text into tokens and their respective mask

```python
from parti_pytorch.t5 import t5_encode_text

images = torch.randn(4, 3, 256, 256).cuda()

text_token_embeds, text_mask = t5_encode_text([
    'a child screaming at finding a worm within a half-eaten apple',
    'lizard running across the desert on two feet',
    'waking up to a psychedelic landscape',
    'seashells sparkling in the shallow waters'
], name = 't5-large', output_device = images.device)

# store somewhere, then load with the dataloader

loss = parti(
    text_token_embeds = text_token_embeds,
    text_mask = text_mask,
    images = images,
    return_loss = True
)

loss.backward()
```

## Appreciation

- <a href="https://stability.ai/">StabilityAI</a> for the sponsorship, as well as my other sponsors, for affording me the independence to open source artificial intelligence.

- <a href="https://huggingface.co/">🤗 Huggingface</a> for the transformers library and the ease for encoding text with T5 language model

## Todo

- [x] add 2d relative positional bias to parti autoregressive transformer
- [x] cite all techniques adopted from vision transformer literature in vit vqgan if they work
- [x] get working vit vqgan-vae trainer code, as discriminator needs to be trained
- [x] use crossformer embed layer for initial convolution in discriminator
- [ ] preencoding of text with designated t5
- [ ] training code for parti
- [ ] inference caching
- [ ] automatic filtering with Coca https://github.com/lucidrains/CoCa-pytorch
- [ ] bring in the super-resoluting convolutional net mentioned in the paper, with training code
- [ ] initialize 2d rel pos bias in conv-like pattern
- [ ] consider a small nerf-like MLP at the end of vit-vqgan, similar to https://arxiv.org/abs/2107.04589

## Citations

```bibtex
@inproceedings{Yu2022Pathways
    title   = {Pathways Autoregressive Text-to-Image Model},
    author  = {Jiahui Yu*, Yuanzhong Xu†, Jing Yu Koh†, Thang Luong†, Gunjan Baid†, Zirui Wang†, Vijay Vasudevan†, Alexander Ku†, Yinfei Yang, Burcu Karagol Ayan, Ben Hutchinson, Wei Han, Zarana Parekh, Xin Li, Han Zhang, Jason Baldridge†, Yonghui Wu*},
    year    = {2022}
}
```

```bibtex
@article{Shleifer2021NormFormerIT,
    title   = {NormFormer: Improved Transformer Pretraining with Extra Normalization},
    author  = {Sam Shleifer and Jason Weston and Myle Ott},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2110.09456}
}
```

```bibtex
@article{Sankararaman2022BayesFormerTW,
    title   = {BayesFormer: Transformer with Uncertainty Estimation},
    author  = {Karthik Abinav Sankararaman and Sinong Wang and Han Fang},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2206.00826}
}
```

```bibtex
@article{Lee2021VisionTF,
    title   = {Vision Transformer for Small-Size Datasets},
    author  = {Seung Hoon Lee and Seunghyun Lee and Byung Cheol Song},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2112.13492}
}
```

```bibtex
@article{Chu2021DoWR,
    title   = {Do We Really Need Explicit Position Encodings for Vision Transformers?},
    author  = {Xiangxiang Chu and Bo Zhang and Zhi Tian and Xiaolin Wei and Huaxia Xia},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2102.10882}
}
```

```bibtex
@article{So2021PrimerSF,
    title   = {Primer: Searching for Efficient Transformers for Language Modeling},
    author  = {David R. So and Wojciech Ma'nke and Hanxiao Liu and Zihang Dai and Noam M. Shazeer and Quoc V. Le},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2109.08668}
}
```

```bibtex
@inproceedings{Wang2021CrossFormerAV,
    title   = {CrossFormer: A Versatile Vision Transformer Hinging on Cross-scale Attention},
    author  = {Wenxiao Wang and Lulian Yao and Long Chen and Binbin Lin and Deng Cai and Xiaofei He and Wei Liu},
    year    = {2021}
}
```
