<img src="./parti.jpeg" width="450px"></img>

## Parti - Pytorch (wip)

Implementation of <a href="https://parti.research.google/">Parti</a>, Google's pure attention-based text-to-image neural network, in Pytorch

## Install

```bash
$ pip install parti-pytorch
```

## Usage

```python
import torch
from parti_pytorch import Parti
from parti_pytorch.vit_vqgan import VitVQGanVAE

# first instantiate your ViT VQGan VAE
# a VQGan VAE made of transformers

vit_vae = VitVQGanVAE(
    dim = 512,               # dimensions
    image_size = 256,        # target image size
    num_layers = 4           # number of layers
).cuda()

images = torch.randn(4, 3, 256, 256).cuda()

loss = vit_vae(images, return_loss = True)
loss.backward()

# do the above with as many images as possible
# then you plugin the ViT VqGan VAE into your Parti

parti = Parti(
    vae = vit_vae,            # vit vqgan vae
    dim = 512,                # model dimension
    depth = 8,                # depth
    dim_head = 64,            # attention head dimension
    heads = 8,                # attention heads
    dropout = 0.,             # dropout
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
], cond_scale = 3.)

# (3, 3, 256, 256) <-- save your images
```

## Appreciation

- <a href="https://stability.ai/">StabilityAI</a> for the sponsorship, as well as my other sponsors, for affording me the independence to open source artificial intelligence.

- <a href="https://huggingface.co/">🤗 Huggingface</a> for the transformers library and the ease for encoding text with T5 language model

## Todo

- [ ] get working vit vqgan-vae trainer code, as discriminator needs to be trained
- [ ] preencoding of text with designated t5
- [ ] training code for parti
- [ ] inference caching
- [ ] automatic filtering with Coca https://github.com/lucidrains/CoCa-pytorch
- [ ] bring in the super-resoluting convolutional net mentioned in the paper, with training code

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
