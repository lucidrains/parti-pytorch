from setuptools import setup, find_packages
exec(open('parti_pytorch/version.py').read())

setup(
  name = 'parti-pytorch',
  packages = find_packages(exclude=[]),
  version = __version__,
  license='MIT',
  description = 'Parti - Pathways Autoregressive Text-to-Image Model - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/parti-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'text-to-image'
  ],
  install_requires=[
    'einops>=0.4',
    'einops-exts',
    'ema-pytorch',
    'torch>=1.6',
    'torchvision',
    'transformers',
    'vector-quantize-pytorch>=0.9.2'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
