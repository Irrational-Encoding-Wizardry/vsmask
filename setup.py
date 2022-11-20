from distutils.util import convert_path
from typing import Any, Dict

from setuptools import setup

meta: Dict[str, Any] = {}
with open(convert_path('vsmask/_metadata.py'), encoding='utf-8') as f:
    exec(f.read(), meta)

with open('README.md', encoding='utf-8') as fh:
    long_description = fh.read()

with open('requirements.txt', encoding='utf-8') as fh:
    install_requires = fh.read()


setup(
    name='vsmask',
    version=meta['__version__'],
    author=meta['__author__'],
    author_email=meta['__author_email__'],
    maintainer=meta['__maintainer__'],
    maintainer_email=meta['__maintainer_email__'],
    description='Various masking tools for Vapoursynth',
    long_description=long_description,
    long_description_content_type='text/markdown',
    project_urls={
        'Source Code': 'https://github.com/Irrational-Encoding-Wizardry/vs-mask',
        'Documentation': 'https://vsmask.encode.moe/en/latest/',
        'Contact': 'https://discord.gg/qxTxVJGtst',
    },
    packages=['vsmask', 'vsmask.edge'],
    package_data={
        'vsmask': ['py.typed'],
    },
    install_requires=install_requires,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    zip_safe=False,
    python_requires='>=3.10',
)
