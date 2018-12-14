import sys
from setuptools import find_packages, setup
import simplecv

install_requires = [
    'numpy>=1.11.1',
    'tensorboardX',
]
setup(
    name='simplecv',
    version=simplecv.__version__,
    description='Simplify training, evaluation, prediction in Pytorch',
    keywords='computer vision using pytorch 1.0',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Utilities',
    ],
    url='https://github.com/Z-Zheng/simplecv.git',
    author='Zhuo Zheng',
    author_email='zhuozheng_2017@163.com',
    license='GPLv3',
    setup_requires=[],
    tests_require=[],
    install_requires=install_requires,
    zip_safe=False)
