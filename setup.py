#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy',
    'tensorboardX',
    'gitpython',
    'pytorch=0.4.0'
]

setup(
    name='tensorboard-wrapper',
    version='0.1',
    description='wrapper of TensorBoardX for pytorch models',
    long_description= history,
    author='Lucas Liu',
    author_email='llychinalz@gmail.com',
    url='https://github.com/LiyuanLucasLiu/tensorboard-wrapper',
    packages=find_packages(exclude=['docs']),
    include_package_data=True,
    install_requires=requirements,
    license='MIT license',
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)

# python setup.py sdist bdist_wheel --universal
# twine upload dist/*