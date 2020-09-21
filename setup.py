# coding: utf-8

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='xsim',
    version='0.4.0',
    description="xikasan's simulation tool set",
    long_description=readme,
    author='xikasan',
    # author_email='',
    url='https://github.com/xikasan/xsim',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
