from setuptools import setup

setup(name='elastopy',
      version='0.3',
      description='Stead state linear elasticity problem',
      url='https://github.com/nasseralkmim/elastopy',
      download_url='https://github.com/nasseralkmim/elastopy/releases/tag/0.3',
      author='Nasser Alkmim',
      author_email='nasser.alkmim@gmail.com',
      license='GLP3',
      packages=['elastopy'],
      install_requires=[
          'numpy',
          'matplotlib',
          'networkx'
      ],
      zip_safe=False)
