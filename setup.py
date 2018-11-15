from setuptools import setup, find_packages

setup(name='leap utils',
      version='0.1',
      description='leap_utils',
      url='http://github.com/janclemenslab/leap_tuils',
      author='Jan Clemens',
      author_email='clemensjan@googlemail.com',
      license='MIT',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      # install_requires=['numpy', 'videoreader', 'skimage', 'leap'],
      tests_require=['nose', 'matplotlib'],
      test_suite='nose.collector',
      include_package_data=True,
      zip_safe=False
      )
