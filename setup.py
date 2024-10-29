# via https://github.com/kennethreitz/setup.py

import setuptools

with open('README.md') as f:
    readme = f.read()

license = ''

setuptools.setup(
    name='scattering_functions',
    version='0.0.1',
    description='',
    long_description=readme,
    author='Adam Carter',
    author_email='adam.rob.carter@gmail.com',
    url='',
    license=license,
    py_modules=['scattering_functions'],
)