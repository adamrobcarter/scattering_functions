from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

license = ''

setup(
    name="scattering_functions",
    version="0.0.1",
    description="",
    long_description=readme,
    author="Adam Carter",
    author_email="adam.rob.carter@gmail.com",
    url="",
    license=license,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["numpy", "scipy", "tqdm"],
)
