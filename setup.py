import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="naif",
    version="0.1.0",
    author="Leandro Beraldo e Silva",
    author_email="lberaldoesilva@gmail.com",
    description="Numerical Analysis and Integer combinations of Frequencies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(include=['naif','naif.*']),
    python_requires='>=3',
    install_requires=["numpy","scipy"],
    package_data={"": ["README.md","LICENSE"]},
    license='MIT'
)
