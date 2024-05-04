from setuptools import find_packages,setup

setup(
    name='mcqgenrator',
    version='0.0.1',
    author='abdelanlah',
    author_email='abdelwizanlah@gmail.com',
    install_requires=["langchain","streamlit","python-dotenv","PyPDF2","huggingface_hub","google-search-results"],
    packages=find_packages()
)



