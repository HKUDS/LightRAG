import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


vars2find = ["__author__", "__version__", "__url__"]
vars2readme = {}
with open("./lightrag/__init__.py") as f:
    for line in f.readlines():
        for v in vars2find:
            if line.startswith(v):
                line = line.replace(" ", "").replace('"', "").replace("'", "").strip()
                vars2readme[v] = line.split("=")[1]

deps = []
with open("./requirements.txt") as f:
    for line in f.readlines():
        if not line.strip():
            continue
        deps.append(line.strip())

setuptools.setup(
    name="lightrag-hku",
    url=vars2readme["__url__"],
    version=vars2readme["__version__"],
    author=vars2readme["__author__"],
    description="LightRAG: Simple and Fast Retrieval-Augmented Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["lightrag"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=deps,
)
