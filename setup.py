import setuptools

tests_require = ["pytest>=6.1.1", "coverage>=5.3", "pytest-cov>=2.10.1"]

setuptools.setup(
    name="equity-risk-model",
    version="0.0.3",
    description="Portfolio analysis using an equity multi-factor risk model.",
    packages=setuptools.find_packages(),
    install_requires=["cvxpy", "numpy", "pandas", "scipy"],
    tests_require=tests_require,
    extras_requires={"test": tests_require},
)
