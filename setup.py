#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
#
from setuptools import find_packages, setup

setup(
    name="online_conformal",
    version="1.0.0",
    author="Aadyot Bhatnagar",
    author_email="abhatnagar@salesforce.com",
    description="A library for time series conformal prediction",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    packages=find_packages(include=["online_conformal*"]),
    install_requires=open("requirements.txt").read().split("\n"),
)
