#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

COPY requirements.txt requirements.txt
COPY README.md README.md
COPY conformal_ts conformal_ts
COPY setup.py setup.py
RUN python3 -m pip install -e .
