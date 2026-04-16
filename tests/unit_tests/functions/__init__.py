# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Function-level unit tests for torchtitan-npu.

These tests use fake tensor and mock operations to test individual
NPU-specific functions without requiring actual NPU hardware.

Test modules:
    - test_dsa_fake: DSA (Dynamic Sparse Attention) tests
    - test_li_loss_fake: LI Loss tests
    - test_gmm_fake: GMM (Gaussian Mixture Model) tests
"""
