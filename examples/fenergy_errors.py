# Copyright 2021-2024 Gavin E. Crooks
#
# This source code is licensed under the Apache License 2.0
# found in the LICENSE file in the root directory of this source tree.

import numpy as np

from thermoflow import fenergy_bar, fenergy_bayesian

fe = 0
diss = 100
count = 1000


def errors(diss, count):
    fe = 0
    work_f = np.random.normal(loc=diss + fe, scale=np.sqrt(2 * diss), size=(count,))
    work_r = np.random.normal(loc=diss - fe, scale=np.sqrt(2 * diss), size=(count,))

    print()
    print(f"Gaussian work, count: {count}, mean dissipation: {diss} kT")
    print("BAR error:", fenergy_bar(work_f, work_r, uncertainty_method="BAR")[1])
    print("MBAR error:", fenergy_bar(work_f, work_r, uncertainty_method="MBAR")[1])
    print("Bayesian error:", fenergy_bayesian(work_f, work_r)[1])
    print(
        "Logistic approx:",
        fenergy_bar(work_f, work_r, uncertainty_method="Logistic")[1],
    )


errors(10, 1000)
errors(100, 1000)
