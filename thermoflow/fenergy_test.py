# Copyright 2021, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import numpy as np
import pytest
from numpy.random import normal
from scipy.special import expit

from disorder import (
    fenergy_bar,
    fenergy_bayesian,
    fenergy_logmeanexp,
    fenergy_logmeanexp_gaussian,
    fenergy_symmetric_bar,
    fenergy_symmetric_bidirectional,
    fenergy_symmetric_nnznm,
)
from disorder.fenergy import _logexpit


def test_fenergy_logmeanexp() -> None:
    assert np.isclose(fenergy_logmeanexp(work_f), 9.462820605969167)
    assert np.isclose(fenergy_logmeanexp(work_r), -8.992876786373467)


def test_fenergy_logmeanexp_gaussian() -> None:
    assert np.isclose(fenergy_logmeanexp_gaussian(work_f), 8.687926143699123)


def test_fenergy_bar() -> None:
    assert np.allclose(
        fenergy_bar(work_f, work_r), (10.126372790165997, 0.39715725693786963)
    )
    assert np.allclose(
        fenergy_bar(work_r, work_f), (-10.126372790165997, 0.39715725693786963)
    )

    assert np.allclose(
        fenergy_bar(work_f, work_r, uncertainty_method="BAR"),
        (10.126372790165997, 0.39715725693786963),
    )
    assert np.allclose(
        fenergy_bar(work_f, work_r, uncertainty_method="MBAR"),
        (10.126372790165997, 0.40080301959064535),
    )
    assert np.allclose(
        fenergy_bar(work_f, work_r, uncertainty_method="Logistic"),
        (10.126372790165997, 0.40080301959064535),
    )

    # Note: for high dissipation the error estimate too low with BAR, too high with MBAR
    assert np.allclose(
        fenergy_bar(work_f_diss, work_r_diss, uncertainty_method="BAR"),
        (9.341808032205543, 1.1165063184118718),
    )
    assert np.allclose(
        fenergy_bar(work_f_diss, work_r_diss, uncertainty_method="MBAR"),
        (9.341808032205543, 78.83605670383308),
    )
    assert np.allclose(
        fenergy_bar(work_f_diss, work_r_diss, uncertainty_method="Logistic"),
        (9.341808032205543, 5.928521477224136),
    )
    assert np.allclose(
        fenergy_bar(
            work_f_diss,
            work_r_diss,
            weights_f=np.ones_like(work_f_diss),
            weights_r=np.ones_like(work_r_diss),
            uncertainty_method="Logistic",
        ),
        (9.341808032205543, 5.928521477224136),
    )


def test_fenergy_bar_error() -> None:
    with pytest.raises(ValueError):
        fenergy_bar(work_f, work_r, uncertainty_method="NOT_A_METHOD")


def test_fenergy_bayesian() -> None:
    assert np.allclose(
        fenergy_bayesian(work_f, work_r),
        (10.129329739999651, 0.5091539010758899),
    )

    assert np.allclose(
        fenergy_bayesian(work_f_diss, work_r_diss),
        (9.491682713739426, 5.564798805609418),
    )


def test_logexpit() -> None:

    # For values near 0, no tricks
    for n in range(10):
        x = normal()
        assert np.isclose(_logexpit(x), np.log(expit(x)))

    assert np.isclose(_logexpit(-1000.0), -1000.0)  # type: ignore
    assert np.isclose(_logexpit(1000.0), 0.0)  # type: ignore


def test_fenergy_symmetric_bar() -> None:

    assert np.allclose(
        fenergy_symmetric_bar(work_sym_f, work_sym_r),
        (1.1867614815656458, 0.21931716664445491),
    )


def test_fenergy_symmetric_nnznm() -> None:
    # Note only tests that code runs. No regression as of yet.
    fenergy_symmetric_nnznm(work_sym_f, work_sym_r)


def test_fenergy_symmetric_bidirectional() -> None:
    # Note only tests that code runs. No regression as of yet.
    fenergy_symmetric_bidirectional(work_sym_f, work_sym_r)


# -- data --

# Fixed random work samples for testing.
# fe = 10
# diss = 4
# work_f = random.normal(loc=diss + fe, scale=np.sqrt(2 * diss), size=(20,))
# work_r = random.normal(loc=diss - fe, scale=np.sqrt(2 * diss), size=(20,))

work_f = np.array(
    [
        12.26186058,
        12.02296197,
        15.81950764,
        10.89312792,
        10.25310224,
        6.57237884,
        20.70918778,
        12.52717333,
        10.98228327,
        15.11407533,
        10.1928446,
        11.86228712,
        13.85018359,
        17.51308605,
        13.71277499,
        14.78423781,
        10.79028309,
        15.73678937,
        13.34587623,
        14.83121349,
    ]
)

work_r = np.array(
    [
        -7.03024785,
        -5.97147992,
        -9.54545887,
        -4.15253714,
        -5.13042167,
        -10.94999928,
        -8.83169733,
        -3.70606829,
        -8.25225619,
        -4.41218262,
        -8.61761745,
        -4.86517229,
        -4.23155588,
        -9.87869866,
        -8.15082176,
        -9.83398181,
        -7.43388798,
        -8.42498504,
        -8.58419969,
        -9.82309618,
    ]
)

# Fixed random work samples for testing, high dissipation
# fe = 10
# diss = 20

work_f_diss = np.array(
    [
        31.93691244,
        37.52776423,
        27.84175637,
        44.88480481,
        42.96347851,
        28.25698468,
        23.82297813,
        23.34256961,
        18.7881143,
        25.31003161,
    ]
)
work_r_diss = np.array(
    [
        8.99668577,
        0.83206027,
        4.19698101,
        9.21375163,
        7.14493204,
        7.36804987,
        0.76412305,
        13.23269751,
        17.78017694,
        16.22043168,
    ]
)

# Generated from a simple simulation of
# harmonic oscillator moving to and fro.
work_sym_f = np.array(
    [
        1.82819018,
        1.36855722,
        3.15124547,
        3.70311204,
        7.97671385,
        2.65068191,
        8.19611493,
        0.24970324,
        3.98034835,
        5.1818346,
        1.57846703,
        1.19969679,
        -0.40316984,
        3.66036448,
        8.0229261,
        4.25364127,
        3.94486632,
        2.96836278,
        0.36038236,
        1.76974979,
        0.8004306,
        4.67280096,
        1.49874792,
        4.81476805,
        4.96769674,
        1.90638043,
        -0.32600167,
        3.38392651,
        3.79314834,
        0.94598643,
        1.95623881,
        6.11134062,
        1.87657695,
        5.82658531,
        5.87858761,
        4.54366715,
        1.96390498,
        3.63142078,
        4.4287167,
        3.3585574,
        5.1397567,
        4.79327128,
        3.15155961,
        0.30086432,
        2.04971419,
        1.34595448,
        6.27214952,
        5.25437326,
        8.03978488,
        2.72734344,
        0.88520126,
        5.75472987,
        4.97882958,
        0.31924861,
        7.07856703,
        7.95909254,
        2.3827212,
        6.32013187,
        6.40346971,
        6.75324138,
        8.28685036,
        1.20987223,
        5.30231901,
        2.64366919,
        4.66766505,
        4.92502304,
        -3.55804016,
        8.83309716,
        -0.15187293,
        6.99749128,
        2.13440322,
        7.14182222,
        7.47293331,
        5.5724692,
        5.10984664,
        5.72953769,
        4.5543944,
        2.03872872,
        5.85949342,
        7.90117102,
        3.08241451,
        7.00464683,
        5.18575533,
        5.67510774,
        5.70554863,
        3.31311725,
        2.00246591,
        9.08164636,
        3.63798151,
        3.65981263,
        3.30865548,
        1.22848849,
        5.15646295,
        5.58547037,
        11.92743536,
        0.14573392,
        4.44491747,
        4.68708094,
        5.80561741,
        4.29157184,
    ]
)
work_sym_r = np.array(
    [
        1.32803648,
        -1.46565987,
        -3.91438172,
        -0.18983798,
        -4.88579712,
        -1.26741734,
        -5.27859627,
        0.55945393,
        -2.11904812,
        -1.02599031,
        0.38483052,
        0.70294448,
        5.04710393,
        -2.83653239,
        -4.57776588,
        -3.05089428,
        -4.42805097,
        -1.63812375,
        2.07513872,
        3.0545816,
        2.97141303,
        -2.83771512,
        0.62048284,
        -4.16506427,
        -1.80694098,
        -1.00990265,
        2.36238464,
        -2.49992193,
        -4.76327979,
        2.46269644,
        2.31389956,
        -6.14219227,
        -0.74538983,
        -7.77629193,
        -2.64513459,
        -4.53824845,
        0.72225589,
        1.09491531,
        -1.33724845,
        -3.13851951,
        -1.98835653,
        -6.11549039,
        -3.92871721,
        -3.64929078,
        -0.92120638,
        0.30030679,
        -6.64325116,
        -1.59348937,
        -5.08270715,
        -1.07814159,
        3.03018205,
        -4.29619546,
        -0.84256382,
        -0.43260979,
        -7.26314053,
        -5.56629806,
        -3.60012248,
        -6.22214006,
        -5.7474615,
        -3.2881238,
        -7.07396391,
        -0.85047779,
        -4.68073175,
        0.08697061,
        -1.2678808,
        -3.86896004,
        1.0097779,
        -2.35230744,
        4.13973734,
        -8.47694328,
        3.85231732,
        -5.34715043,
        -6.48893033,
        -3.52519835,
        -2.20753183,
        0.48452572,
        -3.16530233,
        -0.29440516,
        -0.9906192,
        -3.58376776,
        -2.49564153,
        -6.31664668,
        1.31543616,
        -2.63037867,
        -3.11807526,
        -2.59228726,
        -2.09934505,
        -3.36107132,
        -1.67505966,
        -0.59628801,
        2.64434669,
        -0.8510169,
        -0.46134474,
        -4.36162684,
        -12.99690175,
        -1.31253928,
        0.72939001,
        -1.35800046,
        -4.76475975,
        -1.66313632,
    ]
)
