# Copyright 2021, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License 2.0
# found in the LICENSE file in the root directory of this source tree.

import glob
import io
import subprocess

import thermoflow


def test_version() -> None:
    assert thermoflow.__version__


def test_about() -> None:
    out = io.StringIO()
    thermoflow.about(out)
    print(out)


def test_about_main() -> None:
    rval = subprocess.call(["python", "-m", "thermoflow.about"])
    assert rval == 0


def test_copyright() -> None:
    """Check that source code files contain a copyright line"""
    for fname in glob.glob("thermoflow/**/*.py", recursive=True):
        print("Checking " + fname + " for copyright header")

        with open(fname) as f:
            for line in f.readlines():
                if not line.strip():
                    continue
                assert line.startswith("# Copyright")
                break
