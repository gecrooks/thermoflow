# Copyright 2021-2024 Gavin E. Crooks
#
# This source code is licensed under the Apache License 2.0
# found in the LICENSE file in the root directory of this source tree.

# Command line interface for the about() function
# > python -m thermoflow.about
#
# NB: This module should not be imported by any other code in the package
# (else we will get multiple import warnings)

if __name__ == "__main__":
    import thermoflow

    thermoflow.about()
