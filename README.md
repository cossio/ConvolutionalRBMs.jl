# ConvolutionalRBMs Julia package

[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/cossio/ConvolutionalRBMs.jl/blob/master/LICENSE.md)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://cossio.github.io/ConvolutionalRBMs.jl/dev)
![](https://github.com/cossio/ConvolutionalRBMs.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/cossio/ConvolutionalRBMs.jl/branch/master/graph/badge.svg?token=ZIPKLORX51)](https://codecov.io/gh/cossio/ConvolutionalRBMs.jl)
![GitHub repo size](https://img.shields.io/github/repo-size/cossio/ConvolutionalRBMs.jl)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/cossio/ConvolutionalRBMs.jl)

Convolutional Restricted Boltzmann machines (RBM) in Julia.

## Installation

This package is not registered.
Install with:

```julia
import Pkg
Pkg.add(url="https://github.com/cossio/ConvolutionalRBMs.jl")
```

This package does not export any symbols.
To avoid typing a long name every time, import as:

```julia
import ConvolutionalRBMs as ConvRBMs 
```
## Compat

Needs Julia 1.7, because type inference sometimes fails on Julia 1.6.
Also using `Base.Returns`, introduced in Julia 1.7.

## Related

Restricted Boltzmann machines in Julia: <https://github.com/cossio/RestrictedBoltzmannMachines.jl>.