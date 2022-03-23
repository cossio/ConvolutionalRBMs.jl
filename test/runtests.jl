import MKL # must load before anything else
using SafeTestsets: @safetestset

@time @safetestset "util" begin include("util.jl") end
@time @safetestset "conv" begin include("conv.jl") end
@time @safetestset "categorical" begin include("categorical.jl") end
@time @safetestset "convrbm" begin include("convrbm.jl") end
@time @safetestset "pooling" begin include("pooling.jl") end
@time @safetestset "denseconvrbm" begin include("denseconvrbm.jl") end
@time @safetestset "hankel" begin include("hankel.jl") end
@time @safetestset "free energy" begin include("free_energy.jl") end
