using Test
using VascularTreeSim
using StaticArrays
using Statistics
using LinearAlgebra
using Random

@testset "VascularTreeSim.jl" begin
    include("test_cube_shell.jl")
    include("test_sphere_shell.jl")
    include("test_cylinder.jl")
end
