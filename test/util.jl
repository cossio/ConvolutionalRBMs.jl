using Test: @testset, @test, @test_throws, @inferred
import ConvolutionalRBMs as ConvRBMs
using ConvolutionalRBMs: reshape_maybe, expand_size, samepad

@testset "reshape_maybe" begin
    @test (@inferred reshape_maybe(1, ())) == 1
    @test_throws Exception reshape_maybe(1, 1)
    @test_throws Exception reshape_maybe(1, (1,))

    @test (@inferred reshape_maybe(fill(1), ())) == 1
    @test (@inferred reshape_maybe(fill(1), (1,))) == [1]
    @test (@inferred reshape_maybe(fill(1), (1,1))) == hcat([1])
    @test_throws Exception reshape_maybe(fill(1), (1,2))

    @test (@inferred reshape_maybe([1], ())) == 1
    @test (@inferred reshape_maybe([1], (1,))) == [1]
    @test (@inferred reshape_maybe([1], (1,1))) == hcat([1])
    @test_throws Exception reshape_maybe([1], (1,2))

    A = randn(2,2)
    @test (@inferred reshape_maybe(A, 4)) == reshape(A, 4)
end

@testset "expand_size" begin
    @test expand_size(Val(2), 1) == (1,1)
    @test expand_size(Val(2), 2) == (2,2)
    @test expand_size(Val(3), 1) == (1,1,1)
    @test expand_size(Val(2), (1,2)) == (1,2)
    @test expand_size(Val(3), (1,2,3)) == (1,2,3)
    @test_throws MethodError expand_size(Val(3), (1,2))
end

@testset "samepad" begin
    @test expand_size(Val(2), 1) == (1,1)
    @test expand_size(Val(2), 2) == (2,2)
    @test expand_size(Val(3), 1) == (1,1,1)
    @test expand_size(Val(2), (1,2)) == (1,2)
    @test expand_size(Val(3), (1,2,3)) == (1,2,3)
    @test_throws MethodError expand_size(Val(3), (1,2))
end
