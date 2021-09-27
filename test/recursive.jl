using Test, JETTest

using ClimaCore.RecursiveApply

if VERSION < v"1.7.0-beta1"
for x in [
    1.0,
    1.0f0,
    (1.0, 2.0),
    (1.0f0, 2.0f0),
    (a = 1.0, b = (x1 = 2.0, x2 = 3.0)),
    (a = 1.0f0, b = (x1 = 2.0f0, x2 = 3.0f0)),
]
    @test_nodispatch 2 ⊠ x
    @test_nodispatch x ⊞ x
    @test_nodispatch RecursiveApply.rdiv(x, 3)
end
end