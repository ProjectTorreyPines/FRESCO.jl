using FRESCO
using Test
import FRESCO: IMAS

dd = IMAS.json2imas(joinpath(@__DIR__, "dd_d3d.json"))
fw_r, fw_z = IMAS.first_wall(dd.wall)
ΔR = maximum(fw_r) - minimum(fw_r)
ΔZ = maximum(fw_z) - minimum(fw_z)
Rs = range(max(0.01, minimum(fw_r) - ΔR / 20), maximum(fw_r) + ΔR / 20, 33)
Zs = range(minimum(fw_z) - ΔZ / 20, maximum(fw_z) + ΔZ / 20, 65)

@testset "FRESCO Initialize" begin
    @test Canvas(dd, 33, 65) isa Canvas
    @test Canvas(dd, Rs, Zs) isa Canvas
end
canvas = Canvas(dd, Rs, Zs)

# to restore old test behavior, we set λ_regularize to 0.0 and
#   generally use initialize_current=true in the solves

@testset "FRESCO Solve" begin
    profile = FRESCO.PprimeFFprime(dd; grid=:psi_norm)
    @test FRESCO.solve!(canvas, profile, 10, 3; relax=0.5, debug=true, control=:shape, tolerance=0.0, initialize_current=true) == 0

    profile = FRESCO.PressureJtoR(dd, grid=:rho_tor_norm)
    @test FRESCO.solve!(canvas, profile, 20, 3; relax=0.5, debug=false, control=:shape, tolerance=1e-4, fixed_coils=25:48, initialize_current=true) == 0
    canvas.λ_regularize = 0.0 # turn off regularization for the next test
    @test FRESCO.solve!(canvas, profile, 10, 3; relax=0.5, debug=false, control=:shape, tolerance=1e-3, initialize_current=true) == 1 # won't converge in iterations

    canvas.λ_regularize = 1e-14 # reset regularization for the next tests
    profile = FRESCO.PressureJt(dd, grid=:rho_tor_norm)
    @test FRESCO.solve!(canvas, profile, 30, 3; relax=0.5, debug=false, control=:vertical, tolerance=1e-4, compute_Ip_from=:grid, initialize_current=false) == 0
    @test FRESCO.solve!(canvas, profile, 10, 3; relax=0.5, debug=false, control=:radial, tolerance=1e-3, initialize_current=true) == 1 # should not converge, but end before it errors
    @test FRESCO.solve!(canvas, profile, 30, 3; relax=0.5, debug=false, control=:position, tolerance=1e-3, Rtarget=1.75, Ztarget=0.0, initialize_current=true) == 0
    @test FRESCO.solve!(canvas, profile, 20, 3; relax=0.5, debug=false, control=:eddy, tolerance=1e-3) == 0
end

psin = 0.1 * π
gm1 = canvas._gm1_itp(psin)
@testset "FRESCO Profiles" begin
    profile = FRESCO.PprimeFFprime(dd)
    pprime = FRESCO.Pprime(canvas, profile, psin)
    ffprime = FRESCO.FFprime(canvas, profile, psin)
    jtor = FRESCO.JtoR(canvas, profile, psin)
    @test jtor ≈ -2π * (pprime + ffprime * gm1 / FRESCO.μ₀)

    profile = FRESCO.PressureJtoR(dd)
    pprime = FRESCO.Pprime(canvas, profile, psin)
    ffprime = FRESCO.FFprime(canvas, profile, psin)
    jtor = FRESCO.JtoR(canvas, profile, psin)
    @test jtor ≈ -2π * (pprime + ffprime * gm1 / FRESCO.μ₀)

    profile = FRESCO.PressureJt(dd)
    pprime = FRESCO.Pprime(canvas, profile, psin)
    ffprime = FRESCO.FFprime(canvas, profile, psin)
    jtor = FRESCO.JtoR(canvas, profile, psin)
    @test jtor ≈ -2π * (pprime + ffprime * gm1 / FRESCO.μ₀)
end
