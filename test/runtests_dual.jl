using FRESCO
using Test
import FRESCO: IMAS
import FRESCO.IMAS.ForwardDiff: value, partials

# Equilibrium solve on a canvas of dual numbers: the derivatives carried through the solve are compared
# with central finite differences of the Float64 solve

@testset "solve with dual numbers" begin
    D = IMAS.InputsDual{2}
    dd0 = IMAS.json2imas(joinpath(@__DIR__, "dd_d3d.json"))
    fw_r, fw_z = IMAS.first_wall(dd0.wall)
    ΔR = maximum(fw_r) - minimum(fw_r)
    ΔZ = maximum(fw_z) - minimum(fw_z)

    # s[1] scales the pressure, s[2] the toroidal current density
    function run(s)
        T = typeof(s[1] * s[2])
        dd = T === Float64 ? deepcopy(dd0) : IMAS.retype(T, dd0)
        e1 = dd.equilibrium.time_slice[].profiles_1d
        e1.pressure = e1.pressure .* s[1]
        e1.j_tor = e1.j_tor .* s[2]
        Rs = range(T(max(0.01, minimum(fw_r) - ΔR / 20)), T(maximum(fw_r) + ΔR / 20), 33)
        Zs = range(T(minimum(fw_z) - ΔZ / 20), T(maximum(fw_z) + ΔZ / 20), 65)
        canvas = Canvas(dd, Rs, Zs)
        profile = FRESCO.PressureJt(dd; grid=:rho_tor_norm)
        status = FRESCO.solve!(canvas, profile, 200, 3; relax=0.5, control=:shape, tolerance=1e-9)
        return status, [canvas.Raxis, canvas.Ψbnd - canvas.Ψaxis, canvas.Ip, canvas.Ψ[17, 33] - canvas.Ψaxis]
    end

    s0, y0 = run([1.0, 1.0])
    sd, yd = run([IMAS.seed(D, 1), IMAS.seed(D, 2)])
    @test s0 == 0 && sd == 0
    @test eltype(yd) === D
    @test value.(yd) ≈ y0 rtol = 1e-6
    h = 1e-3
    for k in 1:2
        e = [Float64(k == 1), Float64(k == 2)]
        fd = (run(1.0 .+ h .* e)[2] .- run(1.0 .- h .* e)[2]) ./ (2h)
        @test partials.(yd, k) ≈ fd rtol = 2e-2 atol = 2e-3 * maximum(abs.(y0))
    end
end
