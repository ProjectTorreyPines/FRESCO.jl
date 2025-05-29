function sync_Ψ!(canvas::Canvas; update_vacuum::Bool=false, update_Ψitp::Bool=true)
    update_vacuum && set_Ψvac!(canvas)
    @. canvas.Ψ = canvas._Ψpl + canvas._Ψvac
    update_Ψitp && update_interpolation!(canvas)
    return canvas
end

function set_Ψvac!(canvas::Canvas)
    Rs, Zs, coils, Ψvac, Gvac = canvas.Rs, canvas.Zs, canvas.coils, canvas._Ψvac, canvas._Gvac
    VacuumFields.flux_on_grid!(Ψvac, Gvac, Rs, Zs, coils)
    return canvas
end

function set_boundary_flux!(canvas::Canvas, Jtor::Union{Nothing,Function}, relax=1.0)
    gridded_Jtor!(canvas, Jtor)
    return set_boundary_flux!(canvas, relax)
end

function set_boundary_flux!(canvas::Canvas, relax=1.0)
    Rs, Zs, Ψpl = canvas.Rs, canvas.Zs, canvas._Ψpl
    Nr, Nz = length(Rs), length(Zs)
    Nb = Nbnd(Nr, Nz)
    for k in 1:Nb
        i, j = bnd2mat(Nr, Nz, k)
        @inbounds Ψpl[i, j] = (1.0 - relax) * Ψpl[i, j] + relax * bndflux(k, canvas)
    end
    return canvas
end

function invert_GS!(canvas::Canvas, Jtor::Union{Nothing,Function})
    set_boundary_flux!(canvas, Jtor)
    return invert_GS!(canvas)
end

# Approximate integral from k-1 to k+1 along boundary bnd
# Based on DeLucia, Jardin, & Todd (1980) and King & Jardin (1985)
# K.S. Han et al. (2021) also useful
function self_field(k::Int, bnd::Symbol, canvas::Canvas)
    @assert bnd in (:left, :right, :top, :bottom)
    Rs, Zs, U = canvas.Rs, canvas.Zs, canvas._U
    dr, dz = step(Rs), step(Zs)
    if bnd === :left
        x = Rs[1]
        h = dz
        dUdn = (U[3, k] - 4 * U[2, k]) / (2dr)
    elseif bnd === :right
        x = Rs[end]
        h = dz
        dUdn = (U[end-2, k] - 4 * U[end-1, k]) / (2dr)
    elseif bnd === :bottom
        x = Rs[k]
        h = dr
        dUdn = (U[k, 3] - 4 * U[k, 2]) / (2dz)
    else #if bnd === :top
        x = Rs[k]
        h = dr
        dUdn = (U[k, end-2] - 4 * U[k, end-1]) / (2dz)
    end

    return h * dUdn * (1.0 + log(0.125 * h / x)) / π
end

# Compute flux at point k on boundary bnd using von Hagenow boundary integral and self-field approximation at k
# Future work: speed up by storing the Green's function between every boundary point and point k (size 4N²)
function flux(k::Int, bnd::Symbol, canvas::Canvas)
    @assert bnd in (:left, :right, :top, :bottom)
    Rs = canvas.Rs
    Zs = canvas.Zs
    x = (bnd in (:top, :bottom)) ? Rs[k] : ((bnd === :left) ? Rs[1] : Rs[end])
    y = (bnd in (:left, :right)) ? Zs[k] : ((bnd === :bottom) ? Zs[1] : Zs[end])

    U = canvas._U
    dr, dz = step(Rs), step(Zs)

    Rl, Rr = Rs[1], Rs[end]
    inv_Rl, inv_Rr = 1.0 / Rl, 1.0 / Rr
    Zb, Zt = Zs[1], Zs[end]

    left_integrand = j -> (bnd === :left && j == k) ? 0.0 : (U[3, j] - 4 * U[2, j]) * VacuumFields.Green(x, y, Rl, Zs[j]) * inv_Rl
    right_integrand = j -> (bnd === :right && j == k) ? 0.0 : (U[end-2, j] - 4 * U[end-1, j]) * VacuumFields.Green(x, y, Rr, Zs[j]) * inv_Rr
    bottom_integrand = i -> (bnd === :bottom && i == k) ? 0.0 : (U[i, 3] - 4 * U[i, 2]) * VacuumFields.Green(x, y, Rs[i], Zb) / Rs[i]
    top_integrand = i -> (bnd === :top && i == k) ? 0.0 : (U[i, end-2] - 4 * U[i, end-1]) * VacuumFields.Green(x, y, Rs[i], Zt) / Rs[i]

    psi = (0.5 * dz / dr) * sum(left_integrand(j) + right_integrand(j) for j in eachindex(Zs)[2:end-1])
    psi += (0.5 * dr / dz) * sum(top_integrand(i) + bottom_integrand(i) for i in eachindex(Rs)[2:end-1])
    psi += self_field(k, bnd, canvas)

    # Subtract out half of neighboring points (boundary of trapezoidal rule)
    # Don't do at corners since self_field is zero there
    if bnd === :left
        (k > 1 && k < length(Zs)) && (psi -= (0.25 * dz / dr) * (left_integrand(k - 1) + left_integrand(k + 1)))
    elseif bnd === :right
        (k > 1 && k < length(Zs)) && (psi -= (0.25 * dz / dr) * (right_integrand(k - 1) + right_integrand(k + 1)))
    elseif bnd === :bottom
        (k > 1 && k < length(Rs)) && (psi -= (0.25 * dr / dz) * (bottom_integrand(k - 1) + bottom_integrand(k + 1)))
    else #if bnd === :top
        (k > 1 && k < length(Rs)) && (psi -= (0.25 * dr / dz) * (top_integrand(k - 1) + top_integrand(k + 1)))
    end

    return psi
end

function bndflux(k::Int, canvas::Canvas)
    Rs, Zs, Gbnd, U = canvas.Rs, canvas.Zs, canvas._Gbnd, canvas._U
    Nr, Nz = length(Rs), length(Zs)
    Nb = Nbnd(Nr, Nz)

    @assert size(Gbnd) == (Nb, Nb)
    @assert 1 <= k <= Nb

    ki, kj = bnd2mat(Nr, Nz, k)
    if kj == 1
        bnd = :bottom
        kcorner = ki in (1, Nr)
        psi = self_field(ki, bnd, canvas)
    elseif ki == Nr
        bnd = :right
        kcorner = kj in (1, Nz)
        psi = self_field(kj, bnd, canvas)
    elseif kj == Nz
        bnd = :top
        kcorner = ki in (1, Nr)
        psi = self_field(ki, bnd, canvas)
    elseif ki == 1
        bnd = :left
        kcorner = kj in (1, Nz)
        psi = self_field(kj, bnd, canvas)
    else
        @error "Problem indexing boundary"
    end

    dr, dz = Base.step(Rs), Base.step(Zs)
    inv_Rl, inv_Rr = 1.0 / Rs[1], 1.0 / Rs[end]
    hfac = 0.5 * dr / dz
    vfac = 0.5 * dz / dr
    vfac_Rr = vfac * inv_Rr
    vfac_Rl = vfac * inv_Rl
    for l in 1:Nb
        l == k && continue
        li, lj = bnd2mat(Nr, Nz, l)
        @assert (1 <= li <= Nr) && (1 <= lj <= Nz)
        if lj == 1
            # bottom
            li in (1, Nr) && continue # corner
            @inbounds psil = hfac * (U[li, 3] - 4 * U[li, 2]) * Gbnd[l, k] / Rs[li]

        elseif li == Nr
            # right
            lj in (1, Nz) && continue # corner
            @inbounds psil = vfac_Rr * (U[end-2, lj] - 4 * U[end-1, lj]) * Gbnd[l, k]
        elseif lj == Nz
            # top
            li in (1, Nr) && continue # corner
            @inbounds psil = hfac *  (U[li, end-2] - 4 * U[li, end-1]) * Gbnd[l, k] / Rs[li]
        elseif li == 1
            # left
            lj in (1, Nz) && continue # corner
            @inbounds psil = vfac_Rl * (U[3, lj] - 4 * U[2, lj]) * Gbnd[l, k]
        else
            @error "Problem indexing boundary"
        end
        if !kcorner && (l in (k-1, k+1))
            # Remove half of neighboring points (boundary of trapezoidal rule)
            #   to properly account for self field
            # Don't do at corners since self_field is zero there
            psil *= 0.5
        end

        psi += psil

    end

    return psi
end

##################
# Plasma flux
##################


coil_flux(canvas::Canvas, x::Real, y::Real) = _cfunc(VacuumFields.ψ, canvas, x, y)
coil_dψdR(canvas::Canvas, x::Real, y::Real) = _cfunc(VacuumFields.dψ_dR, canvas, x, y)
coil_dψdZ(canvas::Canvas, x::Real, y::Real) = _cfunc(VacuumFields.dψ_dZ, canvas, x, y)
_cfunc(Pfunc::Function, canvas::Canvas, x::Real, y::Real) = sum(VacuumFields.current_per_turn(coil) != 0.0 ? Pfunc(coil, x, y) : 0.0 for coil in canvas.coils)

function in_domain(r::Real, z::Real, canvas::Canvas)
    Rs, Zs = canvas.Rs, canvas.Zs
    return (r >= Rs[1]) && (r <= Rs[end]) && (z >= Zs[1]) && (z <= Zs[end])
end

plasma_flux_external(canvas::Canvas, x::Real, y::Real) = _pfunc(VacuumFields.Green, canvas, x, y)
plasma_dψdR_external(canvas::Canvas, x::Real, y::Real) = _pfunc(VacuumFields.dG_dR, canvas, x, y)
plasma_dψdZ_external(canvas::Canvas, x::Real, y::Real) = _pfunc(VacuumFields.dG_dZ, canvas, x, y)
function _pfunc(Gfunc, canvas::Canvas, x::Real, y::Real)
    # von Hagenow method
    Rs, Zs = canvas.Rs, canvas.Zs

    U = canvas._U

    dr, dz = step(Rs), step(Zs)

    Rl, Rr = Rs[1], Rs[end]
    inv_Rl, inv_Rr = 1.0 / Rl, 1.0 / Rr
    Zb, Zt = Zs[1], Zs[end]

    horizontal_integrand = i -> ((U[i, 3] - 4 * U[i, 2]) * Gfunc(Rs[i], Zb, x, y) +
                                (U[i, end-2] - 4 * U[i, end-1]) * Gfunc(Rs[i], Zt, x, y)) / Rs[i]
    psi = (0.5 * dr / dz) * sum(horizontal_integrand(i) for i in eachindex(Rs)[2:end-1])

    vertical_integrand = j -> ((U[end-2, j] - 4 * U[end-1, j]) * Gfunc(Rr, Zs[j], x, y) * inv_Rr +
                            (U[3, j] - 4 * U[2, j]) * Gfunc(Rl, Zs[j], x, y) * inv_Rl)
    psi += (0.5 * dz / dr) * sum(vertical_integrand(j) for j in eachindex(Zs)[2:end-1])

    return psi
end

function plasma_flux(canvas::Canvas, x::Real, y::Real, blank::Nothing=nothing)
    return in_domain(x, y, canvas) ? canvas._Ψitp(x, y) - coil_flux(canvas, x, y) : plasma_flux_external(canvas, x, y)
end
function plasma_flux(canvas::Canvas, x::Real, y::Real, Ψpl_itp::Interpolations.AbstractInterpolation)
    return in_domain(x, y, canvas) ? Ψpl_itp(x, y) : plasma_flux_external(canvas, x, y)
end

function plasma_dψdR(canvas::Canvas, x::Real, y::Real, blank::Nothing=nothing)
    ∇ = Interpolations.gradient
    return in_domain(x, y, canvas) ? ∇(canvas._Ψitp, x, y)[1] - coil_dψdR(canvas, x, y) : plasma_dψdR_external(canvas, x, y)
end
function plasma_dψdR(canvas::Canvas, x::Real, y::Real, Ψpl_itp::Interpolations.AbstractInterpolation)
    ∇ = Interpolations.gradient
    return in_domain(x, y, canvas) ? ∇(Ψpl_itp, x, y)[1] : plasma_dψdR_external(canvas, x, y)
end

function plasma_dψdZ(canvas::Canvas, x::Real, y::Real, blank::Nothing=nothing)
    ∇ = Interpolations.gradient
    return in_domain(x, y, canvas) ? ∇(canvas._Ψitp, x, y)[2] - coil_dψdZ(canvas, x, y) : plasma_dψdZ_external(canvas, x, y)
end
function plasma_dψdZ(canvas::Canvas, x::Real, y::Real, Ψpl_itp::Interpolations.AbstractInterpolation)
    ∇ = Interpolations.gradient
    return in_domain(x, y, canvas) ? ∇(Ψpl_itp, x, y)[2] : plasma_dψdZ_external(canvas, x, y)
end

function flux(canvas::Canvas, x::Real, y::Real; update_Ψitp::Bool=true)

    if in_domain(x, y, canvas)
        update_Ψitp && update_interpolation!(canvas)
        return canvas._Ψitp(x, y)
    else
        return plasma_flux_external(canvas, x, y) + coil_flux(canvas, x, y)
    end
end

# compute flux at (x, y) using 2D surface integral over plasma current
# For diagnostics only
function plasma_flux_2D(canvas::Canvas, x::Real, y::Real)
    Rs, Zs, Jt = canvas.Rs, canvas.Zs, canvas._Jt
    coeff = step(Rs) * step(Zs) * twopi * μ₀
    f = (i, j) -> (Jt[i, j] == 0.0) ? 0.0 : VacuumFields.Green(x, y, Rs[i], Zs[j]) * Jt[i, j]
    return coeff * sum(f(i, j) for j in eachindex(Zs)[2:end-1], i in eachindex(Rs)[2:end-1])
end

function plasma_flux_at_coil(k::Int, canvas::Canvas)
    Rs, Zs, coils, Gvac, Jt = canvas.Rs, canvas.Zs, canvas.coils, canvas._Gvac, canvas._Jt
    @assert k in eachindex(coils)
    Gv = @view(Gvac[:, :, k])
    flux = zero(eltype(Jt))
    @turbo for j in eachindex(Zs)
        for i in eachindex(Rs)
            J = Jt[i, j]
            flux += (J == 0.0) ? J :  J * Gv[i, j]
        end
    end
    return (twopi * μ₀) * (step(Rs) * step(Zs) * flux)
end

function set_flux_at_coils!(canvas::Canvas)
    coils, Ψ_at_coils, mutuals = canvas.coils, canvas._Ψ_at_coils, canvas._mutuals

    # poloidal flux from one coil to another is -M * current_per_turn
    for i in eachindex(coils)
        Ψ_at_coils[i] = plasma_flux_at_coil(i, canvas)
        Ψ_at_coils[i] -= sum(mutuals[j, i] * VacuumFields.current_per_turn(coils[j]) for j in eachindex(coils))
    end
    return canvas
end