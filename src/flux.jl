coil_flux(r, z, canvas::Canvas) = sum(VacuumFields.current(coil) != 0.0 ? VacuumFields.ψ(coil, r, z) : 0.0 for coil in canvas.coils)

function sync_Ψ!(canvas::Canvas; update_vacuum::Bool=false, update_Ψitp::Bool=true)
    update_vacuum && set_Ψvac!(canvas)
    @. canvas.Ψ = canvas._Ψpl + canvas._Ψvac
    update_Ψitp && update_interpolation!(canvas)
    return canvas
end

function set_Ψvac!(canvas::Canvas)
    Rs, Zs, coils, Ψvac, Gvac = canvas.Rs, canvas.Zs, canvas.coils, canvas._Ψvac, canvas._Gvac
    Ψvac .= 0.0
    for (k, coil) in enumerate(coils)
        Ic = VacuumFields.current(coil)
        @views Ψvac .+= Ic * Gvac[:,:,k]
    end
    Ψvac .*= 2π * μ₀
    return canvas
end

function set_boundary_flux!(canvas::Canvas, Jtor::Union{Nothing,Function}, relax = 1.0)
    gridded_Jtor!(canvas, Jtor)
    return set_boundary_flux!(canvas, relax)
end

function set_boundary_flux!(canvas::Canvas, relax = 1.0)
    Rs, Zs, Ψpl = canvas.Rs, canvas.Zs, canvas._Ψpl
    Ψpl[:, 1]   .= (1.0 - relax) * Ψpl[:, 1]   + relax * flux.(eachindex(Rs), :bottom, Ref(canvas))
    Ψpl[:, end] .= (1.0 - relax) * Ψpl[:, end] + relax * flux.(eachindex(Rs), :top,    Ref(canvas))
    Ψpl[1, :]   .= (1.0 - relax) * Ψpl[1, :]   + relax * flux.(eachindex(Zs), :left,   Ref(canvas))
    Ψpl[end, :] .= (1.0 - relax) * Ψpl[end, :] + relax * flux.(eachindex(Zs), :right,  Ref(canvas))
    return canvas
end

function invert_GS!(canvas::Canvas, Jtor::Union{Nothing,Function})
    set_boundary_flux!(canvas, Jtor)
    invert_GS!(canvas)
end

# compute flux at (x, y) using 2D surface integral over plasma current
function flux_2D(x::Real, y::Real, canvas::Canvas)
    Rs, Zs, Jt = canvas.Rs, canvas.Zs, canvas._Jt
    coeff = step(Rs) * step(Zs) * twopi * μ₀
    return coeff * sum(Green(x, y, Rs[i], Zs[j]) * Jt[i, j] for j in eachindex(Zs)[2:end-1], i in eachindex(Rs)[2:end-1])
end

# Compute flux at (x, y) outside of plasma using von Hagenow boundary integral
function flux(x::Real, y::Real, canvas::Canvas)
    Rs, Zs = canvas.Rs, canvas.Zs
    @assert (x < Rs[1]) || (x > Rs[end]) || (y < Zs[1]) || (y > Zs[end])
    U = canvas._U

    dr, dz = step(Rs), step(Zs)

    Rl, Rr = Rs[1], Rs[end]
    inv_Rl, inv_Rr = 1.0 / Rl, 1.0 / Rr
    Zb, Zt = Zs[1], Zs[end]

    horizontal_integrand = i -> ((U[i, 3]     - 4 * U[i, 2])     * Green(x, y, Rs[i], Zb) +
                                    (U[i, end-2] - 4 * U[i, end-1]) * Green(x, y, Rs[i], Zt)) / Rs[i]
    psi  = (0.5 * dr / dz) * sum(horizontal_integrand(i) for i in eachindex(Rs)[2:end-1])

    vertical_integrand = j -> ((U[end-2, j] - 4 * U[end-1, j]) * Green(x, y, Rr, Zs[j]) * inv_Rr +
                                (U[3, j]     - 4 * U[2, j])     * Green(x, y, Rl, Zs[j]) * inv_Rl)
    psi += (0.5 * dz / dr) * sum(vertical_integrand(j) for j in eachindex(Zs)[2:end-1])

    return psi
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
    x = (bnd in (:top, :bottom)) ? Rs[k] : ((bnd === :left)   ? Rs[1] : Rs[end])
    y = (bnd in (:left, :right)) ? Zs[k] : ((bnd === :bottom) ? Zs[1] : Zs[end])

    U = canvas._U
    dr, dz = step(Rs), step(Zs)

    Rl, Rr = Rs[1], Rs[end]
    inv_Rl, inv_Rr = 1.0 / Rl, 1.0 / Rr
    Zb, Zt = Zs[1], Zs[end]

    left_integrand   = j -> (bnd === :left   && j == k) ? 0.0 : (U[3, j]     - 4 * U[2, j])     * Green(x, y, Rl, Zs[j]) * inv_Rl
    right_integrand  = j -> (bnd === :right  && j == k) ? 0.0 : (U[end-2, j] - 4 * U[end-1, j]) * Green(x, y, Rr, Zs[j]) * inv_Rr
    bottom_integrand = i -> (bnd === :bottom && i == k) ? 0.0 : (U[i, 3]     - 4 * U[i, 2])     * Green(x, y, Rs[i], Zb) / Rs[i]
    top_integrand    = i -> (bnd === :top    && i == k) ? 0.0 : (U[i, end-2] - 4 * U[i, end-1]) * Green(x, y, Rs[i], Zt) / Rs[i]

    psi  = (0.5 * dz / dr) * sum(left_integrand(j) + right_integrand(j) for j in eachindex(Zs)[2:end-1])
    psi += (0.5 * dr / dz) * sum(top_integrand(i) + bottom_integrand(i) for i in eachindex(Rs)[2:end-1])
    psi += self_field(k, bnd, canvas)

    # Subtract out half of neighboring points (boundary of trapezoidal rule)
    # Don't do at corners since self_field is zero there
    if bnd === :left
        (k > 1 && k < length(Zs)) && (psi -= (0.25 * dz / dr) * (left_integrand(k-1) + left_integrand(k+1)))
    elseif bnd === :right
        (k > 1 && k < length(Zs)) && (psi -= (0.25 * dz / dr) * (right_integrand(k-1) + right_integrand(k+1)))
    elseif bnd === :bottom
        (k > 1 && k < length(Rs)) && (psi -= (0.25 * dr / dz) * (bottom_integrand(k-1) + bottom_integrand(k+1)))
    else #if bnd === :top
        (k > 1 && k < length(Rs)) && (psi -= (0.25 * dr / dz) * (top_integrand(k-1) + top_integrand(k+1)))
    end

    return psi
end


function find_axis(canvas::Canvas; update_Ψitp::Bool=true)
    update_Ψitp && update_interpolation!(canvas)
    Rs, Zs, Ψ, Ip, Ψitp = canvas.Rs, canvas.Zs, canvas.Ψ, canvas.Ip, canvas._Ψitp
    psisign = sign(Ip)

    # find initial guess
    ia, ja = 0, 0
    found = false
    for j in eachindex(Zs)[2:end-1]
        for i in eachindex(Rs)[2:end-1]
            P  = psisign * Ψ[i, j]
            Pl = psisign * Ψ[i-1, j]
            Pr = psisign * Ψ[i+1, j]
            Pd = psisign * Ψ[i, j-1]
            Pu = psisign * Ψ[i, j+1]
            if P < Pl && P < Pr && P < Pd && P < Pu
                if !found
                    ia, ja = i, j
                else
                    throw("Found multiple minimum Ψ points")
                end
            end
        end
    end

    Rg, Zg = Rs[ia], Zs[ja]
    Raxis, Zaxis = IMAS.find_magnetic_axis(Rs, Zs, Ψitp, psisign; rguess=Rg, zguess=Zg)
    return Raxis, Zaxis, Ψitp(Raxis, Zaxis)
end

function flux_bounds!(canvas::Canvas; update_Ψitp::Bool=true)
    update_Ψitp && update_interpolation!(canvas)
    Rs, Zs, Ψ, Rw, Zw, Ψbnd, Ψitp = canvas.Rs, canvas.Zs, canvas.Ψ, canvas.Rw, canvas.Zw, canvas.Ψbnd, canvas._Ψitp
    Raxis, Zaxis, Ψaxis = find_axis(canvas; update_Ψitp=false)
    # BCL 7/31/24 - Potential error here with Ψbnd as that's supposed to be the "original" psi on the boundary
    #               I'm not sure how to handle that, though hopefully it gets fixed on the IMAS side
    axis2bnd = (canvas.Ip >= 0.0) ? :increasing : :decreasing
    Ψbnd = IMAS.find_psi_boundary(Rs, Zs, Ψ, Ψaxis, axis2bnd, Raxis, Zaxis, Rw, Zw; PSI_interpolant=Ψitp, raise_error_on_not_open=false, raise_error_on_not_closed=false).last_closed
    canvas.Raxis, canvas.Zaxis, canvas.Ψaxis, canvas.Ψbnd = Raxis, Zaxis, Ψaxis, Ψbnd
end

psinorm(psi::Real, canvas::Canvas) = (psi - canvas.Ψaxis) / (canvas.Ψbnd - canvas.Ψaxis)

function boundary!(canvas::Canvas)
    Rs, Zs, Ψ, Raxis, Zaxis, Ψbnd, is_inside  = canvas.Rs, canvas.Zs, canvas.Ψ, canvas.Raxis, canvas.Zaxis, canvas.Ψbnd, canvas._is_inside
    B = IMAS.flux_surface(Rs, Zs, Ψ, Raxis, Zaxis, Float64[], Float64[], Ψbnd, :closed)
    @assert length(B) == 1
    r, z =  B[1].r, B[1].z
    canvas._bnd = [@SVector[r[k], z[k]] for k in eachindex(r)]
    canvas._rextrema = extrema(r)
    canvas._zextrema = extrema(z)

    ellipse = compute_ellipse(canvas)
    for (j, z) in enumerate(Zs)
        for (i, r) in enumerate(Rs)
            psin = psinorm(Ψ[i, j], canvas)
            is_inside[i, j] = in_core(r, z, psin, canvas, ellipse)
        end
    end
end

function update_bounds!(canvas; update_Ψitp::Bool=true)
    flux_bounds!(canvas; update_Ψitp)
    boundary!(canvas)
    return canvas
end

function in_core(r::Real, z::Real, psin::Real, canvas::Canvas,
                 ellipse::Union{Nothing, AbstractVector{<:Real}}=nothing)

    # Check psinorm value
    psin > 1.0 && return false

    # Check outside bounding box
    rmin, rmax = canvas._rextrema
    (r < rmin || r > rmax) && return false

    zmin, zmax = canvas._zextrema
    (z < zmin || z > zmax) && return false

    in_ellipse(r, z, ellipse) && return true

    # Finally make sure it's in the boundary
    return inpolygon((r, z), canvas._bnd) == 1
end

in_ellipse(r, z, ellipse::Nothing) = false

function in_ellipse(r::Real, z::Real, ellipse::AbstractVector{<:Real})
    @assert length(ellipse) == 4
    R0, Z0, a, b = ellipse
    return ((r - R0) / a) ^ 2 + ((z - Z0) / b) ^ 2 <= 1.0
end

function compute_ellipse(canvas::Canvas)
    bnd, rext, zext = canvas._bnd, canvas._rextrema, canvas._zextrema
    κ = (zext[2] - zext[1]) / (rext[2] - rext[1])
    R0, Z0 = centroid(bnd)
    radius = p -> sqrt((p[1] - R0) ^ 2 + ((p[2] - Z0) / κ) ^ 2)
    a = minimum(radius, bnd)
    b = κ * a
    return @SVector[R0, Z0, a, b]
end