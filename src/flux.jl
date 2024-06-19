# compute flux at (x, y) using 2D surface integral over plasma current
function flux_2D(x::Real, y::Real, canvas::Canvas, Ψvac::Function; include_Jt::Bool=any(J !== 0.0 for J in canvas._Jt))
    psi = Ψvac(x, y)
    if include_Jt
        Rs, Zs, Jt = canvas.Rs, canvas.Zs, canvas._Jt
        coeff = step(Rs) * step(Zs) * twopi * μ₀
        psi += coeff * sum(Green(x, y, Rs[i], Zs[j]) * Jt[i, j] for j in eachindex(Zs)[2:end-1], i in eachindex(Rs)[2:end-1])
    end
    return psi
end

# Compute flux at (x, y) outside of plasma using von Hagenow boundary integral
function flux(x::Real, y::Real, canvas::Canvas, Ψvac::Function; include_Jt::Bool=any(J !== 0.0 for J in canvas._Jt))
    Rs, Zs = canvas.Rs, canvas.Zs
    @assert (x < Rs[1]) || (x > Rs[end]) || (y < Zs[1]) || (y > Zs[end])
    psi = Ψvac(x, y)
    if include_Jt
        U = canvas._U

        dr, dz = step(Rs), step(Zs)

        Rl, Rr = Rs[1], Rs[end]
        inv_Rl, inv_Rr = 1.0 / Rl, 1.0 / Rr
        Zb, Zt = Zs[1], Zs[end]

        horizontal_integrand = i -> ((U[i, 3]     - 4 * U[i, 2])     * Green(x, y, Rs[i], Zb) +
                                     (U[i, end-2] - 4 * U[i, end-1]) * Green(x, y, Rs[i], Zt)) / Rs[i]
        psi += (0.5 * dr / dz) * sum(horizontal_integrand(i) for i in eachindex(Rs)[2:end-1])

        vertical_integrand = j -> ((U[end-2, j] - 4 * U[end-1, j]) * Green(x, y, Rr, Zs[j]) * inv_Rr +
                                   (U[3, j]     - 4 * U[2, j])     * Green(x, y, Rl, Zs[j]) * inv_Rl)
        psi += (0.5 * dz / dr) * sum(vertical_integrand(j) for j in eachindex(Zs)[2:end-1])
    end

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
function flux(k::Int, bnd::Symbol, canvas::Canvas, Ψvac::Function; include_Jt::Bool=any(J !== 0.0 for J in canvas._Jt))
    @assert bnd in (:left, :right, :top, :bottom)
    Rs = canvas.Rs
    Zs = canvas.Zs
    x = (bnd in (:top, :bottom)) ? Rs[k] : ((bnd === :left)   ? Rs[1] : Rs[end])
    y = (bnd in (:left, :right)) ? Zs[k] : ((bnd === :bottom) ? Zs[1] : Zs[end])

    psi = Ψvac(x, y)

    if include_Jt
        U = canvas._U
        dr, dz = step(Rs), step(Zs)

        Rl, Rr = Rs[1], Rs[end]
        inv_Rl, inv_Rr = 1.0 / Rl, 1.0 / Rr
        Zb, Zt = Zs[1], Zs[end]

        left_integrand   = j -> (bnd === :left   && j == k) ? 0.0 : (U[3, j]     - 4 * U[2, j])     * Green(x, y, Rl, Zs[j]) * inv_Rl
        right_integrand  = j -> (bnd === :right  && j == k) ? 0.0 : (U[end-2, j] - 4 * U[end-1, j]) * Green(x, y, Rr, Zs[j]) * inv_Rr
        bottom_integrand = i -> (bnd === :bottom && i == k) ? 0.0 : (U[i, 3]     - 4 * U[i, 2])     * Green(x, y, Rs[i], Zb) / Rs[i]
        top_integrand    = i -> (bnd === :top    && i == k) ? 0.0 : (U[i, end-2] - 4 * U[i, end-1]) * Green(x, y, Rs[i], Zt) / Rs[i]

        psi += (0.5 * dz / dr) * sum(left_integrand(j) + right_integrand(j) for j in eachindex(Zs)[2:end-1])
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

    end

    return psi
end


function find_axis(canvas::Canvas)
    Rs, Zs, Ψ, Ip = canvas.Rs, canvas.Zs, canvas.Ψ, canvas.Ip
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
    PSI_interpolant = IMAS.ψ_interpolant(Rs, Zs, Ψ).PSI_interpolant
    Raxis, Zaxis = IMAS.find_magnetic_axis(Rs, Zs, PSI_interpolant, psisign; rguess=Rg, zguess=Zg)
    return Raxis, Zaxis, PSI_interpolant(Raxis, Zaxis)
end

function flux_bounds!(canvas::Canvas)
    Rs, Zs, Ψ = canvas.Rs, canvas.Zs, canvas.Ψ
    Raxis, Zaxis, Ψaxis = find_axis(canvas)
    PSI_interpolant = IMAS.ψ_interpolant(Rs, Zs, Ψ).PSI_interpolant
    Ψbnd = IMAS.find_psi_boundary(Rs, Zs, Ψ, Ψaxis, Raxis, Zaxis, Float64[], Float64[]; PSI_interpolant, raise_error_on_not_open=false, raise_error_on_not_closed=false).last_closed
    canvas.Raxis, canvas.Zaxis, canvas.Ψaxis, canvas.Ψbnd = Raxis, Zaxis, Ψaxis, Ψbnd
end

psinorm(psi::Real, canvas::Canvas) = (psi - canvas.Ψaxis) / (canvas.Ψbnd - canvas.Ψaxis)