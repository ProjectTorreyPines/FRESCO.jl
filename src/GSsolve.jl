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
        U = canvas.Rs, canvas.Zs

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

function invert_GS!(canvas::Canvas, Ψvac::Function, Jtor::Union{Nothing,Function}=nothing)

    Rs, Zs, Ψ, Jt, a, b, c, MST, u, A, B, M, S = canvas.Rs, canvas.Zs, canvas.Ψ, canvas._Jt, canvas._a, canvas._b, canvas._c, canvas._MST, canvas._u, canvas._A, canvas._B, canvas._M, canvas._S

    gridded_Jtor!(canvas, Jtor)
    include_Jt = any(J !== 0.0 for J in canvas._Jt)

    Nr = length(Rs) - 1
    Nz = length(Zs) - 1
    hr = (Rs[end] - Rs[1]) / Nr
    hz = (Zs[end] - Zs[1]) / Nz
    hr2 = hr ^ 2
    hz2 = hz ^ 2
    hr2_hz2 = hr2 / hz2

    A .=  flux.(eachindex(Rs), :bottom, Ref(canvas), Ψvac; include_Jt)
    B .= (flux.(eachindex(Rs), :top,    Ref(canvas), Ψvac; include_Jt) .- A) ./ Nz

    Dl = M.dl
    D  = M.d
    Du = M.du

    @. S = twopi * μ₀ * Jt
    for j in 0:Nz
        for i in 1:Nr-1
            S[i+1, j+1] *= Rs[i+1]
            S[i+1, j+1] -= (a[i+1] * (A[i+2]  + j * B[i+2]) - b[i+1]* (A[i+1]  + j * B[i+1]) + c[i+1] * (A[i]  + j * B[i])) / hr2
        end
        S[1, j+1]   = flux(j+1, :left,  canvas, Ψvac; include_Jt) - (A[1]   + j * B[1])
        S[end, j+1] = flux(j+1, :right, canvas, Ψvac; include_Jt) - (A[end] + j * B[end])
    end
    S *= MST

    pi_Nz = π / Nz

    D[1] = 1.0
    D[end] = 1.0
    @. @views Dl[1:end-1] = c[2:end-1] / hr2
    @. @views Du[2:end] = a[2:end-1] / hr2

    for k in 0:Nz
        tmp = 2  * hr2_hz2 * (1.0 - cos(k * pi_Nz))
        @. @views D[2:end-1] = -(b[2:end-1] + tmp) / hr2
        u[:, k+1] .= M \ S[:, k+1]
    end

    mul!(Ψ, u, MST)
    for j in 0:Nz
        @. @views  Ψ[:, j+1] += A + j * B
    end

    return canvas
end

# Invert GS with U=0 on boundary -- needed for von Hagenow boundary integral
function invert_GS_zero_bnd!(canvas::Canvas, Jtor::Union{Nothing,Function}=nothing)

    Rs, Zs, U, Jt, a, b, c, MST, u, M, S = canvas.Rs, canvas.Zs, canvas._U, canvas._Jt, canvas._a, canvas._b, canvas._c, canvas._MST, canvas._u, canvas._M, canvas._S

    gridded_Jtor!(canvas, Jtor)

    Nr = length(Rs) - 1
    Nz = length(Zs) - 1
    hr = (Rs[end] - Rs[1]) / Nr
    hz = (Zs[end] - Zs[1]) / Nz
    hr2 = hr ^ 2
    hz2 = hz ^ 2
    hr2_hz2 = hr2 / hz2

    Dl = M.dl
    D  = M.d
    Du = M.du

    @. S = twopi * μ₀ * Jt
    for (i, r) in enumerate(Rs)
        S[i, :] .*= r
    end
    S[1,:] .= 0.0
    S[end, :] .= 0.0
    S *= MST

    pi_Nz = π / Nz

    D[1] = 1.0
    D[end] = 1.0
    @. @views Dl[1:end-1] = c[2:end-1] / hr2
    @. @views Du[2:end] = a[2:end-1] / hr2

    for k in 0:Nz
        tmp = 2  * hr2_hz2 * (1.0 - cos(k * pi_Nz))
        @. @views D[2:end-1] = -(b[2:end-1] + tmp) / hr2
        u[:, k+1] .= M \ S[:, k+1]
    end

    mul!(U, u, MST)

    return canvas
end