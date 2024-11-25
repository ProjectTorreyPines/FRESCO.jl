function invert_GS!(canvas::Canvas; reset_boundary_flux=false, update_Ψitp::Bool=false)

    reset_boundary_flux && set_boundary_flux!(canvas)
    Rs, Zs, Ψpl, Jt, a, b, c = canvas.Rs, canvas.Zs, canvas._Ψpl, canvas._Jt, canvas._a, canvas._b, canvas._c
    MST, u, A, B, M, LU, S, tmp = canvas._MST, canvas._u, canvas._A, canvas._B, canvas._M, canvas._LU, canvas._S, canvas._tmp_Ψ

    # Here each N is subtracted by one, as the implemented algorithm assumes 0:N indexing
    Nr = length(Rs) - 1
    Nz = length(Zs) - 1
    hr = Base.step(Rs)
    hz = Base.step(Zs)
    hr2 = hr ^ 2
    hz2 = hz ^ 2
    hr2_hz2 = hr2 / hz2

    @views A .= Ψpl[:, 1]
    @views B .= (Ψpl[:, end] .- A) ./ Nz

    M .= 0.0
    Dl = M.dl
    D  = M.d
    Du = M.du

    @. tmp = twopi * μ₀ * Jt
    for j in 0:Nz
        for i in 1:Nr-1
            tmp[i+1, j+1] *= Rs[i+1]
            tmp[i+1, j+1] -= (a[i+1] * (A[i+2]  + j * B[i+2]) - b[i+1]* (A[i+1]  + j * B[i+1]) + c[i+1] * (A[i]  + j * B[i])) / hr2
        end
        tmp[1,   j+1] = Ψpl[1,   j+1] - (A[1]   + j * B[1])
        tmp[end, j+1] = Ψpl[end, j+1] - (A[end] + j * B[end])
    end
    mul!(S, tmp, MST)

    pi_Nz = π / Nz

    D[1] = 1.0
    D[end] = 1.0
    @. @views Dl[1:end-1] = c[2:end-1] / hr2
    @. @views Du[2:end] = a[2:end-1] / hr2

    for k in 0:Nz
        tmp = 2  * hr2_hz2 * (1.0 - cos(k * pi_Nz))
        @. @views D[2:end-1] = -(b[2:end-1] + tmp) / hr2
        lu!(LU, M)
        @views ldiv!(u[:, k+1], LU, S[:, k+1])
    end

    mul!(Ψpl, u, MST)
    for j in 0:Nz
        @. @views  Ψpl[:, j+1] += A + j * B
    end

    sync_Ψ!(canvas; update_Ψitp)

    return canvas
end

# Invert GS with U=0 on boundary -- needed for von Hagenow boundary integral
function invert_GS_zero_bnd!(canvas::Canvas, Jtor::Union{Nothing,Function})
    gridded_Jtor!(canvas, Jtor)
    invert_GS_zero_bnd!(canvas)
end

function invert_GS_zero_bnd!(canvas::Canvas)
    Rs, Zs, U, Jt, a, b, c = canvas.Rs, canvas.Zs, canvas._U, canvas._Jt, canvas._a, canvas._b, canvas._c
    MST, u, M, LU, S, tmp = canvas._MST, canvas._u, canvas._M, canvas._LU, canvas._S, canvas._tmp_Ψ

    Nr = length(Rs) - 1
    Nz = length(Zs) - 1
    hr = (Rs[end] - Rs[1]) / Nr
    hz = (Zs[end] - Zs[1]) / Nz
    hr2 = hr ^ 2
    hz2 = hz ^ 2
    hr2_hz2 = hr2 / hz2

    M .= 0.0
    Dl = M.dl
    D  = M.d
    Du = M.du

    @. tmp = twopi * μ₀ * Jt
    for (i, r) in enumerate(Rs)
        tmp[i, :] .*= r
    end
    tmp[1,:] .= 0.0
    tmp[end, :] .= 0.0
    mul!(S, tmp, MST)

    pi_Nz = π / Nz

    D[1] = 1.0
    D[end] = 1.0
    @. @views Dl[1:end-1] = c[2:end-1] / hr2
    @. @views Du[2:end] = a[2:end-1] / hr2

    for k in 0:Nz
        tmp = 2  * hr2_hz2 * (1.0 - cos(k * pi_Nz))
        @. @views D[2:end-1] = -(b[2:end-1] + tmp) / hr2
        lu!(LU, M)
        @views ldiv!(u[:, k+1], LU, S[:, k+1])
    end

    mul!(U, u, MST)

    return canvas
end