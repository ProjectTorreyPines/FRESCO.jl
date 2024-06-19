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