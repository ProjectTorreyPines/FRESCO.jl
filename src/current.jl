# Initialize to uniform current in ellipse half the size of domain
function initial_current(canvas::Canvas, R::Real, Z::Real)
    Rs, Zs, Ip = canvas.Rs, canvas.Zs, canvas.Ip
    R0 = 0.5 * (Rs[end] + Rs[1])
    Z0 = 0.5 * (Zs[end] + Zs[1])
    a = 0.25 * (Rs[end] - Rs[1])
    b = 0.25 * (Zs[end] - Zs[1])
    erad = ((R - R0) / a) ^ 2 + ((Z - Z0) / b) ^ 2
    if erad <= 1.0
        return Ip / (π * a * b)
    else
        return 0.0
    end
end

function gridded_Jtor!(canvas::Canvas, Jtor::Nothing)
    canvas._Jt .= 0.0
    return canvas
end

function gridded_Jtor!(canvas::Canvas, Jtor)
    Rs, Zs, Jt = canvas.Rs, canvas.Zs, canvas._Jt
    Nr, Nz = length(Rs), length(Zs)
    for (j, z) in enumerate(Zs)
        if (j === 1) || (j === Nz + 1)
            Jt[:, j] .= 0.0
        else
            for (i, r) in enumerate(Rs)
                Jt[i, j] = ((i === 1) || (i === Nr + 1)) ? 0.0 : Jtor(r, z)
            end
        end
    end
    return canvas
end

# Based on https://arxiv.org/pdf/1503.03135
abstract type CurrentProfile end

mutable struct BetapIp{T} <: CurrentProfile
    betap::T
    alpha_m::T
    alpha_n::T
    Beta0::T
    L::T
end

function BetapIp(betap,alpha_m,alpha_n)
    if alpha_m < 0 || alpha_n < 0
        @error "alpha_m/n must be positive"
    end
    return BetapIp(betap,alpha_m, alpha_n,zero(betap),zero(betap))
end

function shape_integral(canvas::Canvas, profile::Union{BetapIp, PaxisIp}, psi_norm)
    psi_axis, psi_bdry = canvas.Ψaxis, canvas.Ψbnd
    I, _ = hquadrature(x -> (1 - x^profile.alpha_m)^profile.alpha_n, psi_norm, 1.0)
    return I*(psi_bdry - psi_axis)
end

function Jtor(canvas::Canvas, profile::BetapIp)
    psi_axis, psi_bdry = canvas.Ψaxis, canvas.Ψbnd
    Rs, Zs, Ψ = canvas.Rs, canvas.Zs, canvas.Ψ
    psi_norm = clamp.((Ψ .- psi_axis)/(psi_bdry - psi_axis), 0, 1)
    Raxis, Zaxis = canvas.Raxis, canvas.Zaxis
    Ip = canvas.Ip

    jtor_shape = (1.0 .- psi_norm.^profile.alpha_m).^profile.alpha_n

    pfunc = zero(Ψ)
    IR_mat = zero(Ψ)
    I_R_mat = zero(Ψ)
    for (i, r) in enumerate(Rs)
        for (j, z) in enumerate(Zs)
            psin = psi_norm[i,j]
            if 0.0 <= psin <= 1.0
                pfunc[i,j] = r*shape_integral(canvas, profile, psin)
                IR_mat[i,j] = jtor_shape[i,j]*r/Raxis
                I_R_mat[i,j] = jtor_shape[i,j]*Raxis/r
            end
        end
    end
    dR = step(Rs)
    dZ = step(Zs)

    #todo better integration
    p_int = sum(pfunc)*dR*dZ

    LBeta0 = (-profile.betap*(μ₀/(8*pi))*Raxis*Ip^2)/p_int

    #better integration
    IR = sum(IR_mat)*dR*dZ
    I_R = sum(I_R_mat)*dR*dZ

    L = Ip / I_R - LBeta0*(IR/I_R - 1)
    Beta0 = LBeta0/L

    Jtor=zero(Ψ)
    for (i,r) in enumerate(Rs)
        for (j,z) in enumerate(Zs)
            if 0.0 <= psi_norm[i,j] <= 1.0
                Jtor[i,j] = L * (Beta0 * r/Raxis + (1 - Beta0) * Raxis/r) * jtor_shape[i,j]
            end
        end
    end
    profile.L = L
    profile.Beta0 = Beta0
    return Jtor
end

mutable struct PaxisIp{T} <: CurrentProfile
    paxis::T
    alpha_m::T
    alpha_n::T
    L::T
    Beta0::T
end

function PaxisIp(paxis, alpha_m, alpha_n)
    if alpha_m < 0 || alpha_n < 0
        @error "alpha_m/n must be positive"
    end
    return PaxisIp(paxis, alpha_m, alpha_n, zero(paxis), zero(paxis))
end

function Jtor(canvas::Canvas, profile::PaxisIp)
    psi_axis, psi_bdry = canvas.Ψaxis, canvas.Ψbnd
    Rs, Zs, Ψ = canvas.Rs, canvas.Zs, canvas.Ψ
    psi_norm = clamp.((Ψ .- psi_axis)/(psi_bdry - psi_axis), 0, 1)
    Raxis, Zaxis = canvas.Raxis, canvas.Zaxis
    Ip = canvas.Ip

    jtor_shape = (1.0 .- psi_norm.^profile.alpha_m).^profile.alpha_n

    IR_mat = zero(Ψ)
    I_R_mat = zero(Ψ)
    for (i, r) in enumerate(Rs)
        for (j, z) in enumerate(Zs)
            psin = psi_norm[i,j]
            if 0.0 <= psin <= 1.0
                IR_mat[i,j] = jtor_shape[i,j]*r/Raxis
                I_R_mat[i,j] = jtor_shape[i,j]*Raxis/r
            end
        end
    end
    dR = step(Rs)
    dZ = step(Zs)

    #better integration
    IR = sum(IR_mat)*dR*dZ
    I_R = sum(I_R_mat)*dR*dZ

    LBeta0 = -profile.paxis * Raxis / shape_integral(canvas,profile,0.0)

    L = Ip / I_R - LBeta0 * (IR/I_R - 1)
    Beta0 = LBeta0/L

    Jtor=zero(Ψ)
    for (i,r) in enumerate(Rs)
        for (j,z) in enumerate(Zs)
            if 0.0 <= psi_norm[i,j] <= 1.0
                Jtor[i,j] = L * (Beta0 * r/Raxis + (1 - Beta0) * Raxis/r) * jtor_shape[i,j]
            end
        end
    end
    profile.L = L
    profile.Beta0 = Beta0
    return Jtor
end

function pprime(canvas::Canvas,p::Union{BetapIp,PaxisIp}, psi_norm)
    shape = (1 - clamp(psi_norm,0,1)^p.alpha_m)^p.alpha_n
    return (p.L*p.Beta0/canvas.Raxis)*shape
end

function ffprime(canvas::Canvas,p::Union{BetapIp,PaxisIp}, psi_norm)
    shape = (1 - clamp(psi_norm,0,1)^p.alpha_m)^p.alpha_n
    return μ₀*p.L*(1 - p.Beta0)*canvas.Raxis*shape
end


struct PprimeFFprime{F<:Function} <: CurrentProfile
    pprime::F
    ffprime::F
end

function pprime(canvas::Canvas, p::PprimeFFprime, psi_norm)
    p.pprime(psi_norm)
end

function ffprime(canvas::Canvas, p::PprimeFFprime, psi_norm)
    p.ffprime(psi_norm)
end

function Jtor(canvas::Canvas, profile::PprimeFFprime)

    psi_axis, psi_bdry = canvas.Ψaxis, canvas.Ψbnd
    psi_norm = clamp.((canvas.Ψ .- psi_axis)/(psi_bdry - psi_axis), 0, 1)

    Jt = zero(canvas.Ψ)
    for (i,R) in enumerate(canvas.Rs)
        for (j,Z) in enumerate(canvas.Zs)
            psin = psi_norm[i,j]
            if 0.0 <= psin <= 1.0
                Jt[i,j] = R*pprime(profile,psin) + ffprime(profile,psin)/(R*μ₀)
            end
        end
    end
    return Jt
end
