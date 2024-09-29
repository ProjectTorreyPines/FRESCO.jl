# Initialize to uniform current in ellipse half the size of domain
function initial_current(canvas::Canvas, R::Real, Z::Real)
    Rb, Zb, Ip = canvas._Rb_target, canvas._Zb_target, canvas.Ip
    Rmin, Rmax = extrema(Rb)
    Zmin, Zmax = extrema(Zb)
    R0 = 0.5 * (Rmax + Rmin)
    Z0 = 0.5 * (Zmax + Zmin)
    a = 0.4 * (Rmax - Rmin)
    b = 0.4 * (Zmax - Zmin)
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

mutable struct PaxisIp{T} <: CurrentProfile
    paxis::T
    alpha_m::T
    alpha_n::T
    L::T
    Beta0::T
end

mutable struct PprimeFFprime{F1<:Function, F2<:Function} <: CurrentProfile
    pprime::F1
    ffprime::F2
    ffp_scale::Float64
end

PprimeFFprime(pprime::Function, ffprime::Function) = PprimeFFprime(pprime, ffprime, 1.0)

@inline shape_function(psi_norm::Real, profile::Union{BetapIp, PaxisIp}) = (1.0 - psi_norm ^ profile.alpha_m) ^ profile.alpha_n

function shape_integral(canvas::Canvas, profile::Union{BetapIp, PaxisIp}, psi_norm)
    am, an = profile.alpha_m, profile.alpha_n
    invam = 1.0 / am
    man = -an
    invam_p1 = 1.0 + 1.0 / am
    I = F21(invam, man, invam_p1, 1.0 ^ am) - psi_norm * F21(invam, man, invam_p1, psi_norm ^ am)
    return I * (canvas.Ψbnd - canvas.Ψaxis)
end

function BetapIp(betap,alpha_m,alpha_n)
    if alpha_m < 0 || alpha_n < 0
        @error "alpha_m/n must be positive"
    end
    return BetapIp(betap,alpha_m, alpha_n,zero(betap),zero(betap))
end


function Jtor!(canvas::Canvas, profile::BetapIp)
    Rs, Zs, Ψ, Ip, Raxis = canvas.Rs, canvas.Zs, canvas.Ψ, canvas.Ip, canvas.Raxis
    ellipse = compute_ellipse(canvas)
    p_int = zero(eltype(Ψ))
    IR    = zero(eltype(Ψ))
    I_R   = zero(eltype(Ψ))
    for (i,r) in enumerate(Rs)
        r_Raxis = r / Raxis
        Raxis_r = Raxis / r
        for (j, z) in enumerate(Zs)
            if is_inside[i, j]
                psin = psinorm(Ψ[i, j], canvas)
                jtor_shape = shape_function(psin, profile)
                p_int += r * shape_integral(canvas, profile, psin)
                IR  += jtor_shape * r_Raxis
                I_R += jtor_shape * Raxis_r
            end
        end
    end
    dRdZ = step(Rs) * step(Zs)

    p_int *= dRdZ
    IR    *= dRdZ
    I_R   *= dRdZ

    LBeta0 = (-profile.betap * (μ₀ / (8π)) * Raxis * Ip ^ 2) / p_int
    L = Ip / I_R - LBeta0 * (IR / I_R - 1)
    Beta0 = LBeta0/L

    Jt = canvas._Jt
    Jt .= 0.0
    for (i,r) in enumerate(Rs)
        r_Raxis = r / Raxis
        Raxis_r = Raxis / r
        for (j, z) in enumerate(Zs)
            if is_inside[i, j]
                psin = psinorm(Ψ[i, j], canvas)
                jtor_shape = shape_function(psin, profile)
                Jt[i,j] = L * (Beta0 * r_Raxis + (1 - Beta0) * Raxis_r) * jtor_shape
            end
        end
    end
    profile.L = L
    profile.Beta0 = Beta0
    return Jt
end

function PaxisIp(paxis, alpha_m, alpha_n)
    if alpha_m < 0 || alpha_n < 0
        @error "alpha_m/n must be positive"
    end
    return PaxisIp(paxis, alpha_m, alpha_n, zero(paxis), zero(paxis))
end

function Jtor!(canvas::Canvas, profile::PaxisIp)
    Rs, Zs, Ψ, Ip, Raxis, is_inside = canvas.Rs, canvas.Zs, canvas.Ψ, canvas.Ip, canvas.Raxis, canvas._is_inside

    IR    = zero(eltype(Ψ))
    I_R   = zero(eltype(Ψ))
    for (i, r) in enumerate(Rs)
        r_Raxis = r / Raxis
        Raxis_r = Raxis / r
        for (j, z) in enumerate(Zs)
            if is_inside[i, j]
                psin = psinorm(Ψ[i, j], canvas)
                jtor_shape = shape_function(psin, profile)
                IR  += jtor_shape * r_Raxis
                I_R += jtor_shape * Raxis_r
            end
        end
    end
    dRdZ = step(Rs) * step(Zs)

    IR  *= dRdZ
    I_R *= dRdZ

    LBeta0 = -profile.paxis * Raxis / shape_integral(canvas, profile, 0.0)

    L = Ip / I_R - LBeta0 * (IR / I_R - 1)
    Beta0 = LBeta0 / L

    Jt = canvas._Jt
    Jt .= 0.0
    for (i,r) in enumerate(Rs)
        r_Raxis = r / Raxis
        Raxis_r = Raxis / r
        for (j, z) in enumerate(Zs)
            if is_inside[i, j]
                psin = psinorm(Ψ[i, j], canvas)
                jtor_shape = shape_function(psin, profile)
                Jt[i,j] = L * (Beta0 * r_Raxis + (1 - Beta0) * Raxis_r) * jtor_shape
            end
        end
    end
    profile.L = L
    profile.Beta0 = Beta0
    return Jt
end

function pprime(canvas::Canvas,p::Union{BetapIp,PaxisIp}, psi_norm)
    return (p.L*p.Beta0/canvas.Raxis) * shape_function(psi_norm, profile)
end

function ffprime(canvas::Canvas,p::Union{BetapIp,PaxisIp}, psi_norm)
    return μ₀*p.L*(1 - p.Beta0) * shape_function(psi_norm, profile)
end

function pprime(canvas::Canvas, p::PprimeFFprime, psi_norm)
    p.pprime(psi_norm)
end

function ffprime(canvas::Canvas, p::PprimeFFprime, psi_norm)
    p.ffprime(psi_norm) * p.ffp_scale
end

function Jtor!(canvas::Canvas, profile::PprimeFFprime)
    Rs, Zs, Ψ, Ip, Jt, is_inside = canvas.Rs, canvas.Zs, canvas.Ψ, canvas.Ip, canvas._Jt, canvas._is_inside
    Jt .= 0.0

    # compute the FF' contribution to Ip
    for (i,R) in enumerate(Rs)
        for (j, z) in enumerate(Zs)
            if is_inside[i, j]
                psin = psinorm(Ψ[i, j], canvas)
                Jt[i, j] = -twopi * ffprime(canvas, profile, psin) / (R * μ₀)
            end
        end
    end
    If_c = sum(Jt) * step(Rs) * step(Zs)

    # compute total Ip without scaling
    for (i,R) in enumerate(Rs)
        for (j, z) in enumerate(Zs)
            if is_inside[i, j]
                psin = psinorm(Ψ[i, j], canvas)
                Jt[i, j] += -twopi * R * pprime(canvas, profile, psin)
            end
        end
    end
    Ic = sum(Jt) * step(Rs) * step(Zs)

    # scale FF' to fix Ip
    fac =  1 + (Ip - Ic) / If_c
    for (i,R) in enumerate(Rs)
        for (j, z) in enumerate(Zs)
            if is_inside[i, j]
                psin = psinorm(Ψ[i, j], canvas)
                Jt[i, j] = -twopi * (R * pprime(canvas, profile, psin) + fac * ffprime(canvas, profile, psin) / (R * μ₀))
            end
        end
    end

    profile.ffp_scale *= fac

    return Jt
end
