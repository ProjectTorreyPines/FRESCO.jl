# Helper for constructing Duals with correct tag
dual_with_partials_like(like_el, value::Real, partials::AbstractVector{<:Real}) =
    Dual{ForwardDiff.tagtype(like_el)}(value, Tuple(partials))

function search_axis_guess(canvas::Canvas)
    Rs, Zs, Ψ, Ip, is_in_wall = canvas.Rs, canvas.Zs, canvas.Ψ, canvas.Ip, canvas._is_in_wall
    psisign = sign(Ip)

    ia, ja = 0, 0
    found = false
    for j in eachindex(Zs)[2:end-1]
        for i in eachindex(Rs)[2:end-1]
            !is_in_wall[i, j] && continue
            P = psisign * Ψ[i, j]
            if P == minimum(p -> psisign * p, @view(Ψ[i-1:i+1, j-1:j+1]))
                if !found
                    ia, ja = i, j
                    found = true
                else
                    error("Found multiple minimum Ψ points: $((ia, ja)) and $((i, j))")
                end
            end
        end
    end
    return Rs[ia], Zs[ja]
end


function find_axis(canvas::Canvas; update_Ψitp::Bool=true)
    update_Ψitp && update_interpolation!(canvas)

    # find initial guess
    Rg, Zg = if canvas.Raxis > 0.0
        # use the last known axis location, if valid
        canvas.Raxis, canvas.Zaxis
    else
         try
            search_axis_guess(canvas)
        catch e
            println("search_axis_guess() reported error: ", e)
            display(plot(canvas))
            error("Could not find magnetic axis guess from grid and no previous axis location available")
        end
    end

    return _find_axis(canvas, Rg, Zg)
end

grad!(G, x, p) = Interpolations.gradient!(G, p.Ψitp, x[1], x[2])
hess!(H, x, p) = Interpolations.hessian!(H, p.Ψitp, x[1], x[2])
const nf = NonlinearSolve.NonlinearFunction(grad!; jac = hess!)

function _find_axis(Ψitp::Interpolations.AbstractInterpolation, Rg::Real, Zg::Real)
    x0 = @SVector[Rg, Zg]
    prob = NonlinearSolve.NonlinearProblem(nf, x0, (; Ψitp=Ψitp))
    sol  = NonlinearSolve.solve(prob, NonlinearSolve.SimpleNewtonRaphson())#, show_trace=Val(true))
    Raxis, Zaxis = sol.u[1], sol.u[2]

    return Raxis, Zaxis, Ψitp(Raxis, Zaxis)
end

function _find_axis(canvas::Canvas{T, DT, VC, II, DI, C1, C2}, Rg::Real, Zg::Real) where {T, DT<:Real, VC, II, DI, C1, C2}
    # Float path
    return _find_axis(canvas._Ψitp, Rg, Zg)
end

function _find_axis(canvas::Canvas{T, DT, VC, II, DI, C1, C2}, Rg::Real, Zg::Real) where {T, DT<:Dual, VC, II, DI, C1, C2}
     # Dual number path: use implicit differentiation
    Rs, Zs, Ψ, Ψitp = canvas.Rs, canvas.Zs, canvas.Ψ, canvas._Ψitp

    # 1) Strip duals and solve with Float interpolant
    ψ_val = ForwardDiff.value.(Ψ)
    Rg_val, Zg_val = ForwardDiff.value(Rg), ForwardDiff.value(Zg)
    itp_val = ψ_interpolant(Rs, Zs, ψ_val)
    RAf, ZAf, _ = _find_axis(itp_val, Rg_val, Zg_val)

    # 2) Compute Hessian wrt (r,z) at float axis
    H = ForwardDiff.hessian(xx -> itp_val(xx[1], xx[2]), [RAf, ZAf])
    # light diagonal regularization
    H[1,1] += 1e-9
    H[2,2] += 1e-9

    # 3)Compute gradient at axis (w.r.t. Ψ)
    # Use the spline's analytic gradient; returns Duals because coefficients are Dual
    g_dual = Interpolations.gradient(Ψitp, RAf, ZAf)
    # NOTE: g_dual is a length-2 Tuple of Duals; its partials encode ∂∇_x f / ∂Ψ · δΨ

    # 4) Extract sensitivities for each AD direction
    el = Ψ[1, 1]  # any psi element to read AD tag/width
    K = ForwardDiff.npartials(el)
    vcols = Vector{Vector{Float64}}(undef, K)
    for k in 1:K
        v = Vector{Float64}(undef, 2)
        @inbounds begin
            v[1] = ForwardDiff.partials(g_dual[1])[k]
            v[2] = ForwardDiff.partials(g_dual[2])[k]
        end
        vcols[k] = v
    end
    δxcols = map(vk -> -H \ vk, vcols)

    # 5) Rewrap axis coords as Duals with computed partials
    r_partials = map(δx -> δx[1], δxcols)
    z_partials = map(δx -> δx[2], δxcols)
    Raxis = dual_with_partials_like(el, RAf, r_partials)
    Zaxis = dual_with_partials_like(el, ZAf, z_partials)

    value_dual = Ψitp(Raxis, Zaxis)
    return Raxis, Zaxis, value_dual
end

function flux_bounds!(canvas::Canvas; update_Ψitp::Bool=true)
    update_Ψitp && update_interpolation!(canvas)
    Rs, Zs, Ψ, Rw, Zw, Ψitp, r_cache, z_cache = canvas.Rs, canvas.Zs, canvas.Ψ, canvas.Rw, canvas.Zw, canvas._Ψitp, canvas._r_cache, canvas._z_cache
    Raxis, Zaxis, Ψaxis = find_axis(canvas; update_Ψitp=false)
    # BCL 7/31/24 - Potential error here with Ψbnd as that's supposed to be the "original" psi on the boundary
    #               I'm not sure how to handle that, though hopefully it gets fixed on the IMAS side
    axis2bnd = (canvas.Ip >= 0.0) ? :increasing : :decreasing
    Ψbnd = IMAS.find_psi_boundary(Rs, Zs, Ψ, Ψaxis, axis2bnd, Raxis, Zaxis, Rw, Zw, r_cache, z_cache;
                                  PSI_interpolant=Ψitp, raise_error_on_not_open=false, raise_error_on_not_closed=false).last_closed
    try
        canvas.Raxis, canvas.Zaxis, canvas.Ψaxis, canvas.Ψbnd = Raxis, Zaxis, Ψaxis, Ψbnd
    catch e
        p = plot(canvas)
        scatter!([Raxis], [Zaxis], markersize=8, color=:cyan)
        display(p)
        rethrow(e)
    end
end

psinorm(psi::Real, canvas::Canvas) = psinorm(psi, canvas.Ψaxis, canvas.Ψbnd)
psinorm(psi::Real, Ψaxis::Real, Ψbnd::Real) = (psi - Ψaxis) / (Ψbnd - Ψaxis)
psinorm(canvas::Canvas) = range(0.0, 1.0, length(canvas.surfaces))

function boundary!(canvas::Canvas)
    Rs, Zs, Ψ, Raxis, Zaxis, Ψaxis, Ψbnd = canvas.Rs, canvas.Zs, canvas.Ψ, canvas.Raxis, canvas.Zaxis, canvas.Ψaxis, canvas.Ψbnd
    is_inside, r_cache, z_cache = canvas._is_inside, canvas._r_cache, canvas._z_cache
    r, z = IMASutils.contour_from_midplane!(r_cache, z_cache, Ψ, Rs, Zs, Ψbnd, Raxis, Zaxis, Ψaxis)
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

function in_plasma_bb(r::Real, z::Real, rext::Tuple{<:Real, <:Real}, zext::Tuple{<:Real, <:Real})
    rmin, rmax = rext
    (r < rmin || r > rmax) && return false

    zmin, zmax = zext
    (z < zmin || z > zmax) && return false

    return true
end

function in_core(r::Real, z::Real, psin::Real, canvas::Canvas,
    ellipse::Union{Nothing,AbstractVector{<:Real}}=nothing)
    rext, zext, bnd = canvas._rextrema, canvas._zextrema, canvas._bnd
    return in_core(r, z, psin, rext, zext, bnd, ellipse)
end

function in_core(r::Real, z::Real, psin::Real, rext::Tuple{<:Real, <:Real}, zext::Tuple{<:Real, <:Real},
    bnd::Vector{<:SVector{2,<:Real}}, ellipse::Union{Nothing,AbstractVector{<:Real}}=nothing)
    # Check psinorm value
    psin > 1.0 && return false

    # Check outside bounding box
    !in_plasma_bb(r, z, rext, zext) && return false

    in_ellipse(r, z, ellipse) && return true

    # Finally make sure it's in the boundary
    return inpolygon((r, z), bnd) == 1
end

in_ellipse(r, z, ellipse::Nothing) = false

function in_ellipse(r::Real, z::Real, ellipse::AbstractVector{<:Real})
    @assert length(ellipse) == 4
    R0, Z0, a, b = ellipse
    return ((r - R0) / a)^2 + ((z - Z0) / b)^2 <= 1.0
end

compute_ellipse(canvas::Canvas) = compute_ellipse(canvas._bnd, canvas._rextrema, canvas._zextrema)
function compute_ellipse(bnd, rext, zext)
    κ = (zext[2] - zext[1]) / (rext[2] - rext[1])
    R0, Z0 = centroid(bnd)
    radius = p -> sqrt((p[1] - R0)^2 + ((p[2] - Z0) / κ)^2)
    a = minimum(radius, bnd)
    b = κ * a
    return @SVector[R0, Z0, a, b]
end


function trace_surfaces!(canvas::Canvas)
    Rs, Zs, Ψ, Raxis, Zaxis, Ψaxis = canvas.Rs, canvas.Zs, canvas.Ψ, canvas.Raxis, canvas.Zaxis, canvas.Ψaxis
    Ψbnd, Ψitp, Rw, Zw, surfaces = canvas.Ψbnd, canvas._Ψitp, canvas.Rw, canvas.Zw, canvas.surfaces
    r_cache, z_cache = canvas._r_cache, canvas._z_cache
    psi_levels = range(Ψaxis, Ψbnd, length(surfaces))
    return IMAS.trace_simple_surfaces!(surfaces, psi_levels, Rs, Zs, Ψ, Ψitp, Raxis, Zaxis, Rw, Zw, r_cache, z_cache)
end

function update_Vp!(canvas::Canvas)
    Ip, Vp, surfaces = canvas.Ip, canvas._Vp, canvas.surfaces
    sign_dpsi = sign(Ip)
    Threads.@threads for k in eachindex(surfaces)
        Vp[k] = sign_dpsi * surfaces[k].int_fluxexpansion_dl
    end
    x = psinorm(canvas)
    canvas._Vp_itp = DataInterpolations.CubicSpline(Vp, x; extrapolation=ExtrapolationType.None)
    return canvas
end

gm1_integrand(j, surface) = surface.fluxexpansion[j] / surface.r[j] ^ 2
function update_gm1!(canvas::Canvas)
    gm1, surfaces = canvas._gm1, canvas.surfaces
    Threads.@threads for k in eachindex(surfaces)
        f1 = (j, xx) -> gm1_integrand(j, surfaces[k])
        gm1[k] = IMAS.flux_surface_avg(f1, surfaces[k])
    end
    x = psinorm(canvas)
    canvas._gm1_itp = DataInterpolations.CubicSpline(gm1, x; extrapolation=ExtrapolationType.None)
    return canvas
end

gm9_integrand(j, surface) = surface.fluxexpansion[j] / surface.r[j]
function update_gm9!(canvas::Canvas)
    gm9, surfaces = canvas._gm9, canvas.surfaces
    Threads.@threads for k in eachindex(surfaces)
        f9 = (j, xx) -> gm9_integrand(j, surfaces[k])
        gm9[k] = IMAS.flux_surface_avg(f9, surfaces[k])
    end
    x = psinorm(canvas)
    canvas._gm9_itp = DataInterpolations.CubicSpline(gm9, x; extrapolation=ExtrapolationType.None)
    return canvas
end

function update_Fpol!(canvas::Canvas, profile::AbstractCurrentProfile)
    Ψaxis, Ψbnd, Fbnd, Fpol = canvas.Ψaxis, canvas.Ψbnd, canvas.Fbnd, canvas._Fpol
    x = psinorm(canvas)
    psi1d = range(Ψaxis, Ψbnd, length(x))
    # starts as F^2
    IMAS.cumtrapz!(Fpol, psi1d, FFprime.(Ref(canvas), Ref(profile), x))
    Fpol .= 2 .* Fpol .- Fpol[end] .+ Fbnd^2
    Fpol .= sign(Fbnd) .* sqrt.(Fpol) # now take sqrt with proper sign
    canvas._Fpol_itp = DataInterpolations.CubicSpline(Fpol, x; extrapolation=ExtrapolationType.None)
    return canvas
end

function update_rho!(canvas::Canvas)

    rho, Ψaxis, Ψbnd, Vp, gm1, Fpol = canvas._rho, canvas.Ψaxis, canvas.Ψbnd, canvas._Vp, canvas._gm1, canvas._Fpol
    x = psinorm(canvas)
    psi1d = range(Ψaxis, Ψbnd, length(x))

    # compute toroidal flux
    #   Φ = ∫dΨ F * dV/dΨ * <R⁻²> / 2π
    # we ignore the 2π because we'll normalize anyway
    IMAS.cumtrapz!(rho, psi1d, Fpol .* Vp .* gm1)

    # then turn into rho_tor_norm
    rho .= sqrt.(rho ./ rho[end])
    rho[1] = 0.0
    rho[end] = 1.0

    canvas._rho_itp = DataInterpolations.CubicSpline(rho, x; extrapolation=ExtrapolationType.None)
end

# This is <|∇Ψ|^2 / R^2>, so like the gm2 metric in IMAS but for Ψ not rho_tor
function update_gm2p!(canvas::Canvas, profile::AbstractCurrentProfile)
    gm2p, Ψaxis, Ψbnd, Vp = canvas._gm2p, canvas.Ψaxis, canvas.Ψbnd, canvas._Vp
    x = psinorm(canvas)
    fac = twopi * μ₀ * (Ψbnd - Ψaxis)
    IMAS.cumtrapz!(gm2p, x, Vp .* JtoR.(Ref(canvas), Ref(profile), x))
    gm2p .*= fac ./ Vp
    canvas._gm2p_itp = DataInterpolations.CubicSpline(gm2p, x; extrapolation=ExtrapolationType.None)
    return canvas
end

function update_area!(canvas::Canvas)
    area, Ψaxis, Ψbnd, Vp, gm9 = canvas._area, canvas.Ψaxis, canvas.Ψbnd, canvas._Vp, canvas._gm9
    x = psinorm(canvas)
    psi1d = range(Ψaxis, Ψbnd, length(x))
    darea_dpsi = (k, x) -> Vp[k] * gm9[k] / 2π
    IMAS.cumtrapz!(area, psi1d, darea_dpsi)
    canvas._area_itp = DataInterpolations.CubicSpline(area, x; extrapolation=ExtrapolationType.None)
    return canvas
end

function _update_common_fsas!(canvas::Canvas, profile::AbstractCurrentProfile; update_surfaces=false, control::Symbol=:eddy)
    update_surfaces && trace_surfaces!(canvas)
    update_Vp!(canvas)
    update_gm1!(canvas)
    update_Fpol!(canvas, profile)
    update_rho!(canvas)
    (control === :implicit_eddy) && update_gm2p!(canvas, profile)
    return canvas
end

compute_FSAs!(canvas::Canvas, profile::AbstractCurrentProfile; update_surfaces=false, control::Symbol=:eddy) =  _update_common_fsas!(canvas, profile; update_surfaces, control)

function compute_FSAs!(canvas::Canvas, profile::PressureJt; update_surfaces=false, control::Symbol=:eddy)
    _update_common_fsas!(canvas, profile; update_surfaces, control)
    update_gm9!(canvas)
    return canvas
end
