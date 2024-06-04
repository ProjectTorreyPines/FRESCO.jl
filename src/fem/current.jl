function toroidal_current(pin::Pinata, R::Real, q_point::Int, global_dofs::Vector{Int})
    psi = flux(pin, q_point, global_dofs)
    psi_norm = (psi - pin.Ψaxis) / (pin.Ψbnd - pin.Ψaxis)
    dp = pin.dp_dψ(psi_norm)
    F_dF = pin.F_dF_dψ(psi_norm)
    pterm  =  μ₀ * dp_dψ
    ffterm = f_df_dψ  / R^2

    return -twopi^2 * (pterm + ffterm)
end
