mutable struct QED_system{Qs<:QED.QED_state, F, Qb<:QED.QED_build, T<:Real}
    QI::Qs
    η::F
    build::Qb
    tmax::T
    Nt::Int
end