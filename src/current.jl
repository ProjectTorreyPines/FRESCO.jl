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