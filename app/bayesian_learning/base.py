def get_bayesian_estimate(nc, n, m, p):
    return (nc + m * p) / (n + m)

def get_probability(nc, n):
    return nc / n
