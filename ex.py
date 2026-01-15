def expr(x, m, cross):
    from itertools import product
    data = []
    for x_i, (m_i, c_i) in product(cross, zip(m, x)):
        experiment = {
            "x": x_i,
            "m": m_i,
            "c": c_i,
        }
        data.append(experiment)
    return data


if __name__ == "__main__":
    from pandas import DataFrame
    out= expr([1, 2, 3], ['@', "%**", 'Z'], [8, 9,])
    DataFrame(out)
