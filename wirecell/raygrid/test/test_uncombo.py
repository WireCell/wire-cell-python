import torch

from wirecell.raygrid.funcs import get_unchosen_elements

def test_uncombo():

    # Your specific example: n=3, k=2
    n = 3
    k = 2
    c = torch.combinations(torch.arange(n), k)
    nc = get_unchosen_elements(n, k)

    print(f"Original n: {n}, k: {k}")
    print(f"Combinations (c):\n{c}")
    print(f"Shape of c: {c.shape}")
    print(f"Unchosen elements (nc):\n{nc}")
    print(f"Shape of nc: {nc.shape}\n")

    # Another example: n=4, k=2
    n_ex2 = 4
    k_ex2 = 2
    c_ex2 = torch.combinations(torch.arange(n_ex2), k_ex2)
    nc_ex2 = get_unchosen_elements(n_ex2, k_ex2)

    print(f"Original n: {n_ex2}, k: {k_ex2}")
    print(f"Combinations (c):\n{c_ex2}")
    print(f"Shape of c: {c_ex2.shape}")
    print(f"Unchosen elements (nc):\n{nc_ex2}")
    print(f"Shape of nc: {nc_ex2.shape}\n")

    # Edge case: k=0 (no elements chosen, all are unchosen)
    n_ex3 = 3
    k_ex3 = 0
    c_ex3 = torch.combinations(torch.arange(n_ex3), k_ex3)
    nc_ex3 = get_unchosen_elements(n_ex3, k_ex3)

    print(f"Original n: {n_ex3}, k: {k_ex3}")
    print(f"Combinations (c):\n{c_ex3}")
    print(f"Shape of c: {c_ex3.shape}")
    print(f"Unchosen elements (nc):\n{nc_ex3}")
    print(f"Shape of nc: {nc_ex3.shape}\n")

    # Edge case: k=n (all elements chosen, no unchosen)
    n_ex4 = 3
    k_ex4 = 3
    c_ex4 = torch.combinations(torch.arange(n_ex4), k_ex4)
    nc_ex4 = get_unchosen_elements(n_ex4, k_ex4)

    print(f"Original n: {n_ex4}, k: {k_ex4}")
    print(f"Combinations (c):\n{c_ex4}")
    print(f"Shape of c: {c_ex4.shape}")
    print(f"Unchosen elements (nc):\n{nc_ex4}")
    print(f"Shape of nc: {nc_ex4.shape}\n")

    # Edge case: k > n (no combinations possible)
    # n_ex5 = 3
    # k_ex5 = 4
    # c_ex5 = torch.combinations(torch.arange(n_ex5), k_ex5)
    # nc_ex5 = get_unchosen_elements(n_ex5, k_ex5)

    # print(f"Original n: {n_ex5}, k: {k_ex5}")
    # print(f"Combinations (c):\n{c_ex5}")
    # print(f"Shape of c: {c_ex5.shape}")
    # print(f"Unchosen elements (nc):\n{nc_ex5}")
    # print(f"Shape of nc: {nc_ex5.shape}\n")
