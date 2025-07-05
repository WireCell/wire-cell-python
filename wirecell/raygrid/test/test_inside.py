from wirecell.raygrid.tiling import fresh_insides

# by gemini
def test_fresh_inside():

    # Example 1: One blob, two strips, one pair
    # Strips: [0, 10], [5, 15]
    blobs_ex1 = torch.tensor([[[0, 10], [5, 15]]])

    # Crossings for the pair of strips (strip 0 and strip 1)
    # The 4 combinations: (lo0,lo1), (lo0,hi1), (hi0,lo1), (hi0,hi1)
    # Values:
    # (0,5) -> P1=0, P2=5
    # (0,15) -> P1=0, P2=15
    # (10,5) -> P1=10, P2=5
    # (10,15) -> P1=10, P2=15
    crossings_ex1 = torch.tensor([[[ # nblobs=0, npairs=0
        [[0,0], [1,0]], # Combination 0: (lo of strip 0, lo of strip 1) -> value P1=blobs[0,0,0]=0, P2=blobs[0,1,0]=5
        [[0,0], [1,1]], # Combination 1: (lo of strip 0, hi of strip 1) -> value P1=blobs[0,0,0]=0, P2=blobs[0,1,1]=15
        [[0,1], [1,0]], # Combination 2: (hi of strip 0, lo of strip 1) -> value P1=blobs[0,0,1]=10, P2=blobs[0,1,0]=5
        [[0,1], [1,1]]  # Combination 3: (hi of strip 0, hi of strip 1) -> value P1=blobs[0,0,1]=10, P2=blobs[0,1,1]=15
    ]]])

    result_ex1 = fresh_insides(blobs_ex1, crossings_ex1)
    print("Example 1 Result:")
    print(result_ex1)
    # Expected Output:
    # tensor([[[False, False,  True, False]]])
    # Explanation for Example 1:
    # - Crossing 0 (P1=0, P2=5):
    #   - P1=0: Not inside [5,15] -> False
    # - Crossing 1 (P1=0, P2=15):
    #   - P1=0: Not inside [5,15] -> False
    # - Crossing 2 (P1=10, P2=5):
    #   - P1=10: Inside [0,10] (True), Inside [5,15] (True) -> P1 is inside all strips (True)
    #   - P2=5: Inside [0,10] (True), Inside [5,15] (True) -> P2 is inside all strips (True)
    #   - Both True -> Result True
    # - Crossing 3 (P1=10, P2=15):
    #   - P1=10: Inside all strips (True)
    #   - P2=15: Not inside [0,10] -> False
    #   - One False -> Result False


    print("\n" + "="*30 + "\n")

    # Example 2: One blob, single strip
    # Strips: [0, 10]
    blobs_ex2 = torch.tensor([[[0, 10]]])

    # Crossings for the single strip (strip 0)
    # All points are from strip 0.
    crossings_ex2 = torch.tensor([[[
        [[0,0], [0,0]], # P1=0, P2=0
        [[0,0], [0,1]], # P1=0, P2=10
        [[0,1], [0,0]], # P1=10, P2=0
        [[0,1], [0,1]]  # P1=10, P2=10
    ]]])

    result_ex2 = fresh_insides(blobs_ex2, crossings_ex2)
    print("Example 2 Result:")
    print(result_ex2)
    # Expected Output:
    # tensor([[[True, True, True, True]]])
    # Explanation for Example 2:
    # All values (0 and 10) are within the single strip [0,10].
    # Therefore, all crossings are considered "inside".

    print("\n" + "="*30 + "\n")

    # Example 3: Multiple blobs, multiple pairs, some outside
    blobs_ex3 = torch.tensor([
        [[0, 10], [5, 15]],  # Blob 0
        [[20, 30], [25, 35]] # Blob 1
    ])

    crossings_ex3 = torch.tensor([
        [[ # Blob 0, Pair 0
            [[0,0], [1,0]], # P1=0, P2=5 -> False (P1 not in S1)
            [[0,1], [1,0]], # P1=10, P2=5 -> True
            [[0,0], [0,0]], # P1=0, P2=0 -> False (P1 not in S1)
            [[1,1], [1,1]]  # P1=15, P2=15 -> False (P1 not in S0)
        ]],
        [[ # Blob 1, Pair 0
            [[0,0], [1,0]], # P1=20, P2=25 -> True
            [[0,1], [1,0]], # P1=30, P2=25 -> True
            [[0,0], [0,0]], # P1=20, P2=20 -> True
            [[1,1], [1,1]]  # P1=35, P2=35 -> False (P1 not in S0)
        ]]
    ])

    result_ex3 = fresh_insides(blobs_ex3, crossings_ex3)
    print("Example 3 Result:")
    print(result_ex3)
    # Expected Output:
    # tensor([[[False,  True, False, False]],
    #         [[ True,  True,  True, False]]])
    # Explanation for Example 3:
    # Blob 0, Crossing 0 (P1=0, P2=5): P1=0 is not in [5,15] -> False
    # Blob 0, Crossing 1 (P1=10, P2=5): Both P1=10 and P2=5 are in [0,10] AND [5,15] -> True
    # Blob 0, Crossing 2 (P1=0, P2=0): P1=0 is not in [5,15] -> False
    # Blob 0, Crossing 3 (P1=15, P2=15): P1=15 is not in [0,10] -> False

    # Blob 1, Crossing 0 (P1=20, P2=25): Both P1=20 and P2=25 are in [20,30] AND [25,35] -> True
    # Blob 1, Crossing 1 (P1=30, P2=25): Both P1=30 and P2=25 are in [20,30] AND [25,35] -> True
    # Blob 1, Crossing 2 (P1=20, P2=20): Both P1=20 and P2=20 are in [20,30] AND [25,35] -> True
    # Blob 1, Crossing 3 (P1=35, P2=35): P1=35 is not in [20,30] -> False


