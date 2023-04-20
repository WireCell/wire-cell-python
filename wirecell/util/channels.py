def parse_range(channels):
    '''
    Parse a "channels string" return list of (beg,end) channel range pairs.

    String like:

        nu,nv,nw 

    or

        ubeg:uend,nv,wbeg:wend 

    '''
    if not channels:
        return ()

    chrange = list()
    for ch in channels.split(","):
        if ":" in ch:
            chrange.append(tuple(map(int, ch.split(":"))))
            continue
        num = int(ch)
        if chrange:
            l = chrange[-1][1]
            chrange.append((l, l+num))
        else:
            chrange.append((0, num))
    return chrange

