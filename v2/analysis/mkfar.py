import json,sys
cmp=sys.argv[1]; out=sys.argv[2]
pts=[]
for dt in ["fp32","fp16"]:
    for sh in ["out-256x64x256","in-256x256x64"]:
        for lay in ["contig","bT","aT","slice"]:
            for d in (-1,0,1):
                B=131072+d; lab=f"2x2^32 elem {d:+d}" if d else "2x2^32 elem"
                pts.append(dict(shape=sh,dtype=dt,layout=lay,B=B,kind="random",seed=0,compare=cmp,label=lab))
                if d==1:
                    for kd in ["idx_a_batch","idx_a_rc","idx_b_batch","idx_b_rc"]:
                        pts.append(dict(shape=sh,dtype=dt,layout=lay,B=B,kind=kd,seed=0,compare=cmp,label=lab))
json.dump(pts,open(out,"w"),indent=0); print(len(pts))
