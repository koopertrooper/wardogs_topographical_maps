import sys, os, json, heapq, numpy as np
from scipy.ndimage import gaussian_filter
SCR=sys.argv[1]; D=os.path.join(SCR,'repo/wardogs-data/data/maps'); N=2048
def flow(z):
    n=z.shape[0]; z=z.astype(np.float64).copy(); done=np.zeros(z.shape,bool); h=[]
    for r in range(n):
        for c in (0,n-1): heapq.heappush(h,(z[r,c],r,c)); done[r,c]=True
    for c in range(1,n-1):
        for r in (0,n-1): heapq.heappush(h,(z[r,c],r,c)); done[r,c]=True
    nb=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    while h:
        zz,r,c=heapq.heappop(h)
        for dr,dc in nb:
            rr,cc=r+dr,c+dc
            if 0<=rr<n and 0<=cc<n and not done[rr,cc]:
                done[rr,cc]=True
                if z[rr,cc]<=zz: z[rr,cc]=zz+1e-5
                heapq.heappush(h,(z[rr,cc],rr,cc))
    pad=np.pad(z,1,constant_values=-1e9); best=np.zeros(z.shape); rec=np.arange(n*n).reshape(n,n); R,C=np.mgrid[0:n,0:n]
    for dr,dc in nb:
        drop=(z-pad[1+dr:1+dr+n,1+dc:1+dc+n])/np.hypot(dr,dc); b=drop>best; best=np.where(b,drop,best)
        rec=np.where(b,np.clip(R+dr,0,n-1)*n+np.clip(C+dc,0,n-1),rec)
    rec=rec.ravel(); acc=np.ones(n*n)
    for i in np.argsort(-z.ravel(),kind='stable'):
        j=rec[i]
        if j!=i: acc[j]+=acc[i]
    return rec,acc
out={}
for w in ['kavkazi','europe']:
    t=json.load(open(f'{D}/{w}/terrain.json')); (x0,y0),(x1,y1)=t['bounds']['min'],t['bounds']['max']
    a=np.load(f'{SCR}/{w}.npy').astype(np.float32).reshape(N,4,N,4).mean(axis=(1,3))
    rec,acc=flow(gaussian_filter(a,1.5))
    np.save(f'{SCR}/{w}-acc.npy',acc); np.save(f'{SCR}/{w}-rec.npy',rec)
    cell=(x1-x0)/N
    print(w,'max catchment km2',round(acc.max()*cell*cell/1e6,1),'cells>=',[(T,int((acc>=T).sum())) for T in (20000,50000,100000,200000)])
