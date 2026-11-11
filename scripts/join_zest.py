import sys, json, math, numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.graph import route_through_array
S=sys.argv[1]; MX,MY=-2304,-3328
roads=json.load(open(f"{S}/roads_zest.json"))
m=np.asarray(Image.open(f"{S}/roadmask_north-america.png").convert("L"))>0
s=gaussian_filter(m.astype(np.float32),3); s=np.clip(s*4,0,1)
H,W=s.shape; cost=1+30*(1-s)
def L(l): return sum(math.dist(a,b) for a,b in zip(l,l[1:]))
def nearest(p,dirv,skip):
    best=(1e9,None)
    for j,l in enumerate(roads):
        if j==skip: continue
        for a,b in zip(l,l[1:]):
            dx,dy=b[0]-a[0],b[1]-a[1]; n=dx*dx+dy*dy
            t=0 if n==0 else max(0,min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/n))
            q=(a[0]+t*dx,a[1]+t*dy); d=math.dist(p,q)
            if d<4: return (d,q)
            if d>170: continue
            v=((q[0]-p[0])/d,(q[1]-p[1])/d)
            if v[0]*dirv[0]+v[1]*dirv[1] < math.cos(math.radians(55)): continue
            if d<best[0]: best=(d,q)
    return best
new=[];n=0
for i,l in enumerate(roads):
    if L(l)<40: continue
    for end in (0,-1):
        p=l[end]; q2=l[1] if end==0 else l[-2]
        # direction of outward travel, from a point ~25 m back
        k=0
        pts=l if end==-1 else l[::-1]
        back=pts[max(0,len(pts)-1-4)]
        dv=(p[0]-back[0],p[1]-back[1]); dn=math.hypot(*dv)
        if dn==0: continue
        dv=(dv[0]/dn,dv[1]/dn)
        d,q=nearest(p,dv,i)
        if q is None or d<4: continue
        px=lambda P:(int(round(P[0]-MX)),int(round(P[1]-MY)))
        x0,y0=px(p);x1,y1=px(q)
        if not(0<=x0<W and 0<=x1<W and 0<=y0<H and 0<=y1<H): continue
        xa,xb=max(min(x0,x1)-25,0),min(max(x0,x1)+26,W);ya,yb=max(min(y0,y1)-25,0),min(max(y0,y1)+26,H)
        path,_=route_through_array(cost[ya:yb,xa:xb],(y0-ya,x0-xa),(y1-ya,x1-xa),fully_connected=True,geometric=True)
        pp=[(c+xa,r+ya) for r,c in path]
        ons=float(np.mean([s[r+ya,c+xa] for r,c in path]))
        if (len(pp)<=1.18*d+6 and ons>=0.3) or (d<=45 and len(pp)<=1.5*d+6):
            sm=pp[::8]+[pp[-1]]
            new.append([p]+[[x+MX,y+MY] for x,y in sm[1:-1]]+[list(q)]); n+=1
print("joins",n)
json.dump(roads+new,open(f"{S}/roads_zest_j.json","w"))
