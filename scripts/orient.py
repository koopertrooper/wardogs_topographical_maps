import json, math, sys, numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, sobel
S=sys.argv[1]; MX,MY=-2304,-3328
im=np.asarray(Image.open(f"{S}/mosaic_north-america.png").convert("RGB")).astype(np.float32)
g=gaussian_filter(im.mean(2),0.8); gx=sobel(g,1); gy=sobel(g,0); mag=np.hypot(gx,gy); phi=np.arctan2(gy,gx)
B=json.load(open(f"{S}/repo/wardogs-data/data/maps/north-america/buildings.json"))["buildings"]
out={}
for b in B:
    x0,y0=b["min"][:2]; x1,y1=b["max"][:2]
    a,c=int(x0-MX),int(y0-MY); e,f=int(math.ceil(x1-MX)),int(math.ceil(y1-MY))
    if a<2 or c<2 or e>im.shape[1]-2 or f>im.shape[0]-2: continue
    W,H=x1-x0,y1-y0
    if min(W,H)<2.2: continue
    m=mag[c:f,a:e]**2; p=phi[c:f,a:e]
    # weight to centre region
    yy,xx=np.mgrid[0:f-c,0:e-a]; wgt=np.exp(-(((xx-(e-a)/2)/(W/1.6))**2+((yy-(f-c)/2)/(H/1.6))**2))
    s=(m*wgt*np.sin(4*p)).sum(); co=(m*wgt*np.cos(4*p)).sum()
    th=0.25*math.atan2(s,co)  # in (-45,45]
    conf=math.hypot(s,co)/((m*wgt).sum()+1e-6)
    out[b["id"]]=(math.degrees(th),conf)
json.dump(out,open(f"{S}/orient.json","w"))
print(len(out), np.percentile([abs(v[0]) for v in out.values()],[10,50,90]), np.percentile([v[1] for v in out.values()],[10,50,90]))
# neighbour smoothing (mod 90 deg): small clustered buildings share the row/block orientation
ids=[b for b in B if b["id"] in out]
C={b["id"]:((b["min"][0]+b["max"][0])/2,(b["min"][1]+b["max"][1])/2) for b in ids}
new={}
for b in ids:
    i=b["id"]; th0,c0=out[i]; sx=3*c0*math.sin(4*math.radians(th0)); cx_=3*c0*math.cos(4*math.radians(th0))
    for b2 in ids:
        j=b2["id"]
        if j==i: continue
        d=math.dist(C[i],C[j])
        if d<40:
            w=out[j][1]*(1-d/40); sx+=w*math.sin(4*math.radians(out[j][0])); cx_+=w*math.cos(4*math.radians(out[j][0]))
    new[i]=(math.degrees(0.25*math.atan2(sx,cx_)),max(out[i][1],0.3))
json.dump(new,open(f"{S}/orient.json","w"))
