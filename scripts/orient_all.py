import json, math, sys, numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel
S=sys.argv[1]; META=json.load(open(f"{S}/mosaic_meta.json"))
OFF={"kavkazi":(32,35),"europe":(38,39)}
ALL=json.load(open(f"{S}/orient_na.json"))
for w in ("kavkazi","europe"):
    m_=META[w]; MX,MY=m_["ox"]-OFF[w][0],m_["oy"]-OFF[w][1]   # world coord of mosaic pixel (0,0)
    im=np.asarray(Image.open(f"{S}/mosaic_{w}.png").convert("RGB")).astype(np.float32)
    g=gaussian_filter(im.mean(2),0.8); gx=sobel(g,1); gy=sobel(g,0); mag=np.hypot(gx,gy); phi=np.arctan2(gy,gx)
    B=json.load(open(f"{S}/repo/wardogs-data/data/maps/{w}/buildings.json"))["buildings"]
    out={}
    for b in B:
        x0,y0=b["min"][:2]; x1,y1=b["max"][:2]
        a,c=int(x0-MX),int(y0-MY); e,f=int(math.ceil(x1-MX)),int(math.ceil(y1-MY))
        if a<2 or c<2 or e>im.shape[1]-2 or f>im.shape[0]-2: continue
        W,H=x1-x0,y1-y0
        if min(W,H)<2.2: continue
        m=mag[c:f,a:e]**2; p=phi[c:f,a:e]
        yy,xx=np.mgrid[0:f-c,0:e-a]; wgt=np.exp(-(((xx-(e-a)/2)/(W/1.6))**2+((yy-(f-c)/2)/(H/1.6))**2))
        s=(m*wgt*np.sin(4*p)).sum(); co=(m*wgt*np.cos(4*p)).sum()
        out[b["id"]]=(math.degrees(0.25*math.atan2(s,co)),math.hypot(s,co)/((m*wgt).sum()+1e-6))
    ids=[b for b in B if b["id"] in out]
    C={b["id"]:((b["min"][0]+b["max"][0])/2,(b["min"][1]+b["max"][1])/2) for b in ids}
    for b in ids:
        i=b["id"]; th0,c0=out[i]; sx=3*c0*math.sin(4*math.radians(th0)); cx_=3*c0*math.cos(4*math.radians(th0))
        for b2 in ids:
            j=b2["id"]
            if j==i: continue
            d=math.dist(C[i],C[j])
            if d<40:
                w_=out[j][1]*(1-d/40); sx+=w_*math.sin(4*math.radians(out[j][0])); cx_+=w_*math.cos(4*math.radians(out[j][0]))
        ALL[i]=(math.degrees(0.25*math.atan2(sx,cx_)),max(out[i][1],0.3))
    print(w,len(B),len(out), np.percentile([abs(out[k][0]) for k in out],[10,50,90]))
json.dump(ALL,open(f"{S}/orient.json","w"))
# preview
from PIL import ImageDraw
w="europe"; m_=META[w]; MX,MY=m_["ox"]-OFF[w][0],m_["oy"]-OFF[w][1]
im=Image.open(f"{S}/mosaic_{w}.png").convert("RGB"); d=ImageDraw.Draw(im)
for b in json.load(open(f"{S}/repo/wardogs-data/data/maps/{w}/buildings.json"))["buildings"]:
    o=ALL.get(b["id"])
    x0,y0=b["min"][:2];x1,y1=b["max"][:2]
    d.rectangle((x0-MX,y0-MY,x1-MX,y1-MY),outline=(255,255,0))
    if not o: continue
    th=math.radians(o[0]);c,s=abs(math.cos(th)),abs(math.sin(th));det=c*c-s*s
    if abs(det)<.15: continue
    W_,H_=x1-x0,y1-y0;ww=(W_*c-H_*s)/det;hh=(H_*c-W_*s)/det
    if ww<1 or hh<1: continue
    cx,cy=(x0+x1)/2-MX,(y0+y1)/2-MY
    d.polygon([(cx+sx*ww/2*math.cos(th)-sy*hh/2*math.sin(th),cy+sx*ww/2*math.sin(th)+sy*hh/2*math.cos(th)) for sx,sy in((-1,-1),(1,-1),(1,1),(-1,1))],outline=(255,0,255))
im.crop((600,700,1300,1300)).save("/tmp/claude-0/ovE.png")
