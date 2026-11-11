import json, math, sys
from PIL import Image, ImageDraw
S=sys.argv[1]; MX,MY=-2304,-3328
O=json.load(open(f"{S}/orient.json"))
B=json.load(open(f"{S}/repo/wardogs-data/data/maps/north-america/buildings.json"))["buildings"]
im=Image.open(f"{S}/mosaic_north-america.png").convert("RGB"); d=ImageDraw.Draw(im)
def rect(b,th):
    W=b["max"][0]-b["min"][0]; H=b["max"][1]-b["min"][1]; c,s=abs(math.cos(th)),abs(math.sin(th))
    det=c*c-s*s
    if abs(det)<0.15: return None
    w=(W*c-H*s)/det; h=(H*c-W*s)/det
    if w<1 or h<1: return None
    return w,h
for b in B:
    if b["id"] not in O: continue
    th=math.radians(O[b["id"]][0]); r=rect(b,th)
    cx=(b["min"][0]+b["max"][0])/2-MX; cy=(b["min"][1]+b["max"][1])/2-MY
    d.rectangle((b["min"][0]-MX,b["min"][1]-MY,b["max"][0]-MX,b["max"][1]-MY),outline=(255,255,0))
    if not r: continue
    w,h=r; pts=[]
    for sx,sy in ((-1,-1),(1,-1),(1,1),(-1,1)):
        x=sx*w/2;y=sy*h/2; pts.append((cx+x*math.cos(th)-y*math.sin(th),cy+x*math.sin(th)+y*math.cos(th)))
    d.polygon(pts,outline=(255,0,255))
im.crop((750,900,1450,1500)).resize((1100,943),Image.LANCZOS).save(f"{S}/ov_check.png")
