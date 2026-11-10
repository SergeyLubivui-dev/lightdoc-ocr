"""Генератор синтетических тест-фото ВУ (вымышленные данные!) на базе публичного образца Wikimedia Commons.
Делает карточку 1026x648 с подставленным текстом, затем «фотографирует» её на тёмном фоне под углом."""
import cv2, numpy as np, sys, random
from PIL import Image, ImageDraw, ImageFont
random.seed(1); np.random.seed(1)
SRC = "samples/spec3.png"
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
card = cv2.resize(cv2.imread(SRC), (1026, 648), interpolation=cv2.INTER_AREA)
S = 1026/900
def R(x0,y0,x1,y1): return tuple(int(v*S) for v in (x0,y0,x1,y1))
mask = np.zeros(card.shape[:2], np.uint8)
for z in [(365,108,640,300),(365,333,520,378),(365,380,530,413)]:
    x0,y0,x1,y1 = R(*z); mask[y0:y1,x0:x1] = 255
card = cv2.inpaint(card, mask, 7, cv2.INPAINT_TELEA)
img = Image.fromarray(cv2.cvtColor(card, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(img)
def T(x,y,s,txt,bold=True,col=(35,35,45)):
    d.text((int(x*S),int(y*S)), txt, font=ImageFont.truetype(FB if bold else FR, int(s*S)), fill=col)
T(372,113,27,"ПЕТРОВ")
T(372,142,13,"PETROV",False,(90,90,100))
T(372,172,26,"АЛЕКСАНДР СЕРГЕЕВИЧ")
T(372,203,13,"ALEKSANDR SERGEEVICH",False,(90,90,100))
T(372,225,24,"14.03.1987")
T(372,253,22,"ГОР. КАЗАНЬ")
T(372,277,13,"KAZAN",False,(90,90,100))
T(372,335,24,"ГИБДД 1608")
T(372,360,13,"GIBDD 1608",False,(90,90,100))
T(372,383,24,"16 12 345678")
FACE = "samples/face.png"
import os
if os.path.exists(FACE):
    f = cv2.imread(FACE); x0,y0,x1,y1 = R(64,133,277,442)
    f = cv2.resize(f,(x1-x0,y1-y0)); a = np.array(img); a[y0:y1,x0:x1] = cv2.cvtColor(f,cv2.COLOR_BGR2RGB); img = Image.fromarray(a)
card = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
cv2.imwrite("samples/synth_card.png", card)

def scene(name, W=1600, H=1200, angle=-14, tilt=0.06, scale=0.55, rot180=False, glare=False, seed=0, bgc=(28,30,32), clutter=False):
    rng = np.random.default_rng(seed)
    bg = np.zeros((H,W,3), np.float32)+ bgc
    n = rng.normal(0,1,(H//8,W//8,3)).astype(np.float32); n = cv2.resize(n,(W,H),interpolation=cv2.INTER_CUBIC)
    bg += n*7 + np.linspace(0,25,H)[:,None,None]*np.linspace(1,0.3,W)[None,:,None]
    bg += rng.normal(0,3,(H,W,3))
    c = card if not rot180 else cv2.rotate(card, cv2.ROTATE_180)
    h,w = c.shape[:2]; cw = W*scale; ch = cw*h/w
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = np.float32([[0,0],[cw,0],[cw*(1-tilt/2),ch*(1+tilt*0)],[cw*tilt/2,ch]])
    dst[1,1]+=ch*tilt*0.3; dst[2,1]+=ch*tilt*0.1
    ang = np.deg2rad(angle); Rm = np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])
    ctr = dst.mean(0); dst = (dst-ctr)@Rm.T + np.array([W*0.5,H*0.52])
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    import json; json.dump({'corners_TL_TR_BR_BL': dst.tolist()}, open(name.replace('.jpg','.gt.json'),'w'))
    warped = cv2.warpPerspective(c, M, (W,H), flags=cv2.INTER_CUBIC)
    m = cv2.warpPerspective(np.full((h,w),255,np.uint8), M, (W,H))
    # rounded corners
    rr = np.zeros((h,w),np.uint8); cv2.rectangle(rr,(0,0),(w-1,h-1),255,-1)
    k = 36; rr2 = np.zeros_like(rr); cv2.rectangle(rr2,(k,0),(w-k,h),255,-1); cv2.rectangle(rr2,(0,k),(w,h-k),255,-1)
    for cx,cy in [(k,k),(w-k,k),(k,h-k),(w-k,h-k)]: cv2.circle(rr2,(cx,cy),k,255,-1)
    m = cv2.warpPerspective(rr2, M, (W,H))
    sh = cv2.GaussianBlur(cv2.warpPerspective(rr2, M, (W,H)),(0,0),14); sh = np.roll(sh,(14,10),(0,1))
    if clutter:
        for _ in range(14):
            cv2.line(bg,(int(rng.integers(0,W)),int(rng.integers(0,H))),(int(rng.integers(0,W)),int(rng.integers(0,H))),tuple(float(v) for v in rng.integers(20,230,3)),int(rng.integers(2,10)))
    out = bg*(1-0.5*sh[...,None]/255)
    a = (cv2.GaussianBlur(m,(0,0),0.8)/255.0)[...,None]
    out = out*(1-a)+warped*a
    if glare:
        yy,xx = np.mgrid[0:H,0:W]; g = np.exp(-(((xx-W*0.6)/220)**2+((yy-H*0.45)/90)**2))
        out += g[...,None]*110
    out = cv2.GaussianBlur(np.clip(out,0,255),(0,0),0.9)+rng.normal(0,2.5,(H,W,3))
    cv2.imwrite(name, np.clip(out,0,255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY,72])
scene("samples/synth1.jpg", angle=-14)
scene("samples/synth2.jpg", angle=9, tilt=0.12, scale=0.5, seed=1, glare=True)
scene("samples/synth3.jpg", angle=-20, tilt=0.05, scale=0.62, rot180=True, seed=2, W=1280, H=960)

scene("samples/synth4.jpg", angle=5, tilt=0.08, scale=0.6, seed=4, bgc=(150,175,200))      # светлый бежевый фон
scene("samples/synth5.jpg", W=800, H=600, angle=-6, tilt=0.04, scale=0.38, seed=5)          # маленькая карточка, низкое разрешение
scene("samples/synth6.jpg", angle=28, tilt=0.22, scale=0.5, seed=6, clutter=True, bgc=(60,70,60))  # сильная перспектива и «мусор»
scene("samples/synth7.jpg", W=1200, H=1600, angle=78, tilt=0.07, scale=0.75, seed=7)        # вертикально
