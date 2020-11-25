import cv2
import imutils
import numpy as np
import pytesseract 
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


img = cv2.imread('D://passat.jpeg') #kep beolvasás
cv2.imshow('A sima kep',img)
    

img = cv2.resize(img, (600,410))#kep atmeretezes
#img = imutils.resize(img, width=600)
cv2.imshow('resizeolt kep',img)

szurke = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #a szinek atkonvertalasa RGBből szürkébe
cv2.imshow('1',szurke)
szurke = cv2.bilateralFilter(szurke, 11, 16, 16)#filter a zajok eltüntetéséhez
cv2.imshow('2',szurke)
#szélek detektálása
edged = cv2.Canny(szurke, 100, 210)# Csak azok az élek amelyek intenzitási értéke a minimum és maximum értékek közé esik lesz kirajzolva
cv2.imshow('sarkositott',edged)

contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#kép másolata edged.copy() mert a findContours() megváltoztatja a képet ,function: first one is source image, second is contour retrieval mode, third is contour approximation method. And it outputs a modified image.
contours = imutils.grab_contours(contours)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:12]#körvonal rendezés nagytól kicsikig és csak az első 12őt nézzük
found = None # 0 találattal kezdünk
#Ahhoz hogy kiszűrjük a rendszámot végig kell iterálni a találatok között
#és leellenőrizni melyiknek van téglalap alakja (négy körvonalból áll és zárt)
for c in contours:
    kerulet = cv2.arcLength(c, True)#kontúr kerület kiszámítás , masodik argument true tehat zart a körvonal
    kozelit = cv2.approxPolyDP(c, 0.02*kerulet, True)#0.02 a pontosság mértéke minel nagyobb ez a szam annal biztosabb hogy eltérő objektumunk van mint amire szamitunk
  #ha 4 sarka van a korvonalunknak megtaláltuk valószínűleg a rendszámot
    if len(kozelit) == 4: #azok kiválasztása aminek 4 sarka van
        found= kozelit
        break

if found is None: # ha nincs talalat
    detected = 0
    print ("A korvonal nem felismereto.")
else:
    cv2.drawContours(img,[found], -1, (0, 255,0), 3)#renszam korberajzolása:-1 az összes kontúr megrajzolását jelenti, zárójelben a szín , utolsó a vastagsága
#cv2.imshow('kontur',img)
#maszkolása mindennek ami nem a rendszám
mask = np.zeros(szurke.shape,np.uint8)#nullakkal feltoltes
cv2.imshow('maszkolt',mask)
new_image = cv2.drawContours(mask,[found],0,255,-1,)
cv2.imshow('maszkolt1',new_image)
new_image = cv2.bitwise_and(img,img,mask=mask)
cv2.imshow('maszkolt2',new_image)
(x, y) = np.where(mask ==255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = szurke[topx:bottomx+2, topy:bottomy+2]#toptol bottomig így rajzoltat ki 
cv2.imshow('cropped',Cropped)
#karakter leolvasas
text = pytesseract.image_to_string(Cropped, config='--psm 11')
print("A rendszam:",text)
Cropped = cv2.resize(Cropped,(300,150))
cv2.imshow('auto',img)
cv2.imshow('vagott kep',Cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()