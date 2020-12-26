import cv2
import numpy as np
import pytesseract 
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
count=0
for i in range(100):

 img=cv2.imread('D:\\tesztgepilatas\\'+str(i+1)+'.jpg')
    
 #img = cv2.resize(img, (600,400))#kep atmeretezes
 #cv2.imshow('resizeolt kep',img)

 szurke = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #a szinek atkonvertalasa RGBből szürkébe
 #cv2.imshow('sima szurke',szurke)
 szurke = cv2.bilateralFilter(szurke, 11, 16, 16)#filter a zajok eltüntetéséhez
 #cv2.imshow('filteres szurke',szurke)
 # szélek detektálása
 edged = cv2.Canny(szurke, 10, 400)# Csak azok az élek amelyek intenzitási értéke a minimum és maximum értékek közé esik lesz kirajzolva
 #cv2.imshow('sarkositott',edged)

 contours,hierarcy= cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#kép másolata edged.copy() mert a findContours() megváltoztatja a képet
 #function: first one is source image, second is contour retrieval mode, third is contour approximation method. hierarcy meagdja melyik melyikben találató (körvonal)
 #cv2.CHAIN_APPROX_SIMPLE memóriát takarít meg hogy nem a hatrvonalakat rajzolja be hanem csak a sarokpontokat egy objektumnál
 print("Körvonalak száma:"+str(len(contours)))
 kont1=cv2.drawContours(img.copy(), contours, -1, (0,255,0), 1)
 #cv2.imshow('kontur vonalak amiket megtalált',kont1)
 contours = sorted(contours, key = cv2.contourArea, reverse = True)[:7]#körvonalrendezés nagytól kicsikig és csak az első 7et nézzük

 found = None # 0 találattal kezdünk
 #Ahhoz hogy kiszűrjük a rendszámot végig kell iterálni a találatok között
 #és leellenőrizni melyiknek van téglalap alakja (négy körvonalból áll és zárt)
 for c in contours:
    kerulet = cv2.arcLength(c, True)#kontúr kerület kiszámítás , masodik argument true tehat zart a körvonal
    kozelit = cv2.approxPolyDP(c, 0.01*kerulet, True)#0.02 a pontosság mértéke (2%) minel nagyobb ez a szam annal biztosabb hogy eltérő objektumunk van mint amire számitunk
  #ha 4 sarka van a korvonalunknak megtaláltuk valószínűleg a rendszámot
    if len(kozelit) == 4: #azok kiválasztása aminek 4 sarka van, len az elemben találat darabszámmal tér vissza
        found= kozelit
        break

 if found is None: # ha nincs talalat
    detected = 0
    print ("A korvonal nem felismereto.")
    print(str(i+1)+'-------------------------------------')
 else:
  count+=1 
  imgcpy=cv2.drawContours(img.copy(),[found],-1, (0, 255,0),2) #renszam korberajzolása:-1 az összes kontúr megrajzolását jelenti, zárójelben a szín , utolsó a vastagsága

  #maszkolása mindennek ami nem a rendszám
  mask = np.zeros(szurke.shape,np.uint8) #nullakkal feltoltes
  new_image = cv2.drawContours(mask,[found],0,255,-1,)#fehérrel a rendszámot maszkolja
  
  
  new_image = cv2.bitwise_and(img,img,mask)#és művelet pixelenként, tehát a fekete rész 0 a fehér 1 ,ezért a fehér részt tölti ki a rendszámmal
  #cv2.imshow('maszkolt2',new_image)
  (x, y) = np.where(mask ==255)#mivel a maszkoltat módosítja a drawcontour ezért benne marad a fehér maszk, így az összes pixel érték ami fehér (255) értékű a rendszám része
  #két x , y numpi tombbel elmenti a rendszám pozícióját
  (minx, miny) = (np.min(x), np.min(y))#min max pozíciók mentése
  (maxx, maxy) = (np.max(x), np.max(y))
  Cropped = szurke[minx+1:maxx+1, miny+1:maxy+1] #mintol maxig ,így rajzoltat ki
  #cv2.imshow('cropped',Cropped)

  #karakter leolvasas
  text = pytesseract.image_to_string(Cropped, config='--psm 11')#11es config ->Sparse text. Find as much text as possible in no particular order.
  print("A rendszam:",text)
  Cropped = cv2.resize(Cropped,(300,150))
  #cv2.imshow('auto',img)
  cv2.imshow('vagott kep',Cropped)

  cv2.waitKey(0)
  cv2.destroyAllWindows()
  print('Az a valamilyen szinten felismertek szama:'+str(count))
  print(str(i+1)+'-------------------------------------')