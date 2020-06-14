import pandas as pd 
import numpy as np 
 
butunVeriler = pd.read_csv("seed.csv", sep=',')
veriler = butunVeriler.copy() 
 
satırNo = len(veriler.index) 
sutunNo = len(veriler.columns) 
ozNitelikSayisi = sutunNo - 2
asilKume = sutunNo - 1
indis = veriler.columns[0]
sinifSutunu = veriler.columns[asilKume]
veriler = veriler.drop(columns = [indis, sinifSutunu])  
npData = veriler.to_numpy(copy=True)

K = 3
merkezler = np.empty((K,ozNitelikSayisi))

atananKumeler= np.random.randint(0, K, size = (satırNo))

for merekezSatiri in range(0, K): 
 
        for merkezSutunu in range(0, ozNitelikSayisi):
 
            toplam = 0.0
            sayac = 0.0
            ortalama = None
 
            for satir in range(0, satırNo):
 
                if(merekezSatiri == atananKumeler[satir]):
                 
                    toplam += npData[satir,merkezSutunu]
                    sayac += 1
         
                    if (sayac > 0):

                        ortalama = toplam / sayac
            merkezler[merekezSatiri,merkezSutunu] = ortalama
            
merkezKontrol = False
iterasyonLimiti = 300
 
while iterasyonLimiti > 0 and not(merkezKontrol):
    for satir in range(0, satırNo):
     
        simdikiSatir = npData[satir]
        minMesafe = float("inf")
 
        for merkezSatiri in range(0, K):
            simdikiMerkez = merkezler[merkezSatiri]    
            mesafe = np.linalg.norm(simdikiSatir - simdikiMerkez)
 
            if mesafe < minMesafe:
                atananKumeler[satir] = merkezSatiri
                minMesafe = mesafe
                
   
    eskiMerkezler = merkezler.copy()
    for merekezSatiri in range(0, K):
 
        for merekezSutunu in range(0, ozNitelikSayisi):
 

            toplam = 0.0
            sayac = 0.0
            ortalama = None
 
            for satir in range(0, satırNo):

                if(merekezSatiri == atananKumeler[satir]):
                 
                    toplam += npData[satir,merekezSutunu]
                    sayac += 1
         
                    if (sayac > 0):

                        ortalama = toplam / sayac
 
            merkezler[merekezSatiri,merekezSutunu] = ortalama
     
    merkezKontrol = np.array_equal(eskiMerkezler,merkezler)
 
    if merkezKontrol:
        print("Kümeler değişmedi algoritma durduruldu.")
 
    iterasyonLimiti -= 1
 
asilSiniflarSutunu = butunVeriler.columns[len(
    butunVeriler.columns) - 1]
 
butunVeriler = butunVeriler.reindex(
      columns=[*butunVeriler.columns.tolist(), 'Küme', 'Sınıf Tahmini', ('Tahmin Kontrol')])
butunVeriler['Küme'] = atananKumeler
 

sinifYerlerstirme = pd.DataFrame(index=range(K),columns=range(1))
for kume in range(0, K):
    temp = butunVeriler.loc[butunVeriler['Küme'] == kume]
    sinifYerlerstirme.iloc[kume,0] = temp.mode()[asilSiniflarSutunu][0]
 
kumeSutunu = asilKume + 1
tahminSutunu = asilKume + 2
tahminKontrolSutunu = asilKume + 3
 
for satir in range(0, satırNo):
    for kume in range(0, K):
        if kume == butunVeriler.iloc[satir,kumeSutunu]:
            butunVeriler.iloc[satir,tahminSutunu] = sinifYerlerstirme.iloc[kume,0]
 
    if butunVeriler.iloc[satir,tahminSutunu] == butunVeriler.iloc[satir,asilKume]:
        butunVeriler.iloc[satir,tahminKontrolSutunu] = 1
    else: 
        butunVeriler.iloc[satir,tahminKontrolSutunu] = 0

accuracy = (butunVeriler.iloc[:,tahminKontrolSutunu].sum())/satırNo 
accuracy *= 100

from sklearn.metrics import confusion_matrix

tahmin=butunVeriler.iloc[:,tahminSutunu].values
asil=butunVeriler.iloc[:,asilKume].values

cm = confusion_matrix(asil,tahmin)


print(cm)
print("Accuracy : %" + str(accuracy) )



