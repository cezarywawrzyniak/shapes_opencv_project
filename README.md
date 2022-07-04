# shapes_opencv_project
Projekt na przedmiot Systemy Wizyjne, którego celem było wykrywanie kształtów oraz kolorów konstrukcji stworzonych z klocków na podstawie zdjęć. 

Rodzaje kształtów:

<img src="https://github.com/cezarywawrzyniak/shapes_opencv_project/blob/main/shape_types.png" width=50% height=50%>

Na wyjście podawana jest lista, której poszczególne wartości oznaczają liczebność danego kształtu (pierwsze 5) i koloru (ostatnie 6) w formacie:

`[typ1, typ2, typ3, typ4, typ5, niebieskim, zielony, czerwony, biały, żólty, kolor_mieszany]`

Dodatkowo każde zdjęcie jest wizualizowane np.:

<img src="https://github.com/cezarywawrzyniak/shapes_opencv_project/blob/main/working_example.png" width=80% height=80%>

## Opis działania: 
Pierwszą funkcją wywoływaną w kodzie jest create_models(), która uzupełnia 5 list wyliczonymi na podstawie przykładów średnimi momentami Hu (pierwszymi 4), które później zostaną wykorzystane do wykrywania kształtów. Reszta działania podzielona jest na 3 etapy:

1. Przetwarzanie wstępne obrazu

Krok ten służy do znalezienia konturów wszystkich obiektów na obrazie. Aby znaleźć ich jak najwięcej wykorzystane zostało sporo technik, które są ostatecznie łączone w jeden obraz wynikowy:
- Operator Sobela do wykrycia krawędzi
- cv2.threhold na każdej składowej BGR
- cv2.inRange do znalezienia żółtych elementów
- algorytm Canny’ego do wykrycia krawędzi

Uzyskane obrazy składowe łączone są przy pomocy funkcji cv2.bitwise_or. Wykorzystywane są także operacje morfologiczne dylacji (do łączenia niecałkowitych konturów) oraz erozji (do eliminacji szumów). W celu uzyskania uzupełnionych wewnątrz konturów wykorzystana została kombinacja cv2.findContours oraz cv2.drawContours. Po wykonaniu erozji jeszcze raz wywoływana jest cv2.findContours do znalezienia ostatecznych konturów do dalszego przetwarzania. Przy pomocy cv2.bitwise_and oraz większej maski do erozji uzyskiwany jest także drugi obraz, który zostanie wykorzystany do wykrywania kolorów (wyeliminowana jest na nim większość tła).

2. Wykrywanie kształtów

Dla każdego konturu wyliczane są momenty Hu oraz ich różnice dla każdego z przygotowanych wzorców. Do pierwszego etapu eliminacji brane są tylko kontury o odpowiedniej wielkości. Aby kontur został przypisany do danej klasy musi on mieć wystarczające małe różnice momentów do wzorca, a każda klasa reprezentowana jest przez słownik do którego kluczem jest powierzchnia a elementem sam kontur. W pierwszym etapie zapisywane są także największe powierzchnie dla każdej klasy na danym zdjęciu oraz zmienna refrence_class wskazuje, którego typu klocka jest najwięcej przykładów. Drugi etap sprawdza wcześniej zapisane słowniki. Jeżeli na zdjęciu znajduje się tylko jeden klocek danej klasy jego wielkość porównywana jest do klasy referencyjnej i jeżeli jest odpowiednio duża to klocek przypisywany jest do klasy. Jeżeli elementów jest więcej to każdy z nich przyrównywany jest do klocka o największej powierzchni i aby został przypisany musi być większy niż ustalony ułamek największej powierzchni. Oba kroki sprawdzania powierzchni służą do eliminacji klocków nienależących do żadnej z klas, których momenty zgadzały się z poszukiwanymi klockami. Pojawiały się one w niewielkich ilościach oraz zwykle miały mniejsze powierzchnie niż klasy docelowe.

3. Wykrywanie kolorów

Wykrywanie kolorów odbywa się podczas sprawdzania słownika każdej klasy (podobnie do 2 kroku wykrywania kształtów). Wykorzystywane jest wcześniej przygotowane zdjęcie z usuniętym tłem. Dla każdego konturu znajdowany jest minimalny prostokąt (cv2.minAreaRect). Aby móc dalej działać na prostokątach obracane są do poziomu lub pionu przy pomocy kombinacji funkcji: cv2.getRotationMatrix2D oraz cv2.warpAffine. Każdy wycięty prostokąt konwertowany jest do przestrzeni kolorów HSV. Przy pomocy funkcji cv2.inRange dla każdego prostokąta tworzone są obrazy binarne zawierające piksele o poszukiwanych kolorach (niebieski, zielony, czerwony, biały żółty). Wartości HSV zostały dobrane eksperymentalnie, ale wszystkie zakresy są większe aby działały także na zdjęciach nie znajdujących się w zbiorze treningowym (przy dosyć podobnych warunkach oświetleniowych). Każdy obraz binarny sprawdzany jest na obecność pikseli o wartościach większych niż 0 i zmienna logiczna ustalana jest na wartość True jeżeli kolor został wykryty. Ostatni zbiór warunków sprawdza czy wystąpiły pojedyncze kolory i odpowiednio dodaje je do listy wyjściowej. Jeżeli wykryto więcej niż jeden kolor, klocek uznawany jest za mieszany.
