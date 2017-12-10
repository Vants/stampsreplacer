# Stamps replacer

# Enne kävitamist

## Sättefailid ja parameetrite selgitus

Kopperi fail _StampsReplacer\resources\properties.ini.sample_ ja kustuta sealt lõpust sample.

Parameetrid failis on järgnevad:
* __path__ - Algfailide asukoht
* __patch_folder__ - Kui path kaust on veel mingis kaustas (_tmp_ on üpris levinud) siis tuleb sinna see ka panna.
* __geo_file__ - .dim fail mida kasutatakse töötluses
* __save_load_path__ - Salvestustee. Koht kuhu tulemused (.npz failid) salvestatakse 
* __rand_dist_cached__ - Kas juhuslike arvude massiiv loetakse vahesalvestusest või mitte. 
Vähendab oluliselt PsEstGamma protsessimise aega. Kui tegemist on uute andmetega siis peaks enne 
vahesalvestatud faili ära kustutama. Asub asukohas __save_load_path\tmp__.

Testiklasside jaoks on oma properties fail. Asukohast _StampsReplacer\tests\resources\properties.ini.sample_ tuleb kopeerida fail ja 
kustutada lõpust .sample.

Parameetrid failis on järgnevad:
* __tests_files_path__ - Testifailide asukoht. Testiklassid otsivad sealt faile/ algandmeid.
* __patch_folder__ - Kui path kaust on veel mingis kaustas (_tmp_ on üpris levinud) siis tuleb sinna see ka panna. 
Kui PATCH_1 kaust on otse eelnimetatud tests_files_path'is on kõik korras ja selle võib jätta tühjaks.

Parameetrid on samad mis mitte testi parameetrites. __tests_files_path__ on sama mis __path__ mitte testi failides.

__Kõik asukohad on absoluutteena.__   

## Algfailid

### SNAP'i loodud failid siin repositooriumis

Juhul kui kasutada git'is olevaid faile siis peab vaatama, et _pscphase.in_ ja _rsc.txt_ oleksid sinu failisüsteemile vastavad. 
Põhjusena, et seal on asukoht kust võtta algfaile absoluutteena ja see peab olema igas süsteemis oma moodi seadistatud.

### Vead seotud puuduvate .npz failidega

Juhul kui test annab teada, et tal on mõni .npz fail puudu näiteks

`FileNotFoundError: [Errno 2] No such file or directory: 'StampsReplacer/tests/resources/process_saves/ps_files.npz'`

See on seepärast, et testil on puudu üks vahetulemuse salvestus. Siis tasub käivitada vastava faili test kõige enne. 
Kui kõiki teste ei soovi teha siis võib käivitada ainult salvestamise ja laadimise. See teeb selle .npz faili ja siis saab 
teine test ka edasi minna. 

See on seepärast selliselt tehtud, et algandmed võivad kõigil erinevad olla ja seega ka selle programmi loodud vahetulemus.  
