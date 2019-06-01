# Stamps replacer

Python rewrite of application named StaMPS https://homepages.see.leeds.ac.uk/~earahoo/stamps/.

**This isn't complete StaMPS rewrite!** This only contains steps that are responsible finding persistent scatterers. There are missing phase unwrapping for complete and standalone work.

The rewrite was done from StaMPS version 3.3b1.

# Tutorial in English

Estonian below/ Eesti keeles allpool.

# Before starting application 

## Setting up environment

Application is written in Python 3.6.3.

First it is recommended to make Python virtual environment using _virtualenv_. Do not use 
Anaconda/ conda virtual environment, because you can't load dependencies to conda. In this example 
command we make virtual environment named "virtenv".
 
`virtualenv virtenv`

Then you need to activate it with this command (in Windows).

`C:\Users\<UserName>\Anaconda3\envs\virtenv\Scripts\activate`

In Linux the command is:

`source virtenv/bin/activate`

Then you load dependencies from file  _env.txt_ or _env\_intel.txt_:

`pip install -r env.txt`

First one is for ordinary Python and _env\_intel.txt_ is for Intel Python MKL. For last one you need to 
download Numpy and SciPy packages separately. For Windows you can find them here 
https://www.lfd.uci.edu/~gohlke/pythonlibs/.

## Properties and parameters

Clone file _StampsReplacer\resources\properties.ini.sample_ and remove _.sample_.

Parameters in the file are:
* __path__ - Input/ data files path.
* __patch_folder__ - When PATCH folder is in other folder (like _tmp_) then put that folder here.
* __geo_file__ - .dim file that is used for processing.
* __save_load_path__ - Save and load path or work directory. Results file can be found in this path.
* __rand_dist_cached__ - Is randomly generated file loaded from temporary files (from 
path __save_load_path\tmp__). It reduces PsEstGamma process time. If the processed file or area is 
new is then you should first delete cached file.

For tests there is seperate properties file _properties.ini_. Clone file from 
_StampsReplacer\tests\resources\properties.ini.sample_ and delete _.sample_ from the end.

Parameters in the file are:
* __tests_files_path__ - Tests files path. Test classes are looking source files from that location.
* __patch_folder__ - When PATCH folder is in other folder (like _tmp_) then put that folder here.

Other parameters are the same. __tests_files_path__ is the same __path__ that in not test properties file. 

__All paths must be absolute.__

## Source files

### Files made with SNAP in this code repository

Pay attention that when you are using files that are in this repository that _pscphase.in_ and _rsc.txt_
correspond to your file system. This is because program reads paths from those files.

### Errors related to missing .npz files

When you see error that tells that there is missing .npz file. Error like this:

`FileNotFoundError: [Errno 2] No such file or directory: 'StampsReplacer/tests/resources/process_saves/ps_files.npz'`

This is because test doesn't find file from previous process. Then you need to run test class that creates that file.
You don't need to run whole test class, it is enough to run only save-load test. After that test sould not raise this error.

## Starting program

For starting program/ processing you need to run _Main.py_.

Like so:
`python Main.py`

This starts from zeroth step and ends final. Runs all steps.

You can also add steps where to start (first parameter) and where to end (end parameter). Example:

`python Main.py 0 5`

This is also equal to previous command.

When we need to run only one step, then both parameters need to be equal. Example:

`python Main.py 0 0`

Also you can show step where to start. It starts from this step and ends with final step. Example:

`python Main.py 0`

This is also equal to first and second command.

What each step do:
0 - load SNAP files and data to Python and calculate some additional that is needed for next steps.
1 - Load SNAP files that where made for StaMPS to Python/ Numpy format.
2 - Estimate phase noise
3 - Select persistent scatterers
4 - Filter/ weed out persistent scatterers
5 - Phase correction

**Please note** that you can't start from first step when you haven't started zeroth step and so on. All steps depend from previous steps.

## Cython

Cython compiles Python code into C. It may affect performance positively in some steps.

### For compiling

Compiled files are written in _cython_setup.py_. You need to run with this command:

`python cython_setup.py build_ext --inplace`

After that Python uses compiled files.

# After process compilation

The results can be found in path that you set in _properties.ini_ file, parameter __save_load_path__.

---

# Eestikeelne õpetus

# Enne käivitamist

## Keskkonna seadistus

Rakendus on kirjutatud Python'i versioonis 3.6.3.

Esmalt tuleb luua virtualenv'iga keskkond. Siin luuakse virtuaalkeskkond virtualenv'iga nimetusega
_virtevn_. Virtuaalkeskkonda ei või teha Anaconda/ conda'ga, sest muidu ei saa laadida sõltuvusi sinna keskkonda.

`virtualenv virtenv`

Siis see aktiveerida Windows’is järgneva käsuga:

`C:\Users\<Kasutaja>\Anaconda3\envs\virtenv\Scripts\activate`

või Linux'is

`source virtenv/bin/activate`

Ning siis laadida sõltuvused failist _env.txt_ või _env\_intel.txt_:

`pip install -r env.txt`

_env.txt_ on tavalisele Python'ile ja _env\_intel.txt_ on mõeldud Intel'i Python'ile. 
Viimase puhul Numpy ja SciPy paketid tuleb käsitsi paigaldada. Need saab Windows'i 
operatsioonisüsteemi jaoks alla laadida siit: https://www.lfd.uci.edu/~gohlke/pythonlibs/.

## Sättefailid ja parameetrite selgitus

Kopeeri fail _StampsReplacer\resources\properties.ini.sample_ ja kustuta sealt lõpust _.sample_.

Parameetrid failis on järgnevad:
* __path__ - Algfailide asukoht
* __patch_folder__ - Kui path kaust on veel mingis kaustas (_tmp_ on üpris levinud) siis tuleb sinna see ka panna.
* __geo_file__ - .dim fail mida kasutatakse töötluses
* __save_load_path__ - Salvestustee. Koht kuhu tulemused (.npz failid) salvestatakse 
* __rand_dist_cached__ - Kas juhuslike arvude massiiv loetakse vahesalvestusest või mitte. 
Vähendab oluliselt PsEstGamma protsessimise aega. Kui tegemist on uute andmetega siis peaks enne 
vahesalvestatud faili ära kustutama. Asub asukohas __save_load_path\tmp__.

Testiklasside jaoks on oma _properties.ini_ fail. Asukohast _StampsReplacer\tests\resources\properties.ini.sample_ tuleb kopeerida fail ja 
kustutada lõpust _.sample_.

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

## Käivitamine

Programm käivitatakse klassist Main, failist Main.py, kus on kaks parameetrit. Esimene parameeter näitab millisest protseduurist alustatakse ja viimane näitab millisega lõpetatakse. Mõlemad parameetrid on täisarvud 0-ist 5-ni. Kui neid ei määra tehakse kogu protsess.

Kui parameetreid ei määra siis tehakse kõik protsessid. See käsk näeb välja selline:

`python Main.py`

Või siis näidata, ette mis sammust alustada ja milliselst lõpetada:

`python Main.py 0 5`

See käsk on võrdne esimesega.

Või kui on soov vaid üks samm teha:

`python Main.py 0 0`

Võib näidata ka vaid algussammu:

`python Main.py 0`

See käsk teeb sama asja mis esimene ja teine.

**Parameetrite numbrid** vastavad järgnevatele protsessidele:
1 - Programmiga SNAP loodud failide lugemine ja töötlemine (klass CreateLonLat).
2 - Algandmete laadimine. Loetakse ja konverteeritakse SNAP eksporditud failid,
mis olid tehtud StaMPS programmile formaadiks, Python/ NumPy failideks
(klass PsFiles).
3 - Faasimüra hindamine (klass PsEstGamma).
4 - Püsivpeegeldajate valik (klass PsSelect).
5 - Püsivpeegeldajate filtreerimine (klass PsWeed).
6 - Faasikorrektsioon (klass PhaseCorrection).

**NB!** Pole võimalik kävitada samme mille eeldusandmeid ei ole. See tähendab, et kohe ei saa alustada teisest sammust töötlust, sest esimese sammu tulem on puudu.

## Cython

Cython kompileerib Python'i kood C koodi. See mõningates protsessi sammudes parandab kiirust.

### Kompileerimiseks

Failid mida kompileeritakse on kirjas _cython_setup.py_. See tuleb ka käivitada kompileerimiseks. Käsk on järgnev:

`python cython_setup.py build_ext --inplace`

Peale mida juba Python kasutab ise kompileeritud faile.

# Protessi lõppedes

Tulemusfailid leiab kaustast, mis seadistati _properties.ini_ faili parameetri __save_load_path__.
