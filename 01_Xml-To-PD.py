import xml.etree.ElementTree as ET
import pandas as pd
import os
from collections import Counter
from lxml import etree
from datetime import datetime

# Counter für Dauer von Skriptausführung
startTime = datetime.now()
print(startTime)

# Datenablageort
directory = 'C:/Users/Path/to/File'

# Hilfsvariablen 
pforms = []
haupt_titels = []
son_titels = []
count_w = []
texts = []

# Konsolenoutput
print('ich roedel......')

# Findet alle relevanten XML-Elemente und speichert sie in PandasDataframe
for roots, subdirectories, files in os.walk(directory):
    for filename in files:
        # fängt korrupte xml-Files ab und ignoriert sie
        try:
            parser1 = ET.XMLParser(encoding='utf-8')
            file = os.path.join(roots, filename)
            tree = ET.parse(file, parser=parser1)
            root = tree.getroot()
            pform_find = root.findall('.//PRAESENTATIONSFORM')
            titel_find = root.findall('.//HAUPTTITEL')
            son_titel_find = root.findall('.//SONSTIGER_TITEL')
            textfind_find = root.findall('.//TEXT')
            count_w_find = root.findall('.//ANZAHL_WORTE')
            # Hier weitere zu testende Features hinzufügen


            if not pform_find or not titel_find or not son_titel_find or not textfind_find or not count_w_find: # or not weitere Feature
                continue
            else:
                for pform in root.iter('PRAESENTATIONSFORM'):
                    pforms.append(pform.text)
                        # gibts das feld überhaupt, wenn nicht Platzhalter, wenn doch to das
                    for haupt_titel in root.iter('HAUPTTITEL'):
                        if haupt_titel.text is not None:
                            haupt_titels.append(haupt_titel.text)
                        else:
                            haupt_titels.append('xxxPlatzhalterxxx') 
                    for son_titel in root.iter('SONSTIGER_TITEL'):
                        if son_titel.text is not None:
                            son_titels.append(son_titel.text)
                        else:
                            son_titels.append('xxxPlatzhalterxxx')
                    for text in root.iter('TEXT'):
                        if text.text is not None:
                            texts.append(text.text)
                        else:
                            texts.append('xxxPlatzhalterxxx') 
                    for count in root.iter('ANZAHL_WORTE'):
                        if count.text.strip() is not None:
                            count_w.append(count.text.strip())
                        else:
                            count_w.append('xxxPlatzhalterxxx')
        except ET.ParseError:
            print('{} is corrupt'.format(file))

# Dataframe-Format
train_data = pd.DataFrame(
        {'pform': pforms,
        'haupt_titel': haupt_titels,
        'sonst_titel': son_titels,
        'volltext': texts,
        'anzahl_w':count_w
    })

# Printet fehlerhafte Dateien aus
print('{} ist irgendwie kaputt'.format(file))


# Zählt unique Keys und speichert sie mit Bezeichnung in Liste
#keys = Counter(pforms).keys() # equals to list(set(words))
#values = Counter(pforms).values() # counts the elements' frequency

# Speichern als Pickle-Datei
train_data.to_pickle('more_raw_data.pkl')

# Print Dauer des Skripts
print(datetime.now() - startTime)