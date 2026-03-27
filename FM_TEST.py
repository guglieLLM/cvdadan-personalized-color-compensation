"""
FM_TEST — Implementazione digitale del Farnsworth-Munsell 100 Hue Test.

GUI PyQt5 per la somministrazione del test FM100 e il calcolo dello scoring
quantitativo Vingrys–King-Smith.  L'output è un file JSON contenente il
profilo CVD 3D [θ, C, S] più i metadati dello scoring completo.

Il profilo generato può essere caricato direttamente nella GUI di inferenza
(z__inference_gui) o usato dalla pipeline (z__pipeline).

Dipendenze:
    PyQt5, numpy, matplotlib.
"""
import copy
import sys
import math
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PyQt5.QtWidgets import (QWidget, QLabel, QDialog, QGridLayout, QPushButton, QApplication, QComboBox, 
                             QMessageBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QHBoxLayout,
                             QFileDialog, QInputDialog, QStyle, QFormLayout, QLineEdit, QToolTip)

from PyQt5.QtGui import QColor, QPalette, QPixmap
from PyQt5.QtCore import Qt, QTimer

import colorsys

import csv
import json

import copy

import hashlib
import random

from datetime import datetime

#############################################
# Dati di riferimento (DATA) per FM100 Hue
#############################################
'''
Dati di riferimento (DATA) per i colori relativi al test di Farnswoth-Munsell 100 Hue 
rappresentati nello spazio cololre CIEUV ed estrapolati dal codice inserito nel paper 
"A Quantitative Scoring Technique For Panel Tests of Color Vision" 
di Algis J. Vingrys and P. Ewen King-Smith
'''
data_FM = [
    85, 43.57, 4.76,  # H = 85
    43.18, 8.03, 44.37, 11.34, 44.07, 13.62, 44.95, 16.04, 44.11, 18.52,
    42.92, 20.64, 42.02, 22.49, 42.28, 25.15, 40.96, 27.78, 37.68, 29.55,
    37.11, 32.95, 35.41, 35.94, 33.38, 38.03, 30.88, 39.59, 28.99, 43.07,
    25.00, 44.12, 22.87, 46.44, 18.86, 45.87, 15.47, 44.97, 13.01, 42.12,
    10.91, 42.85, 8.49, 41.35, 3.11, 41.70, 0.68, 39.23, -1.70, 39.23,
    -4.14, 36.66, -6.57, 32.41, -8.53, 33.19, -10.98, 31.47, -15.07, 27.89,
    -17.13, 26.31, -19.39, 23.82, -21.93, 22.52, -23.40, 20.14, -25.32, 17.76,
    -25.10, 13.29, -26.58, 11.87, -27.35, 9.52, -28.41, 7.26, -29.54, 5.10,
    -30.37, 2.63, -31.07, 0.10, -31.72, -2.42, -31.44, -5.13, -32.26, -8.16,
    -29.86, -9.51, -31.13, -10.59, -31.04, -14.30, -29.10, -17.32, -29.67, -19.59,
    -28.61, -22.65, -27.76, -26.66, -26.31, -29.24, -23.16, -31.24, -21.31, -32.92,
    -19.15, -33.17, -16.00, -34.90, -14.10, -35.21, -12.47, -35.84, -10.55, -37.74,
    -8.49, -34.78, -7.21, -35.44, -5.16, -37.08, -3.00, -35.95, -0.31, -33.94,
    1.55, -34.50, 3.68, -30.63, 5.88, -31.18, 8.46, -29.46, 9.75, -29.46,
    12.24, -27.35, 15.61, -25.68, 19.63, -24.79, 21.20, -22.83, 25.60, -20.51,
    26.94, -18.40, 29.39, -16.29, 32.93, -12.30, 34.96, -11.57, 38.24, -8.88,
    39.06, -6.81, 39.51, -3.03, 40.90, -1.50, 42.80, 0.60, 43.57, 4.76
]

#############################################
# White point per illuminanti (valori approssimativi)
#############################################
'''White point per diversi tipi di illuminanti (valori approssimativi)'''
white_points = {
    "D-53": np.array([94.6, 100.0, 107.0]),
    "D-65": np.array([95.047, 100.0, 108.883]),
    "D-50": np.array([96.421, 100.0, 82.519])
}

def get_DATA_test():
    """Restituisce un iteratore sui valori DATA per FM100 Hue."""
    return iter(data_FM)

def convert_space_color(x, y, color_space, source_white=None, target_white=None):
    """
    Definito per poter convertire i colori dallo spazio di rappresentazione dei colori usato per i calcoli secondo quanto riportato
    nel paper "A Quantitative Scoring Technique For Panel Tests of Color Vision" allo spazio colore target (es. sRGB) per permetterne
    una rappresentazione visuale a schermo secondo la tecnologia supportata
    
    Converte coordinate cromatiche (x,y) (con Y=1) in sRGB di default o nel color space specificato.
    Calcola X e Z assumendo Y=1: X = x/y, Z = (1-x-y)/y.
    
    N.B. Se vengono forniti i valori per source_white e target_white allora applica la trasformazione Bradford.
    
    Parameters:
        - x, y: coordinate nello spazio di colori XY
        - color_space: spazio di colore target --> definisce lo spazio di colore selezionato dal menu' a tendina relativo
        - source_white: definisce, se settato, il punto di bianco di riferimento --> dipende dall'illuminante selezionato dal menu' a tendina relativo
        - target_white: 
    Returns:
        tuple[int, int, int]: Tripletta RGB (0–255) nello spazio colore target.
    """
    if y == 0:
        return (0, 0, 0)
    
    X = x / y
    Y_val = 1.0  # fissiamo Y
    Z = (1 - x - y) / y
    XYZ = np.array([X, Y_val, Z])
    
    if source_white is not None and target_white is not None:
        # Se sono settati dei valori non di default allora applichiamo Matrice Bradford
        M_B = np.array([[0.8951,  0.2664, -0.1614],
                        [-0.7502,  1.7135,  0.0367],
                        [0.0389, -0.0685,  1.0296]])
        
        M_B_inv = np.linalg.inv(M_B)
        cone_source = M_B.dot(source_white)
        cone_target = M_B.dot(target_white)
        
        D = np.diag(cone_target / cone_source)
        XYZ = M_B_inv.dot(D).dot(M_B).dot(XYZ)
        
    if color_space == "sRGB":
        M = np.array([[ 3.2406, -1.5372, -0.4986],
                      [-0.9689,  1.8758,  0.0415],
                      [ 0.0557, -0.2040,  1.0570]])
        gamma = lambda c: 12.92*c if c <= 0.0031308 else 1.055*(c**(1/2.4))-0.055
        
    else:
        # Per semplicità, usiamo sRGB come default
        M = np.array([[ 3.2406, -1.5372, -0.4986],
                      [-0.9689,  1.8758,  0.0415],
                      [ 0.0557, -0.2040,  1.0570]])
        gamma = lambda c: 12.92*c if c <= 0.0031308 else 1.055*(c**(1/2.4))-0.055
        
    rgb_linear = M.dot(XYZ)
    rgb_linear = np.maximum(rgb_linear, 0)
    rgb = np.array([gamma(c) for c in rgb_linear])
    rgb = np.clip(rgb, 0, 1)
    
    return (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))



def get_cap_colors(color_space="sRGB", illuminant="D-53", manual_white=None):
    """
    Estrae le coordinate U e V dai dati e le converte in colori RGB per la visualizzazione.
    
    Per la conversione, si usa il tono (hue) da math.atan2(V, U) con S=0.6 e V=0.8 per rendere i colori meno "brillanti" a schermo
    
    Se l'illuminante target è diverso da D-53, applica l'adattamento Bradford (source white fissato a D-53).
    
    Return:
        - Lista di dizionari per ciascun cap:
            - "num_caps": numero del cap
            - "color": colore RGB (tuple 0-255)
            - "x": U/100
            - "y": V/100
    """
    # recupera i dati definiti nella struttura DATA tratti dal paper
    data_iter = get_DATA_test()
    
    try:
        H = int(next(data_iter))
    except StopIteration:
        return []
    
    cap_data = []
    
    if illuminant == "Manual":
        # Se l'illuminante viene specificato manualmente, ad esempio in condizioni non standardizzate
        if manual_white is None:
            target_white = white_points["D-53"]
        else:
            target_white = np.array(manual_white)
    else:
        # Altrimenti recupera i dati dal dizionario --> se non esiste usa il valore di default
        target_white = white_points.get(illuminant, white_points["D-53"])
        
    source_white = white_points["D-53"]
    
    for n in range(85):
        try:
            U_val = float(next(data_iter))
            V_val = float(next(data_iter))
        except StopIteration:
            break
        
        hue = math.degrees(math.atan2(V_val, U_val))
        
        if hue < 0:
            hue += 360
            
        r, g, b = colorsys.hsv_to_rgb(hue/360.0, 0.6, 0.8)
        rgb = (int(r*255), int(g*255), int(b*255))
        
        if illuminant != "D-53":
            x = U_val/100.0
            y = V_val/100.0
            rgb = convert_space_color(x, y, color_space, source_white, target_white)
            
        cap_data.append({
            "num_caps": n+1,
            "color": rgb,
            "x": U_val/100.0,
            "y": V_val/100.0
        })
        
    return cap_data

#############################################
# Classe per le capsule trascinabili
#############################################
class DraggableLabel(QLabel):
    def __init__(self, cap_info, row, col, is_fixed=False, parent=None, visible_number=False):
        super().__init__(parent)
        self.row = row
        self.col = col
        self.num_caps = cap_info["num_caps"]
        self.cap_info = cap_info
        self.is_fixed = is_fixed    # se la capsula rappresentata è un estremo viene fissata
        self.start_pos = None       # serve per permettere lo swap negli eventi di spostamento e rilascio
        self.original_pos = (row, col)
        self.setFixedSize(50,50)
        self.update_color(cap_info["color"])
        
        self.visible_number = visible_number
        self.setText(str(self.num_caps) if self.visible_number else "")
        self.setAlignment(Qt.AlignCenter)
        
    def update_color(self, _color):
        self._color = _color
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(*self._color))
        self.setAutoFillBackground(True)
        self.setPalette(palette)
        
    # def mousePressEvent(self, event):
    #     if event.button() == Qt.LeftButton and not self.is_fixed:
    #         self.start_pos = event.pos()
    #         self.raise_()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self.is_fixed:
            self.is_dragging = True
            self.start_pos = event.pos()
            self.raise_()
            self.setCursor(Qt.ClosedHandCursor)

                
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and not self.is_fixed and self.start_pos is not None:
            new_pos = self.mapToParent(event.pos() - self.start_pos)
            new_x = new_pos.x()
            
            # Limita lo spostamento dentro il widget genitore
            max_x = self.parentWidget().width() - self.width()
            new_x = max(0, min(new_x, max_x))
        
            self.move(new_x, self.geometry().y())
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and not self.is_fixed:
            parent = self.parentWidget()
                
            if isinstance(parent, FarnsworthTest):
                if parent.is_overlapping_another_label(self):
                    parent.update_positions_on_release(self)
                else:
                    self.move_to_original_position()
            elif isinstance(parent, FarnsworthSequentialDialog):
                parent.reorder_tray_capsules(self.row)
                self.start_pos = None
            
    def move_to_original_position(self):
        grid_x = self.original_pos[1] * self.width()
        grid_y = self.original_pos[0] * self.height()
        self.move(grid_x, grid_y)


#############################################
# Classe principale del test
#############################################
class FarnsworthTest(QWidget):
    def __init__(self, mode_expert=True, visible_number=False):
        super().__init__()
        # recuperiamo la lista dei dizionari delle singole capsule
        # Lista di dizionari per ciascun cap:
        #     - "num_caps": numero del cap
        #     - "color": colore
        #     - "x": U/100  rappresentazione nello spazio colore UV
        #     - "y": V/100
        cap_data = get_cap_colors("sRGB", illuminant="D-53")
        
        self.visible_number = visible_number
        self.mode_expert = mode_expert    # Modalità export -> test, simulazioni, grafici clinici   
        
        # serve per tenere traccia di tutte le finestre aperte e chiuderle se viene chiusa questa
        self.graph_windows = []
        # Tiene traccia della finestra dei risultati
        self.result_dialog = None
        
        if len(cap_data) < 85:
            QMessageBox.warning(self, "Error", "Dati insufficienti per generare i colori dei CAP.")
            sys.exit(1)
            
        # Ordine corretto per FM100 Hue: la prima riga parte con la capsula 85, seguita da 1..21
        self.correct_colors = [
            [cap_data[84]] + cap_data[0:21],
            cap_data[21:42],
            cap_data[42:63],
            cap_data[63:84]
        ]
        
        # print("DEBUG: Composizione initiale dei vassoi in self.correct_colors:")
        # for idx, vassoio in enumerate(self.correct_colors):
        #     print(f"  Vassoio {idx+1}: {[cap['num_caps'] for cap in vassoio]}")

        self.fixed_capsules = [
            [self.correct_colors[0][0]["num_caps"], self.correct_colors[0][-1]["num_caps"]],
            [self.correct_colors[1][0]["num_caps"], self.correct_colors[1][-1]["num_caps"]],
            [self.correct_colors[2][0]["num_caps"], self.correct_colors[2][-1]["num_caps"]],
            [self.correct_colors[3][0]["num_caps"], self.correct_colors[3][-1]["num_caps"]],
        ]

        
        # rimescoliamo i colori considerando gli estremi fissi per ogni vassoio come riportato nel paper
        self.shuffled_colors = [self.shuffle_colors(row.copy()) for row in self.correct_colors]
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Test Farnsworth-Munsell - FM100 Hue - with Vingrys & King-Smith")
        self.setFixedSize(1500,700)
        self.layout = QGridLayout()
        self.labels = []

        for i, row in enumerate(self.shuffled_colors):
            row_labels = []
            for j, cap in enumerate(row):
                # fissiamo gli estremi
                is_fixed = (j==0 or j==len(row)-1) # TRUE se j == 0 oppure j == ultimo elemento
                
                # label = CAPSULA -> riferimento ad una istanza della classe DraggableLabel,
                # che rappresenta una capsula del vassoio corrente per la rappresentazione grafica
                label = DraggableLabel(cap, i, j, is_fixed, self, self.visible_number)
                self.layout.addWidget(label, i, j)
                # ROW_LABEL = VASSOIO -> LISTA di capsule = label -> cioè riferimenti ad oggetti DraggableLabel
                row_labels.append(label)
            #self.labels = TUTTI I VASSOI -> lista di liste di oggetti DraggableLabel -> row_label ->
            # cioè lista di liste (vassoi) di capsule
            self.labels.append(row_labels)
            
        self.submit_button = QPushButton("Esegui il test")
        self.submit_button.clicked.connect(self.submit)
        self.layout.addWidget(self.submit_button, len(self.shuffled_colors), 0, 1, len(self.shuffled_colors[0]))

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Test Mode", "Simulazione senza deficit", "Simulazione deficit minimo", "Simulazione Protanopia - Significativa", "Simulazione Protanopia - Lieve", "Simulazione Deuteranopia - Significativa", "Simulazione Deuteranopia - Lieve", "Simulazione Tritanopia - Significativa", "Simulazione Tritanopia - Lieve"])
        self.mode_combo.setStyleSheet("QComboBox { text-align: center; }")
        self.mode_combo.currentIndexChanged.connect(self.mode_changed)




        # -->    COMMENTATI TEMPORANEAMENTE  <--
        
        # self.illuminant_combo = QComboBox()
        # self.illuminant_combo.addItems(["D-53", "D-65", "D-50", "Manual"])
        # self.illuminant_combo.setStyleSheet("QComboBox { text-align: center; }")
        # self.illuminant_combo.currentIndexChanged.connect(self.illuminant_changed)

        # self.color_space_combo = QComboBox()
        # self.color_space_combo.addItems(["sRGB", "Adobe RGB", "ProPhoto RGB", "Display P3"])
        # self.color_space_combo.setStyleSheet("QComboBox { text-align: center; }")
        # self.color_space_combo.currentIndexChanged.connect(self.color_space_changed)
        
        
        
        
        
        # ---- Pulsante Mostra/Nascondi numeri ---------------------------------
        self.toggle_numbers_btn = QPushButton()
        self.toggle_numbers_btn.setCheckable(True)
        self.toggle_numbers_btn.setChecked(self.visible_number)
        self._update_toggle_button_text()
        self.toggle_numbers_btn.clicked.connect(self._toggle_numbers)

        self.lock_labels()  # BLOCCA
        
        if self.get_mode_test():    # TRUE -> expert mode
            self.layout.addWidget(self.mode_combo, len(self.shuffled_colors)+1, 0, 1, len(self.shuffled_colors[0]))
            
            # -->    DESELEZIONATI TEMPORANEAMENTE  <--
            
            #self.layout.addWidget(self.illuminant_combo, len(self.shuffled_colors)+2, 0, 1, len(self.shuffled_colors[0]))
            #self.layout.addWidget(self.color_space_combo, len(self.shuffled_colors)+3, 0, 1, len(self.shuffled_colors[0]))
            
            self.lock_labels(lock=False)    #  SBLOCCA
        
        self.layout.addWidget(self.toggle_numbers_btn, len(self.shuffled_colors)+4, 0, 1, len(self.shuffled_colors[0]))

        self.setLayout(self.layout)
        
        
    def set_mode_test(self, mode=True):
        self.mode_expert = mode
    
    def get_mode_test(self):
        return self.mode_expert
    
    
    def shuffle_colors(self, row):
        """Mescola solo 5 capsule mobili a caso, lasciando le altre in ordine corretto."""
        import random

        if len(row) <= 2:
            return row  # niente da mescolare se ci sono solo due capsule (fisse)

        # Copia la riga per non modificare l'originale direttamente
        shuffled = row[:]

        # Indici delle capsule mobili (escludi prima e ultima)
        mobili_indices = list(range(1, len(row) - 1))

        # Scegli 5 da mescolare (o tutte se sono meno di 5)
        mix_indices = random.sample(mobili_indices, min(5, len(mobili_indices)))

        # Estrai i valori da mescolare
        mix_values = [shuffled[i] for i in mix_indices]
        random.shuffle(mix_values)

        # Reinserisci i valori mescolati nelle posizioni originali
        for idx, val in zip(mix_indices, mix_values):
            shuffled[idx] = val

        return shuffled


    def is_overlapping_another_label(self, label):
        for other in self.labels[label.row]:
            if other != label and not other.is_fixed:
                if label.geometry().intersects(other.geometry()):
                    return True
        return False

    # funzione di aggiornamento delle posizioni delle capsule dopo il rilascio di una casella
    def update_positions_on_release(self, label):
        row_labels = self.labels[label.row]
        current_pos = label.geometry().center().x()
        new_col = None
        min_dist = float('inf')
        
        for target in row_labels:
            if target.is_fixed or target == label:
                continue
            dist = abs(current_pos - target.geometry().center().x())
            if dist < min_dist:
                min_dist = dist
                new_col = target.col
                
        if new_col is not None and new_col != label.col:
            old_col = label.col
            if new_col > old_col:
                for col in range(old_col, new_col):
                    self.labels[label.row][col] = self.labels[label.row][col+1]
                    self.labels[label.row][col].col -= 1
            elif new_col < old_col:
                for col in range(old_col, new_col, -1):
                    self.labels[label.row][col] = self.labels[label.row][col-1]
                    self.labels[label.row][col].col += 1
            self.labels[label.row][new_col] = label
            label.col = new_col
        self.update_ui()

    def update_ui(self):
        for row in self.labels:
            for label in row:
                label.update_color(label._color)
        for i, row in enumerate(self.labels):
            for label in row:
                self.layout.addWidget(label, i, label.col)
    
    
    def _update_toggle_button_text(self):
        self.toggle_numbers_btn.setText(
            "Nascondi numeri capsule" if self.toggle_numbers_btn.isChecked()
            else "Mostra numeri capsule"
        )

    def _toggle_numbers(self):
        self.visible_number = self.toggle_numbers_btn.isChecked()
        for row in self.labels:
            for lbl in row:
                lbl.setText(str(lbl.num_caps) if self.visible_number else "")
        self._update_toggle_button_text()


    ############################################################################
    # Aggiorna il colore delle capsule in base allo spazio colore e illuminante
    ############################################################################
    def update_capsule_colors(self):
        selected_space = self.color_space_combo.currentText()
        illuminant = self.illuminant_combo.currentText()
        manual_white = None
        
        if illuminant == "Manual":
            text, ok = QInputDialog.getText(self, "Inserisci White Point Manuale", "Inserisci X,Y,Z separati da virgola:")
            if ok:
                try:
                    values = [float(v.strip()) for v in text.split(",")]
                    if len(values)==3:
                        manual_white = values
                    else:
                        raise ValueError
                except ValueError:
                    QMessageBox.warning(self, "Error", "Input non valido. Uso default D-53.")
                    illuminant = "D-53"
            else: # default
                illuminant = "D-53"
                
        new_cap_data = get_cap_colors(selected_space, illuminant, manual_white)
        
        for row in self.correct_colors:
            for cap in row:
                for new_cap in new_cap_data:
                    if new_cap["num_caps"] == cap["num_caps"]:
                        cap["color"] = new_cap["color"]
                        break
                    
        for i, row in enumerate(self.shuffled_colors):
            for j, cap in enumerate(row):
                for label in self.labels[i]:
                    if label.num_caps == cap["num_caps"]:
                        label.update_color(cap["color"])
                        
        self.update_ui()


    def color_space_changed(self, index):
        self.update_capsule_colors()

    def illuminant_changed(self, index):
        self.update_capsule_colors()

    #####################################################################
    # Metodo per simulare il riordino in base alla modalità selezionata
    #####################################################################
    '''
    Definisce diverse disposizioni delle capsule in base alla modalita' selezionata da menu' a tendina relativo
    Descrizione mode (la disposizione corrispondente si intende esclusi gli estremi di ogni vassoio che sono sempre fissi per riferimento)
        - Test Mode: vengono disposte le capsule interne in maniera casuale per poter essere riordinate manualmente per effettuare il test
        - Simulazione senza deficit: vengono disposte le capsule interne in maniera perfettamente ordinata secondo la disposizione corretta di riferimento
        - Simulazione deficit minimo: vengono disposte in maniera correttamente ordinata tranne alcune mantenendosi al di sotto della 
          soglia che definisce la presenza di un deficit visivo ai colori
        - Simulazione protanopia, tritanopia, deuteranopia : per ognuna di questa viene definisca una disposizione delle capsule che permetta il rilevamento
          del corrispettivo deficit visivo ai colori.
    '''

    
    def mode_changed(self, index):
        mode = self.mode_combo.currentText()
        
        if mode == "Test Mode":
            new_order = copy.deepcopy(self.shuffled_colors)
        elif mode == "Simulazione senza deficit":
            new_order = copy.deepcopy(self.correct_colors)
        elif mode == "Simulazione deficit minimo":
            new_order = copy.deepcopy(self.simulation_min_deficit())
        elif mode == "Simulazione Protanopia - Significativa":
            new_order = copy.deepcopy(self.simulation_with_deficit("protanopia-significativa"))
        elif mode == "Simulazione Protanopia - Lieve":
            new_order = copy.deepcopy(self.simulation_with_deficit("protanopia-lieve"))
        elif mode == "Simulazione Tritanopia - Significativa":
            new_order = copy.deepcopy(self.simulation_with_deficit("tritanopia-significativa"))
        elif mode == "Simulazione Tritanopia - Lieve":
            new_order = copy.deepcopy(self.simulation_with_deficit("tritanopia-lieve"))
        elif mode == "Simulazione Deuteranopia - Significativa":
            new_order = copy.deepcopy(self.simulation_with_deficit("deuteranopia-significativa"))
        elif mode == "Simulazione Deuteranopia - Lieve":
            new_order = copy.deepcopy(self.simulation_with_deficit("deuteranopia-lieve"))
        else:
            new_order = copy.deepcopy(self.shuffled_colors)
        #lista vuota
        if(len(new_order)==0):
            new_order = copy.deepcopy(self.shuffled_colors)
            
        self.apply_simulation_order(new_order)
        

    def apply_simulation_order(self, new_order):
        new_labels = []
        for i, row in enumerate(new_order):
            new_row = [None] * len(row)
            for j, cap in enumerate(row):
                for label in self.labels[i]:
                    if label.num_caps == cap["num_caps"]:
                        new_row[j] = label
                        label.col = j
                        break
                    
            new_labels.append(new_row)
            
        self.labels = new_labels.copy()
        self.update_ui()
        
        # # Debug finale
        # print("DEBUG: self.labels dopo apply_simulation_order()")
        # for row_index, row in enumerate(self.labels):
        #     print(f"  RIGA {row_index}:", [label.num_caps for label in row])
            

    def simulation_min_deficit(self):
        """
        Simula caso con deficit cromatico minimo ma con ordine minimamente mescolato.
        """
        user_order = []
        for row_index, row in enumerate(self.correct_colors):
            row_order = [{"color": capsule["color"], "num_caps": capsule["num_caps"]} for capsule in row]
            middle_caps = row_order[1:-1]  # Escludi gli estremi (non modificabili)

            # Introduci una mescolanza moderata (difficoltà senza deficit)
            if len(middle_caps) > 3:
                middle_caps[0], middle_caps[1] = middle_caps[1], middle_caps[0]  # Scambia le prime due
                middle_caps[-1], middle_caps[-2] = middle_caps[-2], middle_caps[-1]  # Scambia le ultime due

            row_order[1:-1] = middle_caps
            user_order.append(row_order)
        
        # Aggiungi un print per vedere quali cappucci hai ottenuto in user_order
        print("DEBUG: Risultato di simulation_min_deficit()")
        for row in user_order:
            print([cap["num_caps"] for cap in row])

        return user_order
    
    
    def simulation_with_deficit(self, type_defect=None):
        """
        Crea un ordine utente (user_order) simulando un difetto cromatico definito in type_defect
        """
        if type_defect==None:
            return []
        
        # Inizializza l'ordine utente con una copia dell'ordine corretto
        user_order = copy.deepcopy(self.correct_colors)

        # PROTANOPIA -> RED
        if type_defect == 'protanopia-significativa':
            user_order = copy.deepcopy(self.simulate_protanopia_significativa(user_order))
        elif type_defect == 'protanopia-lieve':
            user_order = copy.deepcopy(self.simulate_protanopia_lieve(user_order))
        # DEUTERANOPIA -> GREEN
        elif type_defect == 'deuteranopia-significativa':
            user_order = copy.deepcopy(self.simulate_deuteranopia_significativa(user_order))
        elif type_defect == 'deuteranopia-lieve':
            user_order = copy.deepcopy(self.simulate_deuteranopia_lieve(user_order))
        # TRITANOPIA -> BLUE
        elif type_defect == 'tritanopia-significativa':
            user_order = copy.deepcopy(self.simulate_tritanopia_significativa(user_order))
        elif type_defect == 'tritanopia-lieve':
            user_order = copy.deepcopy(self.simulate_tritanopia_lieve(user_order))
            
        # Aggiungi un print per vedere quali cappucci hai ottenuto in user_order
        print("DEBUG: Risultato di simulation_with_deficit()")
        for row in user_order:
            print([cap["num_caps"] for cap in row])
            
        return user_order


    def simulate_protanopia_significativa(self, order):
        order[0][1:21] = [  
            order[0][13], order[0][9], order[0][4], order[0][15], order[0][16], order[0][3], 
            order[0][14], order[0][20], order[0][1], order[0][7], order[0][8],
            order[0][11], order[0][18], order[0][2], order[0][17], order[0][12],
            order[0][10], order[0][5], order[0][19], order[0][6]
        ]
        
        # Seconda riga: Regione 22-42
        order[1][1:20] = [
            order[1][4], order[1][7], order[1][16], order[1][9], order[1][15], order[1][6], 
            order[1][8], order[1][11], order[1][17], order[1][2], order[1][1],
            order[1][14], order[1][3], order[1][19], order[1][18], order[1][13], order[1][12], order[1][10], order[1][5]
        ]
        
        return order
    
    
    def simulate_protanopia_lieve(self, order):
        order[0][1:21] = [  
            order[0][13], order[0][9], order[0][4], order[0][15], order[0][16], order[0][3], 
            order[0][14], order[0][20], order[0][1], order[0][7], order[0][8],
            order[0][11], order[0][18], order[0][2], order[0][17], order[0][12],
            order[0][10], order[0][5], order[0][19], order[0][6]
        ]
        
        # Seconda riga: Regione 22-42
        # order[1][1:20] = [
        #     order[1][4], order[1][7], order[1][16], order[1][9], order[1][15], order[1][6], 
        #     order[1][8], order[1][11], order[1][17], order[1][2], order[1][1],
        #     order[1][14], order[1][3], order[1][19], order[1][18], order[1][13], order[1][12], order[1][10], order[1][5]
        # ]
        
        return order


    def simulate_deuteranopia_significativa(self, order):        
        # Quarta riga: Regione 65-84 (1:19), estremi fissi
        order[2][1:20] = [
            order[2][1], order[2][3], order[2][2], order[2][4], order[2][5], order[2][6], 
            order[2][8], order[2][9], order[2][10], order[2][7], order[2][11],
            order[2][13], order[2][12], order[2][19], order[2][14], order[2][16],
            order[2][17], order[2][15], order[2][18],
        ]
        order[3][1:20] = [
            order[3][16], order[3][1], order[3][3], order[3][6], order[3][2], order[3][8], 
            order[3][7], order[3][10], order[3][4], order[3][9], order[3][12],
            order[3][11], order[3][5], order[3][14], order[3][15], order[3][17],
            order[3][13], order[3][18], order[3][19],
        ]
        return order
    
    
    def simulate_deuteranopia_lieve(self, order):        
        # Quarta riga: Regione 65-84 (1:19), estremi fissi
        order[3][1:20] = [
            order[3][2], order[3][1], order[3][4], order[3][3], order[3][6], order[3][5], 
            order[3][8], order[3][7], order[3][10], order[3][9], order[3][12],
            order[3][11], order[3][14], order[3][13], order[3][16], order[3][15],
            order[3][18], order[3][17], order[3][19],
        ]
        return order
    
    
    def simulate_tritanopia_significativa(self, order):
        # Prima riga: Regione 1:22
        order[0][1:21] = [
            order[0][3], order[0][6], order[0][1], order[0][15], order[0][8], order[0][5], order[0][9], 
            order[0][7], order[0][4], order[0][10], order[0][11], order[0][12], order[0][13], 
            order[0][14], order[0][2], order[0][16], order[0][17], order[0][18], 
            order[0][19], order[0][20],
        ]
        # Terza riga: Regione 46-52 (3-9), estremi fissi
        order[2][1:20] = [
            order[2][1], order[2][4], order[2][6], order[2][8], order[2][3], order[2][5], order[2][7], 
            order[2][9], order[2][10], order[2][11], order[2][2], order[2][13], order[2][14], 
            order[2][15], order[2][16], order[2][17], order[2][18], order[2][19], order[2][12]
        ]
        
        return order
    
    
    def simulate_tritanopia_lieve(self, order):
        # Prima riga: Regione 1:22
        order[0][1:21] = [
            order[0][3], order[0][6], order[0][1], order[0][15], order[0][8], order[0][5], order[0][9], 
            order[0][7], order[0][4], order[0][10], order[0][11], order[0][12], order[0][13], 
            order[0][14], order[0][2], order[0][16], order[0][17], order[0][18], 
            order[0][19], order[0][20],
        ]
        # Terza riga: Regione 46-52 (3-9), estremi fissi
        order[2][1:20] = [
            order[2][1], order[2][4], order[2][6], order[2][8], order[2][3], order[2][5], order[2][7], 
            order[2][9], order[2][10], order[2][11], order[2][2], order[2][13], order[2][14], 
            order[2][15], order[2][16], order[2][17], order[2][18], order[2][19], order[2][12]
        ]
        
        return order


    ####################################################################
    #  BLOCCA LE CAPSULE  (solo visualizzazione)
    ####################################################################
    def lock_labels(self, lock=True):
        """
        Rende le capsule non trascinabili e disabilita qualsiasi clic.
        Chiamato quando si vuole mostrare la board finale in sola lettura.
        """
        for row in self.labels:
            for lbl in row:
                if lock:
                    lbl.is_fixed = True       # impedisce il drag (mousePressEvent lo rispetta)
                else:
                    lbl.is_fixed = False       # Sblocca il drag (mousePressEvent lo rispetta)



    #####################################################################
    # Calcolo dell'errore secondo il metodo BASIC (FM100)
    #####################################################################
    '''
        Questo codice è la conversione in python del programma BASIC usato per valutare la capacità di discriminazione dei colori 
        attraverso test standardizzati (come il D-15, il D-15 desaturato e l'FM-100 Hue), estratto dal paper "A Quantitative Scoring Technique For Panel Tests of Color Vision",
        Nello specifico viene riportata soltanto la parte relativa al test FM-100 HUE essendo di solo nostro interesse.
        Questa versione del codice in python è una versione riadattata per poter essere utilizzata con il sistema di posizionamento delle capsule
        definito tramite GUI per trascinamento delle stesse, nel programma originale in BASIC veniva richiesto l'inserimento testuale di ogni singola capsula.
        I controlli e gli adeguamenti rispettano esattamente la logica definita nella versione originale.
        
        Descrizione dettagliata del codice Basic convertito:

        Dati predefiniti (DATA):
            Il programma contiene tre blocchi di dati, uno per ciascun test (noi abbiamo riportato soltanto il terzo per FM-100 HUE) che rappresentano:
                - Il numero di "caps" (ossia il numero di targhette o campioni di colore) da ordinare.
                - Una sequenza di coppie di valori (U e V) che rappresentano coordinate in uno spazio colore
                  Questi valori, nel contesto dei test cromatici, definiscono la posizione dei colori in una griglia o mappa cromatica

        Simulazione del comando RESTORE/READ:  
            Il codice utilizza una funzione (simile al comando BASIC RESTORE) per creare un iteratore sui dati specifici del test scelto.
            In questo modo:
            - Il primo valore letto rappresenta il numero di caps (H)
            - Vengono poi letti, in sequenza (considerato il modo di accedere ai dati del programma in BASIC), 2*(H+1) valori: 
                -- per ogni indice da 0 a H (numero del caps), il primo valore rappresenta U e il secondo V


        Impostazione del cap iniziale:
            In base al test nel codice originale per l'FM-100 Hue viene impostato il cap iniziale (C[0]) pari a quello presente alla posizione 85;
            viceversa veniva impostato a 0.
            Questo passaggio replica la logica del BASIC che gestisce diversamente l'FM-100 rispetto agli altri test.
            
            -- AGGIORNAMENTO:
            Strategia di chiusura:
            Il metodo originale si basa sull'idea che le capsule debbano essere considerate in ordine circolare - cioè, l'ultimo cap si collega al primo.
            Questo implica che, per calcolare le differenze tra capsule adiacenti (DU e DV), l'ordine deve essere "chiuso".
            Quindi si imposta il primo elemento dell'array di confronto come l'ultimo cap dell'ordine utente (ad esempio, C[0] = user_order[-1]) e poi riempire 
            il resto con C[n] = user_order[n-1] garantisce che la differenza tra il primo e l'ultimo cap sia calcolata.
            Questo è esattamente quanto suggerito dalla versione originale BASIC, dove il cap di riferimento (il cap 85) funge da "punto di chiusura" per la serie.
            

        Elaborazione dei Dati e Calcoli

        - Calcolo delle differenze CVD (vettori di colore): 
            Il programma calcola la differenza tra le coordinate dei colori in due capsule consecutive, 
            secondo l'ordine inserito dall'utente:
                - DU = U(C[n]) - U(C[n-1])
                - DV = V(C[n]) - V(C[n-1])

        - Calcolo delle somme:
            Vengono calcolati:
            - La somma dei quadrati di queste differenze (U2 e V2)
            - La somma dei prodotti incrociati (UV)

        - Determinazione dell'angolo e dei momenti:
            - Calcolo dell'angolo (A0):  
                Se la differenza U2 - V2 (indicata con D) è uguale a zero, allora A0 viene impostato a un valore fisso (0.7854 rad).
                Altrimenti se è diversa da zero, A0 viene calcolato usando l'arctan (ossia l'inversa della tangente) in modo da determinare
                la rotazione che "diagonalizza" la dispersione dei dati
            - Momenti I0 e I1:  
                Questi rappresentano le "varianze" lungo gli assi maggiori e minori, calcolati mediante le funzioni trigonometriche
                (seno e coseno) applicate all'angolo A0 e a quello perpendicolare (A1)
            - Scambio di valori:
                Se il momento maggiore (I0) risulta essere inferiore al momento minore (I1), i valori vengono scambiati per garantire
                che I0 rappresenti sempre la "maggiore dispersione"
            - Calcolo dei raggi (R0 e R1):  
                I raggi maggiori e minori sono ottenuti come la radice quadrata di I0/H e I1/H, rispettivamente
                
                -- AGGIORNAMENTO PER LA NORMALIZZAZIONE NEL CALCOLO DEI RAGGI:
                Studi successivi e analisi comparate (es. Bowman, Cameron, e altri):
                Alcuni autori hanno proposto di normalizzare per il numero di transizioni libere, cioè (H-1), perché in un ordine lineare ci sono H-1 differenze.
                Questa scelta può fornire un valore medio dell'errore per transizione, rendendo il TES più comparabile tra test con numeri differenti di capsule.
                Ad esempio, Bowman e Cameron ( esistono diversi riferimenti in studi clinici successivi) hanno osservato che una normalizzazione per (H-1) 
                consente di ridurre l'effetto di un cap fisso (il cap di riferimento) e di ottenere un indice di errore che rispecchia meglio la dispersione complessiva.
                    
                    ---Scelta del divisore:
                    La decisione tra usare H o (H-1) dipende quindi dall'obiettivo: se si vuole replicare fedelmente il calcolo del programma BASIC originale, 
                    si usa H; se invece si cerca di ottenere una media dell'errore per transizione, (H-1) potrebbe essere più indicato.
                    Questa discussione è stata riportata in diversi lavori (ad esempio, basti vedere anche [Bowman et al., 1990] e studi clinici successivi) 
                    e rimane oggetto di scelta metodologica in base ai dati normativi di riferimento.


            - Errore Totale (R):
                Viene calcolato come la radice quadrata della somma dei quadrati dei due raggi

        - Indici specifici del test:
            In base al tipo di test scelto, viene assegnato un valore standard (R2) e vengono calcolati:
            - S-INDEX: Rapporto tra R0 e R1
            - C-INDEX: Rapporto tra R0 e il valore standard R2
            
        Questi indici sono utilizzati per valutare la severità e il tipo di eventuale deficienza cromatica.


        Return:
        Viene creato e return in output un dizionario contenente:
            - Angolo (in gradi): Il valore dell'angolo calcolato (convertito in gradi)
            - Raggi maggiore e minore (MAJ RAD e MIN RAD)
            - Errore Totale (TOT ERR)
            - S-INDEX e C-INDEX: Indici che permettono una valutazione comparativa rispetto a valori standard


        Riepilogo:
        - Raccoglie dati preimpostati relativi alle coordinate dei colori per tre diversi test cromatici
        - Permette all'utente di inserire l'ordine in cui dispone le capsule di colore durante il test
        - Calcola le differenze tra le posizioni dei colori e, tramite operazioni matematiche 
          (somme di quadrati, prodotti incrociati, trasformazioni trigonometriche), determina:
            - L'orientamento della dispersione dei dati cromatici
            - La "varianza" lungo gli assi maggiori e minori
            - Indici che permettono di quantificare l'errore complessivo nell'ordinamento e, indirettamente, 
              la qualità della percezione cromatica del soggetto
              
    '''
    def calculate_error_basic_PCA(self):
        data_iter = get_DATA_test()

        try:
            H = int(next(data_iter))
        except StopIteration:
            QMessageBox.warning(self, "Error", "Errore nella lettura dei dati.")
            return None

        U, V = [0.0] * (H + 1), [0.0] * (H + 1)

        for n in range(0, H + 1):
            try:
                U[n] = float(next(data_iter))
                V[n] = float(next(data_iter))
            except StopIteration:
                QMessageBox.warning(self, "Error", "Dati insufficienti per U e V.")
                return None

        user_order_rows = self.extract_user_order_by_rows()

        # Ricostruzione dell'ordine utente su tutto il test (chiusura circolare globale)
        user_order = [cap for row in user_order_rows for cap in row]
        
        if len(user_order) < H:
            QMessageBox.warning(self, "Error", "Numero insufficiente di CAP nell'ordine utente.")
            return None



        # Ora calcoliamo il TES per ogni vassoio separatamente, senza chiusura locale
        TES_vassoio = []

        U2_total, V2_total, UV_total = 0.0, 0.0, 0.0

        for row in user_order_rows:
            if len(row) < 2:
                continue  # Salta vassoi troppo piccoli

            # Calcola solo le differenze interne senza chiusura su ogni vassoio
            U2, V2, UV = 0.0, 0.0, 0.0

            # Calcolo delle differenze cromatiche tra le capsule adiacenti:
            # Vengono calcolati U^2,V^2,UV, necessari per il calcolo dei momenti di inerzia
            for n in range(1, len(row)):
                du = U[row[n] - 1] - U[row[n - 1] - 1]
                dv = V[row[n] - 1] - V[row[n - 1] - 1]
                # print(f"DEBUG: Step {n}: C[n] = {C[n]}, DU={du}, DV={dv}")
                U2 += du * du
                V2 += dv * dv
                UV += du * dv
                
            
            # Calcolo TES locale (per ogni vassoio)
            D = U2 - V2
            
            # Si preferisce usare 1e-12 a 0 perchè è un approccio più robusto in un contesto moderno con double precision,
            # dove facilmente possono comparire valori molto piccoli invece di 0 esatto
            # -> π/4 = 0.7854
            # -> +π/2 = 1.5708
            A0 = 0.5 * math.atan2(2 * UV, D) if abs(D) >= 1e-12 else (math.pi / 4)
            # Calcolo dell'angolo perpendicolare
            A1 = A0 + (math.pi / 2) if A0 < 0 else A0 - (math.pi / 2)

            # Momenti principali di inerzia:
            # Formula per il calcolo del momento massimo e minimo di inerzia
            # Segue la decomposizione spettrale del tensore d'inerzia come descritto nel paper
            I0 = (U2 * (math.sin(A0) ** 2) + V2 * (math.cos(A0) ** 2) - 2 * UV * math.sin(A0) * math.cos(A0))
            I1 = (U2 * (math.sin(A1) ** 2) + V2 * (math.cos(A1) ** 2) - 2 * UV * math.sin(A1) * math.cos(A1))

            
            # Scambio degli assi principali se I0 < I1:
            # Garantisce che I0 sia sempre il momento massimo e I1 il minimo evitando errori nei calcoli successivi
            if I0 < I1:
                A0, A1 = A1, A0
                I0, I1 = I1, I0

            R0_local = math.sqrt(I0 / (len(row) - 1))
            R1_local = math.sqrt(I1 / (len(row) - 1))
            tes_local = math.sqrt(R0_local ** 2 + R1_local ** 2)

            TES_vassoio.append(tes_local)

            # Accumuliamo per il calcolo globale
            U2_total += U2
            V2_total += V2
            UV_total += UV


        # Analisi dei vassoi problematici
        media_TES = np.mean(TES_vassoio)
        deviazione_TES = np.std(TES_vassoio)
        vassoi_problematici = [i+1 for i, tes in enumerate(TES_vassoio) if tes > (media_TES + deviazione_TES)]


        # Normalizziamo secondo Bowman su (H-1) invece di len(row)-1 per singoli vassoi
        D_total = U2_total - V2_total
        
        # Dopo il ciclo: calcoliamo ora A0_total e A1_total
        A0_total = 0.0
        A1_total = 0.0
        
        A0_total = 0.5 * math.atan2(2 * UV_total, D_total) if abs(D_total) >= 1e-12 else 0.7854
        A1_total = A0_total + 1.5708 if A0_total < 0 else A0_total - 1.5708

        I0_total = 0.0
        I1_total = 0.0
        
        I0_total = (U2_total * (math.sin(A0_total) ** 2) + 
                    V2_total * (math.cos(A0_total) ** 2) - 
                    2 * UV_total * math.sin(A0_total) * math.cos(A0_total))
        
        I1_total = (U2_total * (math.sin(A1_total) ** 2) + 
                    V2_total * (math.cos(A1_total) ** 2) - 
                    2 * UV_total * math.sin(A1_total) * math.cos(A1_total))

        if I0_total < I1_total:
            A0_total, A1_total = A1_total, A0_total
            I0_total, I1_total = I1_total, I0_total

        # N.B. Normalizzazione dei raggi (R0 e R1) su (H-1), come suggerito nella letteratura successiva,
        # coerente con gli studi successivi (Bowman et al.) che migliorano il metodo originale,
        # suggerito anche da Vingrys
        R0_final = 0.0
        R1_final = 0.0
        TES_final = 0.0
        
        R0_final = math.sqrt(I0_total / (H - 1))
        R1_final = math.sqrt(I1_total / (H - 1))
        # TES_FINAL = R_total
        TES_final = math.sqrt(R0_final ** 2 + R1_final ** 2)
        
        
        R2 = 2.525249   # valore standard Farnsworth
        C_index = R0_final / R2
        S_index = R0_final / R1_final if R1_final > 1e-12 else 0.0  # Evita Infinity e instabilità numerica
        

            
            
        # Calcolo della PCA globalmente come previsto dal paper
        
        # Applicazione della PCA per confermare i risultati del metodo del momento di inerzia:
        # La PCA viene usata anche per verificare che i momenti principali calcolati siano corretti
        # Il punteggio TES-PCA è una misura alternativa della gravità del difetto cromatico
        
        # Integrazione PCA (eigen-decomposizione)
        vingrys_matrix_total = np.array([[U2_total, UV_total],
                                        [UV_total, V2_total]])

        eigenvalues, eigenvectors = np.linalg.eig(vingrys_matrix_total)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        TES_pca = math.sqrt(sum(eigenvalues) / (H-1))  # Normalizzazione secondo Bowman


        ################ MATRICE DI CORREZIONE PCA ##################################
                                            
        pca_corr_matrix = self.compute_pca_correction_matrix(vingrys_matrix_total, vassoi_problematici)
        
        # print("Matrice di correzione PCA:")
        # print(pca_corr_matrix)

        # # Salva la matrice di correzione in formato .npy per usi futuri :
        
        # np.save("correction_matrix.npy", pca_corr_matrix)
        # print("Matrice di correzione salvata in 'correction_matrix.npy'.")
        
        ################ END - MATRICE DI CORREZIONE PCA ############################
        
        
        
        ################  CLASSIFICATION ######################
        
        classification = self.classify_defect(math.degrees(A0_total), TES_final, C_index, S_index, vassoi_problematici)
        
            
        ################ END CLASSIFICATION ###################
        
        
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "TestV": "Vingrys-King",
            "Confusion Angle (degrees)": math.degrees(A0_total),
            "C-index": C_index,
            "S-index": S_index,
            "Major Radius": R0_final,
            "Minor Radius": R1_final,
            "TES (Vingrys)": TES_final,
            "TES Local": TES_vassoio,
            "Vassoi Problematici": vassoi_problematici,
            "TestP": "PCA",
            "TES (PCA)": TES_pca,
            "Eigenvalues": eigenvalues,
            "Eigenvectors": eigenvectors,
            "U2": U2_total,
            "V2": V2_total,
            "UV": UV_total,
            "Matrix Correction": pca_corr_matrix,
            "Classification": classification
        }

        print("DEBUG: Risultati finali del calcolo:")
        for k, v in result.items():
            print(f"  {k} = {v}")

        return result


    def extract_user_order_by_rows(self):
        """
        Estrae l'ordine delle capsule dalla GUI preservando la struttura per righe.
        Se il test è FM 100-HUE, assicura che il cap di riferimento (85) sia in prima posizione del primo vassoio.
        Restituisce una lista di liste, dove ogni sottolista rappresenta un vassoio (riga).
        """
        user_order_rows = []
        for row in self.labels:
            # Per ciascuna riga, prendi i numeri dei CAP nell'ordine attuale
            row_order = [label.num_caps for label in row]
            user_order_rows.append(row_order)

        # Per FM 100-HUE, il CAP 85 deve essere in testa al primo vassoio.
        # if user_order_rows:
        #     if 85 in user_order_rows[0]:
        #         index_85 = user_order_rows[0].index(85)
        #         if index_85 != 0:
        #             # Scambia il CAP trovato con quello in posizione 0
        #             user_order_rows[0][0], user_order_rows[0][index_85] = user_order_rows[0][index_85], user_order_rows[0][0]
        #     else:
        #         # Se 85 non è nel primo vassoio, cercalo in un'altra riga e spostalo nel primo
        #         for r in range(1, len(user_order_rows)):
        #             if 85 in user_order_rows[r]:
        #                 user_order_rows[r].remove(85)
        #                 user_order_rows[0].insert(0, 85)
        #                 break
        
        print("DEBUG: user_order_rows in extract_user_order_by_rows():")
        for i, riga in enumerate(user_order_rows):
            print(f"  RIGA {i} -> {riga}")

        return user_order_rows




    ###############################################################################
    # Funzione plot_results: grafico polare per il test Farnsworth (error ellipse)
    ###############################################################################
    def plot_error_ellipse(self, basic_results, ax=None):
        
        """
        Plotta l'ellisse degli errori come nel paper di Vingrys & King-Smith.
        
        Parametri attesi in basic_results:
        - "Confusion Angle (degrees)": angolo di confusione in gradi
        - "Major Radius": raggio maggiore (R0)
        - "Minor Radius": raggio minore (R1)
        """
        # Estrai i parametri
        angle_deg = basic_results["Confusion Angle (degrees)"]
        angle_rad = math.radians(angle_deg)
        R0 = basic_results["Major Radius"]
        R1 = basic_results["Minor Radius"]

        # Parametrizzazione dell'ellisse (non ruotata)
        t = np.linspace(0, 2*np.pi, 300)
        x = R0 * np.cos(t)
        y = R1 * np.sin(t)

        # Applica la rotazione per allineare l'ellisse con l'angolo di confusione
        X = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        Y = x * np.sin(angle_rad) + y * np.cos(angle_rad)

        # Converti le coordinate in polari
        theta = np.arctan2(Y, X)
        # Mantieni theta in [0, 2π)
        theta = np.mod(theta, 2*np.pi)
        r = np.sqrt(X**2 + Y**2)

        # (Opzionale) Ordina i punti per evitare salti nel tracciamento
        sort_idx = np.argsort(theta)
        theta = theta[sort_idx]
        r = r[sort_idx]
        
        # Se 'ax' è None, crea una nuova figura
        if ax is None:
            # Crea il plot polare
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})
            
        else:
            fig = ax.figure  #  Assicuriamoci che 'fig' sia sempre definito

        ax.plot(theta, r, color='purple', linewidth=2, label="Ellisse degli errori")

        # Disegna frecce per i raggi maggiori e minori
        # L'asse MINORE (R1) indica la direzione dell'angolo di confusione
        ax.annotate("", xy=(angle_rad, R1), xytext=(angle_rad, 0),
                    arrowprops=dict(color='green', arrowstyle="->", lw=2))

        # L'asse MAGGIORE (R0) è ruotato di +90° rispetto all'angolo di confusione
        ax.annotate("", xy=(angle_rad + np.pi/2, R0), xytext=(angle_rad + np.pi/2, 0),
                    arrowprops=dict(color='red', arrowstyle="->", lw=2))


        # Creiamo gli elementi della legenda
        legend_elements = [
            Line2D([0], [0], color='purple', lw=2, label="Ellisse degli errori"),
            Line2D([0], [0], color='red', lw=2, label="Major Axis (R0)"),
            Line2D([0], [0], color='green', lw=2, label="Minor Axis (R1)")
        ]

        # Aggiungiamo la legenda
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1.1))

        # Divisione in 10 settori di colore con etichette
        sectors = ["R", "RP", "P", "PB", "B", "BG", "G", "GY", "Y", "YR"]
        sector_colors = ["red", "orangered", "purple", "blueviolet", "blue", "cyan",
                        "green", "yellowgreen", "yellow", "orange"]
        n_sectors = len(sectors)
        sector_angles = np.linspace(0, 2*np.pi, n_sectors+1)
        
        # Scegli un raggio massimo per avere spazio
        rmax = max(R0, R1) * 1.4
        
        # Riempie ciascun settore con il colore scelto (con trasparenza alpha)
        for i in range(n_sectors):
            width = sector_angles[i+1] - sector_angles[i]
            ax.bar(x=sector_angles[i],
                height=rmax,
                width=width,
                bottom=0.0,
                color=sector_colors[i],
                alpha=0.2,
                edgecolor='none')

        for i in range(n_sectors):
            ax.axvline(sector_angles[i], color='gray', linestyle='--', linewidth=0.8)
            label_angle = (sector_angles[i] + sector_angles[i+1]) / 2
            ax.text(label_angle, rmax + 0.2, sectors[i],
                    ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Imposta il grado 0 a EST (destra)
        ax.set_theta_zero_location('E')

        # Mantieni il senso antiorario
        ax.set_theta_direction(1)  # 1 = Antiorario, -1 = Orario


        ax.set_title("Grafico Polare degli Errori\n(FM100 Hue Test)", va='bottom')

        # Imposta i limiti radiali e l'aspect ratio per non deformare l'ellisse
        ax.set_rmax(rmax)
        ax.set_ylim(0, rmax)
        ax.set_aspect('equal', 'box')

        # Testo interpretativo
        interpret_title = ("Interpretazione:")
        interpret_text = (  "L'angolo di confusione\n"
                            "e i raggi (Major e Minor) \n"
                            "determinano tipo e la gravità\n"
                            "del deficit cromatico\n\n"
                            "TES elevato:\nmaggiore dispersione\n" 
                            "e possibile discromatopsia")

        fig.text(0.05, 0.95, interpret_title,  ha='left', va='center', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8), fontweight='bold')
        fig.text(0.05, 0.85, interpret_text, ha='left', va='center', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))
        # Posiziona la finestra a sinistra (x = 50, y = 100)
        fig.canvas.manager.window.move(100, 50)
        
        # Memorizza la finestra per chiusura automatica
        self.graph_windows.append(fig)
        
         # Se 'ax' è stato creato qui, mostra la figura
        if ax is  None:
            plt.show()




    ###########################################################################
    # Funzione plot_fm100_hue_error_distribution: distribuzione degli errori
    ###########################################################################
    def plot_farnsworth_error_distribution(self, ax=None, mode="polar"):
        """
        Plotta la distribuzione degli errori nel test FM100 Hue come da rappresentazione Farnsworth.
        
        Parametri:
        - ax: asse su cui disegnare il grafico
        - mode: "polar" per la versione polare, "linear" per la versione cartesiana

        - ideal_indices_flat: lista dei numeri di cap nell'ordine ideale
        - user_indices_flat: lista dei numeri di cap nell'ordine ottenuto dall'utente
        """
        ideal_indices_flat = {int(i): int(i) for i in range(1, 85)}
        ideal_indices_flat[85] = 0    
        
        user_indices_flat = {}
        len_flat_rows = 0
        for i, row in enumerate(self.labels):
            for j, label in enumerate(row):
                user_indices_flat[label.num_caps] = j + len_flat_rows
            len_flat_rows += len(row)
        
        # Lista flat delle posizioni corrette delle capsule
        ideal_order = [85] + [i for i in range(1, 85)]
        
        
        # Errore per ogni cap: differenza tra posizione utente e ideale
        errors = [user_indices_flat[cap] - ideal_indices_flat[cap] for cap in ideal_order]

        # Imposta il raggio di base e il fattore di scala per l'errore
        r_base = 50
        scale = 3
        radii = [r_base + err * scale for err in errors]

        n_caps = len(ideal_order)
        
        # Calcola gli angoli in modo uniforme intorno al cerchio (partendo da pi/2)
        angles = [(math.pi/2 - (2*math.pi/n_caps)*i) % (2*math.pi) for i in range(n_caps)]
        
        # Chiudi la curva
        angles.append(angles[0])
        radii.append(radii[0])
        
        
        if mode == "polar":
            # Se 'ax' è None, crea una nuova figura
            if ax is None:
                fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            else:
                fig = ax.figure  #  Assicuriamoci che 'fig' sia sempre d
            ax.set_title("Distribuzione Errori - Polare")
        else:
            # Se 'ax' è None, crea una nuova figura
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure  #  Assicuriamoci che 'fig' sia sempre d
            ax.set_title("Distribuzione Errori - Lineare")


        ax.plot(angles, radii, marker='o', color='teal', linewidth=2, label="Distribuzione degli errori")
        
        # Aggiungi le etichette per ciascuna capsula
        for i, cap in enumerate(ideal_order):
            ax.text(angles[i], radii[i] + 2, str(cap),
                    ha='center', va='bottom', fontsize=8,
                    rotation=(-math.degrees(angles[i])+90) % 360)
        
        ax.set_title("Distribuzione Polare degli Errori nel FM100 Hue Test\n", va='bottom')
        ax.legend(loc='upper right')
        
        if mode == "polar":
            # Posiziona la finestra a sinistra (x = 1050, y = 100)
            fig.canvas.manager.window.move(1150, 0)
        else:
            # Posiziona la finestra a sinistra (x = 1050, y = 100)
            fig.canvas.manager.window.move(1150, 550)
        
        # Memorizza la finestra per chiusura automatica
        self.graph_windows.append(fig)
        # Se 'ax' è stato creato qui, mostra la figura
        if ax is None:
            plt.show()



    
    ###########################################################################
    # Funzione plot_error_per_vassoio: distribuzione degli errori nel vassoio
    ###########################################################################
    def plot_error_per_vassoio(self, TES_per_vassoio):
        """
        Visualizza un grafico a barre con il TES per ogni vassoio.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        media_TES = np.mean(TES_per_vassoio)
        deviazione_TES = np.std(TES_per_vassoio)
        vassoi_problematici = [i+1 for i, tes in enumerate(TES_per_vassoio) if tes > (media_TES + deviazione_TES)]
        
        colors = ['red' if i+1 in vassoi_problematici else 'steelblue' for i in range(len(TES_per_vassoio))]
        
        ax.bar(range(1, len(TES_per_vassoio) + 1), TES_per_vassoio, color=colors)
        ax.set_xlabel("Numero del Vassoio")
        ax.set_ylabel("TES (Errore Totale)")
        ax.set_title("Distribuzione degli Errori per Vassoio")
        ax.set_xticks(range(1, len(TES_per_vassoio) + 1))
        plt.show()


    
    #####################################################################
    # Funzione close_all_windows: Funzione per chiudere tutte le finestre
    #####################################################################
    def close_all_windows(self):
        """
        Chiude tutte le finestre aperte, inclusa la finestra dei risultati e i grafici.
        """
        try:
            if self.result_dialog and self.result_dialog.isVisible():
                self.result_dialog.close()
                self.result_dialog = None
        except RuntimeError:
            self.result_dialog = None

        self.close_graph_windows()

    def close_graph_windows(self):
        """
        Chiude tutte le finestre dei grafici.
        """
        for fig in getattr(self, 'graph_windows', []):
            try:
                plt.close(fig)
            except Exception:
                pass
        self.graph_windows = []
                    
        
    ###########################################################################################################
    # Funzione closeEvent: Chiusura automatica di tutte le finestre quando la finestra principale viene chiusa
    ###########################################################################################################
    def closeEvent(self, event):
        """
        Chiusura automatica di tutte le finestre quando la finestra principale viene chiusa.
        """
        self.close_all_windows()  # Chiude grafici e dialoghi aperti
        event.accept()  # Permette la chiusura dell'applicazione




    def get_classification_color(self, classification):
        """
        Ritorna un colore appropriato per la classificazione del test 
        (rappresentazione grafica a schermo non viene usata per i calcoli che vengono
        effettuati nello spazio CIELUV)
        """
        color_map = {
            "Normale": "#2ECC71",  # Verde
            "Protanopia Lieve": "#E74C3C",  # Rosso chiaro
            "Protanopia Significativa": "#C0392B",  # Rosso scuro
            "Deuteranopia Lieve": "#F39C12",  # Arancione chiaro
            "Deuteranopia Significativa": "#D35400",  # Arancione scuro
            "Tritanopia Lieve": "#3498DB",  # Blu chiaro
            "Tritanopia Significativa": "#2980B9",  # Blu scuro
            "Discromatopsia (non specificata) Lieve": "#9B59B6",  # Viola chiaro
            "Discromatopsia (non specificata) Significativa": "#8E44AD",  # Viola scuro
        }
        
        key_colors = [k for k in color_map if classification.startswith(k)]

        # Ritorna il colore se presente, altrimenti nero
        return color_map.get(max(key_colors, key=len), "#000000")



    #####################################################################
    # Funzione display_results: visualizza i risultati in tabella
    #####################################################################
    def display_results(self, basic_results):
            
        # Prova a chiudere la finestra dei risultati precedente, se ancora valida
        try:
            if self.result_dialog and self.result_dialog.isVisible():
                self.result_dialog.close()
        except RuntimeError:
            self.result_dialog = None  # L'oggetto è già stato eliminato, resetta

        self.result_dialog = QDialog(self)
        self.result_dialog.setWindowTitle("Risultati del Test FM100 Hue")
        self.result_dialog.setMinimumWidth(900)
        self.result_dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        self.result_dialog.destroyed.connect(self.close_graph_windows)  # Chiude i grafici alla chiusura del dialog

        layout = QVBoxLayout()

        # Estrai i dati
        confusion_angle = basic_results.get('Confusion Angle (degrees)', 'Dato mancante')
        c_index = basic_results.get('C-index', 'Dato mancante')
        s_index = basic_results.get('S-index', 'Dato mancante')
        major = basic_results.get('Major Radius', 'Dato mancante')
        minor = basic_results.get('Minor Radius', 'Dato mancante')
        tes_vingrys = basic_results.get('TES (Vingrys)', 'Dato mancante')
        tes_pca = basic_results.get('TES (PCA)', 'Dato mancante')
        eigenvalues_list = np.array(basic_results.get('Eigenvalues', [])).tolist()
        eigenvectors_list = np.array(basic_results.get('Eigenvectors', [])).tolist()
        correction_matrix_list = np.array(basic_results.get('Matrix Correction', np.zeros((2,2)))).tolist()

        correction_matrix_str = "<br>".join(
            [", ".join(f"{val:.7f}" for val in row_vals) for row_vals in correction_matrix_list]
        )

        # Funzione per riga
        def add_row(target_layout, label_text, value_text, tooltip_text=None):
            row = QHBoxLayout()

            label = QLabel(label_text)
            label.setStyleSheet("font-family: Verdana; font-size: 16px; font-weight: 600;")
            row.addWidget(label)

            if tooltip_text:
                icon = QLabel()
                icon.setPixmap(QPixmap("icons/info_icon.png").scaled(16, 16, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                icon.setToolTip(tooltip_text)
                icon.setStyleSheet("margin-left: 4px; margin-right: 6px;")
                row.addWidget(icon)
            else:
                row.addSpacing(16)

            value = QLabel(f": {value_text}")
            value.setStyleSheet("font-family: Verdana; font-size: 16px;")
            row.addWidget(value)

            row.addStretch()
            target_layout.addLayout(row)

        # Titolo
        title = QLabel("<h2 style='font-family: Verdana;'>Riepilogo risultati test</h2><hr>")
        layout.addWidget(title)

        classification = basic_results.get('Classification', 'N/A')
        color = self.get_classification_color(classification)
        class_label = QLabel(f"<br><b style='font-family: Verdana; font-size: 17px;'>Classificazione: <span style=' color: {color};'>{classification}</span></b><br><hr>")
        layout.addWidget(class_label)


        
        if self.mode_expert:
            layout.addWidget(QLabel(f"<span style='font-size:17px; font-family: Verdana; font-weight: 600;'>Valutazione Test:  </span> <u style='font-size:18px; font-family: Verdana;' >{basic_results.get('TestV', 'Test-1')}</u><br><br>"))

            # Sezione Vingrys
            add_row(layout, "Confusion Angle (degrees)", f"{confusion_angle:.2f}°", 
            """Angolo medio tra le direzioni di errore delle capsule rispetto all'asse ideale. 
            Valori prossimi a 0 indicano una buona coerenza percettiva, 
            mentre valori elevati suggeriscono disallineamenti significativi nella percezione dei colori.""")

            add_row(layout, "C-index", f"{c_index:.2f}", 
            """Indice di coerenza della sequenza cromatica. 
            Misura quanto ordinatamente le capsule sono state disposte. 
            Valori più bassi indicano maggiore confusione nell'ordine scelto.""")

            add_row(layout, "S-index", f"{s_index:.2f}", 
            """Indice di dispersione. 
            Valuta la variabilità degli errori lungo la traiettoria cromatica. 
            Valori più elevati suggeriscono una percezione irregolare e meno consistente.""")

            add_row(layout, "Major Radius", f"{major:.2f}", 
            """Rappresenta la dimensione dell'asse principale dell'ellisse di errore. 
            Indica la direzione più dominante degli errori nella percezione cromatica.""")

            add_row(layout, "Minor Radius", f"{minor:.2f}", 
            """Rappresenta la dimensione dell'asse secondario dell'ellisse di errore. 
            Descrive la variabilità residua nella direzione meno influente.""")

            add_row(layout, "TES (Vingrys)", f"{tes_vingrys:.2f}", 
            """Total Error Score calcolato secondo il metodo di Vingrys & King. 
            Somma di tutti gli errori tra capsule adiacenti, 
            utile per identificare la gravità generale delle anomalie nella visione cromatica.""")

            add_row(layout, "TES Locale", str(basic_results.get('TES Local', [])), 
            """Serie di punteggi TES calcolati separatamente per ciascun vassoio. 
            Consentono di identificare errori localizzati, anche se il punteggio complessivo è nella norma.""")

            add_row(layout, "Vassoi Problematici", str(basic_results.get('Vassoi Problematici', "Nessun Vassoio problematico")), 
            """Elenco dei vassoi che presentano punteggi TES anomali, 
            cioè superiori alla soglia statistica (media + 2 deviazioni standard). 
            Utili per localizzare errori specifici.""")

            # PCA
            layout.addWidget(QLabel(f"<br><hr><br><span style='font-size:17px; font-family: Verdana; font-weight: 600;'>Valutazione Test:  </span> <u style='font-size:18px; font-family: Verdana;' >{basic_results.get('TestP', 'Test-1')}</u><br><br>"))
            add_row(layout, "TES (PCA)", f"{tes_pca:.2f}", 
            """Errore globale ottenuto tramite l'analisi delle componenti principali (PCA). 
            Rappresenta la deviazione complessiva dalla traiettoria cromatica ideale.""")

            add_row(layout, "Eigenvalues", ", ".join(f"{v:.2f}" for v in eigenvalues_list), 
            """Autovalori calcolati con la PCA. 
            Misurano la quantità di varianza presente lungo ogni direzione principale. 
            Valori elevati indicano componenti dominanti.""")

            add_row(layout, "Eigenvectors", ", ".join(str(v) for v in eigenvectors_list), 
            """Autovettori derivati dalla PCA. 
            Definiscono le direzioni principali in cui si manifestano le anomalie nella visione cromatica. 
            Ogni vettore rappresenta una modalità specifica di distorsione percettiva.""")

            # Matrice
            matrix_row = QHBoxLayout()

            title_label = QLabel("<span style='font-size:17px; font-family: Verdana; font-weight: 600;'>Matrice di correzione (PCA):</span>")
            icon = QLabel()
            icon.setPixmap(QPixmap("icons/info_icon.png").scaled(16, 16, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            icon.setToolTip("""Matrice 2x2 derivata dall'analisi PCA. 
            Viene utilizzata per rappresentare la trasformazione cromatica necessaria 
            a correggere la percezione dei colori distorta. 
            Ogni valore rappresenta il contributo di una componente principale 
            nel riposizionamento delle capsule lungo gli assi percettivi.""")
            icon.setStyleSheet("margin-left: 6px;")

            matrix_row.addWidget(title_label)
            matrix_row.addWidget(icon)
            matrix_row.addStretch()
            layout.addWidget(QLabel("<hr><br>"))
            layout.addLayout(matrix_row)

            # Contenuto della matrice
            matrix_content = QLabel(f"<span style='font-family: Verdana; font-size: 15px;'>{correction_matrix_str}</span>")
            layout.addWidget(matrix_content)
            layout.addWidget(QLabel("<br><hr><br>"))

        
        
        # Pulsanti
        
        button_layout = QHBoxLayout()
        
        plot_button = QPushButton("Mostra i Grafici")
        close_button = QPushButton("Chiudi Risultati")
        close_button.clicked.connect(self.close_all_windows)
        close_button.hide()
        
        def toggle_graphs():
            if plot_button.text() == "Mostra i Grafici":
                self.show_plots(basic_results)
                plot_button.setText("Nascondi i Grafici")
                close_button.show()
            else:
                self.close_graph_windows()
                plot_button.setText("Mostra i Grafici")
                close_button.hide()
        
        plot_button.clicked.connect(toggle_graphs)
        
        button_layout.addWidget(plot_button)    
        
        # se siamo in una simulazione conservo comunque i dati nel csv con id_user = -1 a prescindere quindi non ho bisogno di salvarli manualmente,
        # quindi visualizziamo il bottone solo se siamo in test mode
        if self.mode_combo.currentText() == "Test Mode":
            save_button = QPushButton("Salva/Aggiorna Dati")
            save_button.clicked.connect(lambda: self.save_results_profile(basic_results))
            button_layout.addWidget(save_button)
            

        button_layout.addWidget(close_button)
        
        
        layout.addLayout(button_layout)
        
        layout.addWidget(QLabel("<br><br>"))
        
        self.result_dialog.setLayout(layout)
        self.result_dialog.show()



    def save_results_profile(self, basic_results):
        """
        Funzione richiamata dal bottone nella finestra dei risultati.
        Apre una finestra di dialogo con due opzioni:
        - "Nuovo utente": apre una finestra per la creazione del utente utente e, se completata, recupera il valore in self.profile.
        - "utente esistente": carica il utente già salvato.
        In entrambi i casi, il dizionario del utente viene passato alla funzione save_json_dataset.
        """
        # Crea la finestra di dialogo per la scelta del utente
        choice_dialog = QDialog(self)
        choice_dialog.setWindowTitle("Seleziona Opzione Utente")
        choice_dialog.setFixedSize(480, 280)
        
        layout = QVBoxLayout(choice_dialog)

        
        btn_new = QPushButton("Nuovo Utente", choice_dialog)
        btn_existing = QPushButton("Trova Utente", choice_dialog)
        
        layout.addWidget(btn_new)
        layout.addWidget(btn_existing)
        choice_dialog.setLayout(layout)
        
        
        # Definizione della funzione per il bottone "Nuovo utente"
        def on_new_profile():
            choice_dialog.accept()  # chiude il dialogo di scelta
            # Crea e mostra la finestra per creare un nuovo utente
            new_profile_dialog = UserProfileDialog(self, "new")
            
            if new_profile_dialog.exec_() == QDialog.Accepted:
                # Recupera l'utente creato
                new_profile = new_profile_dialog.profile
                if new_profile is not None and not new_profile_dialog.cancel_save:
                    try:
                        # Passa l'utente alla funzione di salvataggio dei json
                        self.save_json_dataset(basic_results, id_user=new_profile["id_user"])
                        # Salviamo i risultati ottenuti in un dataset in csv per future elaborazioni
                        self.save_csv_dataset(basic_results, id_user=new_profile["id_user"])
                        
                        if(self.mode_expert):
                            if(new_profile_dialog.profile_updated == False):     # profile_updated = TRUE se aggiornato, profile_updated = FALSE se creato
                                QMessageBox.information(self, "Creazione Utente e Salvataggio con Successo", "Utente creato e Dati salvati con Successo!")
                            else:
                                QMessageBox.information(self, "Utente Aggiornato e Salvataggio con Successo", "Utente Aggiornato e Dati Salvati e Aggiunti con Successo!")
                        else:
                            if(new_profile_dialog.profile_updated == False):     # profile_updated = TRUE se aggiornato, profile_updated = FALSE se creato
                                QMessageBox.information(self, "Creazione Utente e Salvataggio con Successo", "Utente creato e Dati salvati con Successo!")
                            else:
                                QMessageBox.information(self, "Utente Non Aggiornato e Salvataggio con Successo", "Utente Non Aggiornato (Non detieni i permessi). Dati Salvati e Aggiunti con Successo!")
                                
                    except RuntimeError:
                        QMessageBox.warning(self, "Creazione Utente", "Utente creato con Successo! Dati Non Salvati!")

                else:
                    QMessageBox.warning(self, "Creazione Utente", "Nessun Utente Creato! Dati Non Salvati!")
            else:
                QMessageBox.warning(self, "Creazione Utente", "Creazione Utente e Salvataggio Annullati!")
                
                
        # Definizione della funzione per il bottone "Trova utente"
        def on_existing_profile():
            choice_dialog.accept()  # chiude il dialogo di scelta
            # Crea e mostra la finestra per trovare un utente
            exist_profile_dialog = UserProfileDialog(self, "exist")
            
            if exist_profile_dialog.exec_() == QDialog.Accepted:
                # Recupera l'utente trovato
                exist_profile = exist_profile_dialog.profile
                
                if exist_profile is not None:
                    try:
                        # Salva i dati basic_result nel JSON separato al utente, e gli passa l'id_user
                        self.save_json_dataset(basic_results, id_user=exist_profile["id_user"])
                        # Salviamo i risultati ottenuti in un dataset in csv per future elaborazioni
                        self.save_csv_dataset(basic_results, id_user=exist_profile["id_user"])
                        
                        QMessageBox.information(self, "Ricerca Utente e Salvataggio", "Utente trovato e Dati Aggiornati con Successo!")
                
                    except RuntimeError:
                        QMessageBox.warning(self, "Ricerca Utente", "Utente trovato con Successo! Dati Non Aggiornati!")

                else:
                    QMessageBox.warning(self, "Ricerca Utente", "Nessun Utente Trovato! Dati Non Aggiornati!")
            else:
                QMessageBox.warning(self, "Ricerca Utente", "Ricerca Utente e Aggiornamento Dati Annullati!")

        
        btn_new.clicked.connect(on_new_profile)
        btn_existing.clicked.connect(on_existing_profile)
        
        choice_dialog.exec_()

    
    def show_plots(self, basic_results):
        """
        Mostra entrambi i grafici: uno polare e uno lineare per la distribuzione degli errori.
        Posiziona le finestre in modo che non si sovrappongano.
        """
        # Crea un nuovo grafico per l'ellisse degli errori
        fig_ellipse, ax_ellipse = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})
        fig_ellipse.canvas.manager.window.move(100, 50)  # Sposta la finestra a sinistra
        self.plot_error_ellipse(basic_results, ax=ax_ellipse)  # Passa l'asse corretto
        self.graph_windows.append(fig_ellipse)

        # Crea un nuovo grafico per la distribuzione degli errori in formato POLARE
        fig_farnsworth_polar, ax_farnsworth_polar = plt.subplots(figsize=(7, 5), subplot_kw={'projection': 'polar'})
        fig_farnsworth_polar.canvas.manager.window.move(1150, 0)  # Sposta la finestra a destra
        self.plot_farnsworth_error_distribution(ax=ax_farnsworth_polar)  #  Passa l'asse corretto
        self.graph_windows.append(fig_farnsworth_polar)

        # Crea un nuovo grafico per la distribuzione degli errori in formato LINEARE
        fig_farnsworth_linear, ax_farnsworth_linear = plt.subplots(figsize=(7, 4))  # Nessuna proiezione polare
        fig_farnsworth_linear.canvas.manager.window.move(1150, 550)  # Sposta la finestra più in basso
        self.plot_farnsworth_error_distribution(ax=ax_farnsworth_linear, mode="linear")  # Passa il parametro
        self.graph_windows.append(fig_farnsworth_linear)
        
        # se siamo in una simulazione conservo comunque i dati con id_user = -1
        if self.mode_combo.currentText() != "Test Mode":
            self.save_csv_dataset(basic_results, id_user="simulazione")

        # Mostra le figure
        plt.show()


    #####################################################################
    # Funzione classify_defect: classifica il difetto cromatico
    #####################################################################
    def classify_defect(self, confusion_angle, tes_vingrys, c_index, s_index, vassoi_problematici):
        """
        Classifica il tipo e la gravità del difetto cromatico.
        Usa:
        - L'angolo di confusione per determinare il tipo di discromatopsia
        - Il TES per determinare la gravità
        - Il C-index per verificare se il soggetto è normale
        - L'analisi dei vassoi problematici per distinguere tra errori localizzati e diffusi.
        """
    
        # 1) Normalizzazione dell'angolo
        confusion_angle = (confusion_angle + 180) % 360 - 180  # Normalizza

        # 2) Soglia di “normalità” per l'errore
        # Definiamo la soglia di normalità (letteratura: TES < ~4.5 indica normalità)
        if tes_vingrys < 4.5 and c_index < 1.2:
            return f"Normale [{'S-Index alto - Possibile outlier' if s_index > 1.5 else 'S-Index basso - Misura corretta'}]"

        # 3) Gravità in base al TES
        severity = "Lieve" if tes_vingrys <= 10 else "Significativa"
        
        # 4) Distinzione tra errore localizzato e diffuso
        if len(vassoi_problematici) > 2:
            localizzazione = "Errore Diffuso"
        else:
            localizzazione = "Errore Localizzato"
        
        # 5) Classificazione dell'asse
                
        # N.B. Utilizziamo l's-index per indicare possibili outlier.
        # NON avendo prove in letteratura che permettano di definire l'uso di questo parametro come valore di soglia,
        # possiamo usarlo per inserire nei risultati un messaggio di warning.
        # Se il rapporto è troppo alto, potrebbe indicare un problema nei dati.
        
        # N.B.B  Secondo il valore di localizzazione abbiamo che:
        # - Se l'errore è diffuso -> indica una discromatopsia più grave e generalizzata
        # - Se l'errore è localizzato -> indica un disturbo lieve o atipico
        
        
        if -15 <= confusion_angle <= 15:
            return f"Protanopia {severity}  -  [{localizzazione}]  -  [{'S-Index alto - Possibile outlier' if s_index > 1.5 else 'S-Index basso - Misura corretta'}]"
        elif -60 <= confusion_angle <= -30:
            return f"Deuteranopia {severity}  -  [{localizzazione}]  -  [{'S-Index alto - Possibile outlier' if s_index > 2.5 else 'S-Index basso - Misura corretta'}]"
        elif 75 <= confusion_angle <= 105 or -105 <= confusion_angle <= -75:
            return f"Tritanopia {severity}  -  [{localizzazione}]  -  [{'S-Index alto - Possibile outlier' if s_index > 2.5 else 'S-Index basso - Misura corretta'}]"
        else:
            return f"Discromatopsia (non specificata) {severity}  -  [{localizzazione}]  -  [{'S-Index alto - Possibile outlier' if s_index > 2.5 else 'S-Index basso - Misura corretta'}]"




    #####################################################################
    # Funzione generate_correction_matrix (invariata)
    #####################################################################
    # def generate_correction_matrix(self, confusion_angle, c_index, s_index):
    #     theta = math.radians(confusion_angle)
    #     rotation_matrix = np.array([
    #         [math.cos(theta), -math.sin(theta)],
    #         [math.sin(theta), math.cos(theta)]
    #     ])
    #     scaling_matrix = np.diag([1/c_index, 1/s_index])
    #     correction_matrix = rotation_matrix @ scaling_matrix @ np.linalg.inv(rotation_matrix)
    #     return correction_matrix


    #####################################################################
    # Funzione apply_correction_to_image (invariata)
    #####################################################################
    # def apply_correction_to_image(self, image, correction_matrix):
    #     ab_channels = image[:, :, 1:3].reshape(-1, 2)
    #     corrected_ab = ab_channels @ correction_matrix.T
    #     corrected_image = image.copy()
    #     corrected_image[:, :, 1:3] = corrected_ab.reshape(image[:, :, 1:3].shape)
    #     return corrected_image


    #########################################################################################
    # Funzione compute_pca_correction_matrix : Calcola la matrice di whitening (correzione)
    #########################################################################################
    def compute_pca_correction_matrix(self, vingrys_matrix, error_localizzato):
        """
        Dato il tensore degli errori (matrice 2x2) derivato dai dati Vingrys,
        calcola la matrice di whitening (correzione) per ottenere una distribuzione isotropica.
        Questa matrice può essere usata per correggere le distorsioni cromatiche nelle immagini.

        Modifica la correzione cromatica in base al tipo di errore
        """
        eigenvalues, eigenvectors = np.linalg.eig(vingrys_matrix)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        matrix_correction = np.eye(2)
        
        # if "Errore Diffuso"
        if len(error_localizzato) > 2:
            #  Correggi selettivamente solo i vassoi più problematici
            eigenvalues[0] *= 1.2  
            eigenvalues[1] *= 0.8  
        else:   # else "Errore Localizzato"
            #  Correzione uniforme per errore diffuso
            eigenvalues *= 1.1  

        #  Evitiamo instabilità numeriche
        D_inv_sqrt = np.diag(1 / np.sqrt(np.maximum(eigenvalues, 1e-12)))
        matrix_correction = eigenvectors @ D_inv_sqrt @ eigenvectors.T
        
        return matrix_correction




    #####################################################################
    # Submit: Esegue il calcolo e mostra i risultati
    #####################################################################

    def submit(self):
        if not self.get_mode_test():    # FALSE = simple mode
            # Avvia la finestra modale con i vassoi uno alla volta
            self.hide()
            self.lock_labels(lock=False)    #  SBLOCCA PER IL DIALOG SEQUENZIALE
            self.dialog = FarnsworthSequentialDialog(self)

            def _on_sequence_end():
                basic_results = self.calculate_error_basic_PCA()
                if basic_results is not None:
                    self.display_results(basic_results)
                self.dialog.close()
                
            self.dialog.finished.connect(_on_sequence_end)
            self.dialog.exec_()

        else:
            basic_results = self.calculate_error_basic_PCA()
            if basic_results is None:
                return
            self.display_results(basic_results)



      
    ################################################################################
    # Funzione save_csv_dataset :  CREAZIONE/APPEND dei risultati del test al CSV 
    ################################################################################

    def save_csv_dataset(self, basic_results, id_user=None):
        if id_user is None:
            QMessageBox.warning(self, "Errore", "ID utente non specificato")
            return

        if id_user == "simulation":
            id_user = -1

        current_directory = os.path.dirname(os.path.abspath(__file__))
        dataset_directory = os.path.join(current_directory, 'dataset_test')
        os.makedirs(dataset_directory, exist_ok=True)
        
        rows_csv = []
        
        csv_path = os.path.join(dataset_directory, 'csv_test_results.csv')

        # Header CSV atteso
        csv_headers = [
            "id_user", "timestamp", "Confusion Angle (degrees)", "C-index", "S-index", "Major Radius",
            "Minor Radius", "TES (Vingrys)", "TES (PCA)", "Eigenvalues", "Eigenvectors", "Matrix Correction",
            "Classificazione non normalizzata", "Normale", "Protanopia Lieve", "Protanopia Significativa",
            "Deuteranopia Lieve", "Deuteranopia Significativa", "Tritanopia Lieve", "Tritanopia Significativa",
            "Discromatopsia (non specificata) Lieve", "Discromatopsia (non specificata) Significativa"
        ]

        new_row = [
            id_user,
            basic_results["timestamp"],
            basic_results["Confusion Angle (degrees)"],
            basic_results["C-index"],
            basic_results["S-index"],
            basic_results["Major Radius"],
            basic_results["Minor Radius"],
            basic_results["TES (Vingrys)"],
            basic_results["TES (PCA)"],
            basic_results["Eigenvalues"],
            basic_results["Eigenvectors"],
            basic_results["Matrix Correction"],
            basic_results["Classification"]
        ] + self.one_hot_encode_classification(basic_results["Classification"])

        file_exists = os.path.exists(csv_path)
        header_corretto = False

        if file_exists:
            with open(csv_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                header_corretto = first_line == ",".join(csv_headers)
                
                if(not header_corretto): # FALSE se l'header non è presente o non è corretto
                    f.seek(0)  # torna all'inizio del file
                    rows_csv = list(csv.reader(f))
                    
        mode = "a" if file_exists and header_corretto else "w"
        
        with open(csv_path, mode , newline="", encoding="utf-8") as f:
            
            writer = csv.writer(f)
            
            if mode == "w":
                writer.writerow(csv_headers)
                
                if not header_corretto: # FALSE se l'header non è presente o non è corretto -> mode = w
                    writer.writerows(rows_csv)
                    
            writer.writerow(new_row)

        print(f"Risultati salvati correttamente in {csv_path}")

    

    ################################################################################
    # Funzione save_json_dataset :  CREAZIONE/APPEND dei risultati del test al JSON 
    ################################################################################
    def save_json_dataset(self, basic_results, id_user=None):
        
        if id_user is None:
            QMessageBox.warning(self, "Errore", "ID utente non specificato")
            return
        
        # Definisce il percorso del file JSON dei risultati
        current_directory = os.path.dirname(os.path.abspath(__file__))
        dataset_directory = os.path.join(current_directory, 'dataset_test')  # Cartella del dataset
        json_test_results_global = os.path.join(dataset_directory, 'json_test_results_global.json')  # File JSON GLOBALE
        json_test_results_profile = os.path.join(dataset_directory, id_user+'.json')  # File JSON GLOBALE
        
        #  Crea la cartella "dataset_test" se non esiste
        os.makedirs(dataset_directory, exist_ok=True)

        # Verifica se il file json_test_results_profile esiste già
        if os.path.exists(json_test_results_profile):
            # Se il file esiste, carica i dati esistenti
            with open(json_test_results_profile, "r", encoding="utf-8") as file:
                try:
                    data = json.load(file)  # Carica i dati esistenti
                except json.JSONDecodeError:
                    data = []  # Se il file è vuoto o corrotto, crea una lista vuota
        else:
            data = []  # Se il file non esiste, inizializza una lista vuota
            
        # Verifica se il file json_test_results_global esiste già
        if os.path.exists(json_test_results_global):
            # Se il file esiste, carica i dati esistenti
            with open(json_test_results_global, "r", encoding="utf-8") as file:
                try:
                    data_global = json.load(file)  # Carica i dati esistenti
                except json.JSONDecodeError:
                    data_global = []  # Se il file è vuoto o corrotto, crea una lista vuota
        else:
            data_global = []  # Se il file non esiste, inizializza una lista vuota
            
            

        # Prepara i dati per il nuovo record
        new_entry = {
            "id_user": id_user,
            "timestamp": basic_results["timestamp"],
            "Confusion Angle (degrees)": basic_results["Confusion Angle (degrees)"],
            "C-index": basic_results["C-index"],
            "S-index": basic_results["S-index"],
            "Major Radius": basic_results["Major Radius"],
            "Minor Radius": basic_results["Minor Radius"],
            "TES (Vingrys)": basic_results["TES (Vingrys)"],
            "TES (PCA)": basic_results["TES (PCA)"],
            "Eigenvalues": np.array(basic_results["Eigenvalues"]).tolist(),  # Converte NumPy array in lista
            "Eigenvectors": np.array(basic_results["Eigenvectors"]).tolist(),  # Converte NumPy array in lista
            "Matrix Correction": np.array(basic_results["Matrix Correction"]).tolist(),  # Converte NumPy array in lista
            "Classificazione non normalizzata": basic_results["Classification"],
            "Classification One-Hot": self.one_hot_encode_classification(basic_results["Classification"])  # One-Hot Encoding
        }

        # Aggiunge il nuovo risultato alla lista esistente
        data.append(new_entry)
        data_global.append(new_entry)

        # Scrive i dati aggiornati nel file JSON
        with open(json_test_results_profile, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
            
        with open(json_test_results_global, "w", encoding="utf-8") as file:
            json.dump(data_global, file, indent=4)

        print(f"Risultati salvati correttamente in {json_test_results_profile} e in {json_test_results_global}")

        
    ######################################################################################################
    # Funzione one_hot_encode_classification :  Serve per la normalizzazione della colonna Classification
    ######################################################################################################
    def one_hot_encode_classification(self, classification):
        '''
            Per poter addestrare una rete neurale o un modello avanzato, utilizziamo la One-Hot Encoding, trasformando ogni classe in un array binario.
            In questo modo possiamo normalizzare la colonna Classification.
        '''
        classes = [
            "Normale", "Protanopia Lieve", "Protanopia Significativa",
            "Deuteranopia Lieve", "Deuteranopia Significativa",
            "Tritanopia Lieve", "Tritanopia Significativa",
            "Discromatopsia (non specificata) Lieve", "Discromatopsia (non specificata) Significativa"
        ]
        vector = [1 if classification == c else 0 for c in classes]
        return vector
    


class UserProfileDialog(QDialog):
    def __init__(self, parent=None, type="new"):
        super(UserProfileDialog, self).__init__(parent)
        self.setWindowTitle("Nuovo Utente")
        self.profile = None  # Qui verrà memorizzato l'utente creato
        self.init_ui(type)
        self.parent = parent
        self.profile_updated = False
        self.cancel_save = False

    def init_ui(self, type):
        # Usa un QFormLayout per organizzare le etichette e i campi di input
        layout = QFormLayout(self)

        # Creazione delle caselle di testo per Codice Fiscale, Nome, Cognome, Età e Sesso
        
        self.CF_edit = QLineEdit(self)
        layout.addRow("Codice Fiscale:", self.CF_edit)
        
        if(type!="exist"):
            self.nome_edit = QLineEdit(self)
            self.cognome_edit = QLineEdit(self)
            self.eta_edit = QLineEdit(self)
            
            self.sesso_edit = QComboBox()
            self.sesso_edit.addItems(["Maschio", "Femmina", "Altro", "Preferisco non dirlo"])
            
            self.sesso_edit.setCurrentIndex(0)  # ad esempio: Maschio
            
            layout.addRow("Nome:", self.nome_edit)
            layout.addRow("Cognome:", self.cognome_edit)
            layout.addRow("Età:", self.eta_edit)
            layout.addRow("Sesso:", self.sesso_edit)


        # Bottone per salvare l'utente
        if(type=="new"):
            self.save_button = QPushButton("Salva Nuovo Utente", self)
            self.save_button.clicked.connect(self.on_save)
        elif(type=="exist"):
            self.save_button = QPushButton("Trova Utente", self)
            self.save_button.clicked.connect(lambda:on_load(self, name = self.CF_edit.text().strip()))
            
        layout.addRow(self.save_button)

        self.setLayout(layout)
        
        
        
    def on_save(self):
        # Recupera i dati dalle caselle di testo
        CF = self.CF_edit.text().strip()
        nome = self.nome_edit.text().strip()
        cognome = self.cognome_edit.text().strip()
        eta = self.eta_edit.text().strip()
        sesso = self.sesso_edit.currentText()


        # Validazione dei dati: tutti i campi devono essere compilati e l'età deve essere numerica
        if not CF or not nome or not cognome or not eta or not sesso:
            QMessageBox.warning(self, "Errore", "Tutti i campi sono obbligatori.")
            return

        try:
            eta_int = int(eta)
        except ValueError:
            QMessageBox.warning(self, "Errore", "L'età deve essere un numero.")
            return
        
        unique_id = generate_unique_id(CF)
        print("ID Unico generato:", unique_id)
    

        # Creazione del utente come dizionario
        profile = {
            "id_user": unique_id,
            "codice_fiscale": CF,
            "nome": nome,
            "cognome": cognome,
            "eta": eta_int,
            "sesso": sesso
        }

        # Conversione del utente in JSON per il debug
        profile_json = json.dumps(profile, indent=4)
        print("Utente creato:", profile_json)

        # Salva l'utente in un attributo per il recupero esterno dopo la chiusura della finestra
        self.profile = profile


        # Definisce il percorso del file JSON dei profili e salva il nuovo utente
        current_directory = os.path.dirname(os.path.abspath(__file__))
        data_directory = os.path.join(current_directory, 'profiles_data')  # Cartella dei profili utenti
        json_test_results = os.path.join(data_directory, unique_id+'.json')  # File JSON con nome id_univoco
        
        #  Crea la cartella "profiles_data" se non esiste
        os.makedirs(data_directory, exist_ok=True)
        

        # Verifica se il file esiste già
        if os.path.exists(json_test_results):
            # Se il file esiste, carica i dati esistenti
            with open(json_test_results, "r", encoding="utf-8") as file:
                try:
                    if self.parent.mode_expert:
                        reply = QMessageBox.question(
                            self,
                            "Utente già esistente!",
                            "L'utente è già esistente. Vuoi aggiornare i suoi dati?",
                            QMessageBox.Yes | QMessageBox.Cancel
                        )


                        if reply == QMessageBox.Yes:
                            with open(json_test_results, "w", encoding="utf-8") as f:
                                json.dump(profile, f, indent=4)
                                
                            print("Aggiornarmento avvenuto con Successo!")
                            
                            QMessageBox.information(self, "Aggiornamento", "Aggiornarmento avvenuto con Successo!")
                            self.profile = profile  # memorizza il profilo aggiornato
                            self.profile_updated = True
                            self.accept()           # chiudi correttamente il dialogo
                            return
                        else:
                            QMessageBox.warning(self, "Error", "Aggiornamento Dati Annullato!")
                            print("Annullato")
                            self.cancel_save = True
                            self.accept()           # chiudi correttamente il dialogo
                            return
                    
                    else:   # Se l'utente è in modalità non esperto non può aggiornare i dati di un'altro utente
                        reply = QMessageBox.question(
                            self,
                            "Utente già esistente!",
                            "I dati inseriti appartengono ad un altro utente esistente. Non hai i permessi per aggiornarli. Vuoi procedere con il salvataggio dei risultati del test per questo profilo selezionato?",
                            QMessageBox.Yes | QMessageBox.Cancel
                        )

                        if reply == QMessageBox.Yes:
                            self.profile_updated = True
                            self.accept()           # chiudi correttamente il dialogo
                            return
                        else:
                            print("Annullato")
                            self.cancel_save = True
                            self.accept()           # chiudi correttamente il dialogo
                            return
                    
                except json.JSONDecodeError:
                    # Se il file è vuoto o corrotto da un errore
                    QMessageBox.warning(self, "Error", "Utente esistente - Dati corrotti!")
                    return
        else:
            # Salvataggio del utente in un file separato
            with open(json_test_results, "w", encoding="utf-8") as f:
                json.dump(profile, f, indent=4)
                
            print(f"Utente salvato in {json_test_results}")

        # Chiude la finestra di dialogo
        self.accept()
        
################# UTILITY ###################################################
        
def unique_identify_path(name, name_add=None, type_file=None, created_id=False, folder_name=None):
    
    if created_id==False:
        unique_id = generate_unique_id(name)
    else:   # in questo caso name è uguale all'id ma senza il resto -> "-profile_feats", ".npy"
        unique_id = name
    
    # Definisce il percorso del file del utente esistente; qui assumiamo di usare la cartella 'profiles_data', gli passiamo l'id_user
    if name_add is not None and isinstance(name_add, str): unique_id = unique_id + name_add
    if type_file is None or not isinstance(type_file, str): type_file = ".json"
    if folder_name is None or not isinstance(type_file, str): folder_name= 'profiles_data'
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), folder_name, unique_id + type_file), unique_id

        
# UTILITY CARICAMENTO
def on_load(parent_ref=None, gui=True, name=None, name_add=None, type_file=None, created_id=False, path=None):
    
    if name is None and path is None:
        if gui and parent_ref:
            QMessageBox.warning(parent_ref, "Errore", "Campo obbligatorio")
        return None
    
    if path is None:
        profile_path, _ = unique_identify_path(name, name_add, type_file, created_id)
    else:
        profile_path = path
    
    
    if gui and parent_ref:
        # recupera i dati dal JSON esistente del utente e li mette in un attributo per il recupero esterno dopo la chiusura della finestra
        parent_ref.profile = load_profile_json(profile_path)

        # Chiude la finestra di dialogo
        
        parent_ref.accept()
    else:
        return load_profile_json(profile_path)


# UTILITY CARICAMENTO JSON DA PATH
def load_profile_json(file_path):
    """
    Carica l'utente salvato da file JSON.
    Se il file non esiste o si verifica un errore, restituisce None.
    """
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                profile = json.load(f)
            return profile
        except Exception as e:
            print("Errore nel caricamento dell'Utente:", e)
            return None
    else:
        print("Nessun Utente esistente trovato.")
        return None
    


    ############## GENERATORE ID #########################################################

def generate_unique_id(str_convert):
    """
    Genera un id_univoco combinando:
      - 20 caratteri di hash (MD5) derivati da str.

    Returns:
        str: L'id_univoco composto da 20 caratteri.
    """
    # Calcola l'hash MD5 e prendi i primi 20 caratteri
    hash_digest = hashlib.md5(str_convert.encode('utf-8')).hexdigest()[:20]
    
    # Concatena l'hash e la stringa random per formare l'id univoco
    return hash_digest


################# END UTILITY ###################################################

    
    
#############################################
# Classe FarnsworthTestSimplified
#############################################
class FarnsworthTestWrapper:
    """
    CLASSE WRAPPER necessaria per il modulo get_profile_feats, e per simulazione di profili sintetici.
    La classe:
        1) Avvia il test FarnsworthTest (GUI)
        2) Alla chiusura, return 'Matrix Correction' + 'Confusion Angle' attraverso il metodo 'calculate_error_basic_PCA()'
    """
    def __init__(self):
        # Controlla se esiste già un'istanza QApplication
        existing_app = QApplication.instance()
        if existing_app is None:
            self.app = QApplication(sys.argv)
            self.own_app = True
        else:
            self.app = existing_app
            self.own_app = False
        
        self.test_gui = FarnsworthTest()

    def run_test_and_get_correction(self):
        # QUI POI METTEREMO LA LOGICA PER CREARE DATI SINTETICI E COMMENTIAMO LA RIGA DOPO
        self.test_gui.show()
        
        # Porta in primo piano la finestra del test
        self.test_gui.raise_()
        self.test_gui.activateWindow()
        
        ##################################################################################
        # Esegui solo se abbiamo creato noi l'app, altrimenti usa processEvents
        if self.own_app:
            self.app.exec_()
        else:
            # Usa un loop personalizzato per non interferire con tkinter
            while self.test_gui.isVisible():
                self.app.processEvents()
                import time
                time.sleep(0.01)  # Piccolo delay per non sovraccaricare la CPU
        
        # Al termine, richiamiamo la calculate_error_basic_PCA() della FarnsworthTest
        results = self.test_gui.calculate_error_basic_PCA()
        if results:
            # Restituisce il dizionario con tutti i valori ottenuti
            return results
        else:
            QMessageBox.warning(self, "Error", "Dati insufficienti per proseguire il test")
            sys.exit(1)
            

class FarnsworthSequentialDialog(QDialog):
    
    def __init__(self, parent_test):
        super().__init__()
        self.setWindowTitle("Test Farnsworth-Munsell - Vassoio 1")
        # Adatta dinamicamente alla lunghezza del vassoio più lungo

        self.parent_test = parent_test
        self.current_tray_index = 0
        self.trays = parent_test.shuffled_colors
        self.tray_data = parent_test.shuffled_colors

        self.visible_number = parent_test.visible_number
        self.setFixedSize(1600, 300)

        self.init_ui()


        
    def clone_labels(self, original_labels, row_index):
        '''
        QWidget non è copiabile in modo sicuro con deepcopy(), perchè gli oggetti QLabel,
        QPushButton, ecc. (tutti derivati da QWidget) sono strettamente legati all'ambiente
        grafico di Qt e al loro parent().
        
        original_labels è una lista di dict {num_caps, color, x, y}, non di QLabel.
        row_index serve per dire a DraggableLabel in quale riga stiamo.
        '''
        clones = []
        for col_index, cap_info in enumerate(original_labels):
            # fissa gli estremi (prima e ultima capsula)
            is_fixed = (col_index == 0 or col_index == len(original_labels) - 1)

            # crea un nuovo DraggableLabel a partire dal dict cap_info
            clone = DraggableLabel(
                cap_info,
                row_index,
                col_index,
                is_fixed,
                self,                  # parent = questo dialog
                self.visible_number
            )
            clones.append(clone)
            clone.hide()

        return clones



    def init_ui(self):
        
        self.layout = QVBoxLayout()
        self.grid = QGridLayout()

        # Estrae i vassoi (rows) e ricostruisce labels come lista di rows estratte e clonate
        self.cloned_trays = [
            self.clone_labels(self.tray_data[i], i)
            for i in range(len(self.tray_data))
        ]

        self.show_tray(self.current_tray_index)


        self.button = QPushButton("Prosegui")
        self.button.clicked.connect(self.next_tray)

        self.layout.addLayout(self.grid)
        self.layout.addWidget(self.button)
        self.setLayout(self.layout)
        

    def show_tray(self, tray_index):
        # Nascondi tutte le capsule
        for tray in self.cloned_trays:
            for lbl in tray:
                lbl.hide()

        # Svuota layout
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

        tray = self.cloned_trays[tray_index]

        for col, label in enumerate(tray):
            label.col = col
            label.show()  # <-- mostra solo quelle del vassoio attuale
            self.grid.addWidget(label, 0, col)



    def reorder_tray_capsules(self, tray_index):
        # 1) Ordino la lista in-place in base alla posizione x corrente
        tray = self.cloned_trays[tray_index]
        tray.sort(key=lambda lbl: lbl.x())

        # 2) Aggiorno la lista interna
        self.cloned_trays[tray_index] = tray

        # 3) Pulisco completamente il layout
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

        # 4) Ridispongo i widget secondo il nuovo ordine:
        for col, label in enumerate(tray):
            label.col = col
            self.grid.addWidget(label, 0, col)




    def next_tray(self):
        """
        Passa al vassoio successivo.
        Dopo l'ultimo (index 3) il click esegue la chiusura del dialogo
        e mostra la board completa in sola lettura con il pulsante
        «Esegui nuovamente».
        """
        # 1) salva l'ordine del vassoio corrente nel test principale
        self._commit_current_tray()

        # 2) se non abbiamo ancora mostrato tutti i 4 vassoi …
        if self.current_tray_index < 3:
            self.current_tray_index += 1
            self.setWindowTitle(
                f"Test Farnsworth-Munsell - Vassoio {self.current_tray_index + 1}"
            )
            self.show_tray(self.current_tray_index)

            # quando entriamo nell'ultimo vassoio cambiamo il testo del bottone
            if self.current_tray_index == 3:
                self.button.setText("Mostra i risultati")
            return  # aspetta il prossimo click

        # 3) Se eravamo già al quarto vassoio, il click finale chiude il dialogo
        #    e mostra la board definitiva
        self.close()
        self.parent_test.lock_labels()
        self.parent_test.toggle_numbers_btn.show()
        self.parent_test.show()
        

        # # aggiunge il pulsante «Esegui nuovamente» una sola volta
        # if not hasattr(self.parent_test, "restart_btn"):
        #     self._add_restart_button()

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    def _commit_current_tray(self):
        """Aggiorna l'ordine del vassoio corrente nel widget principale."""
        tray = self.cloned_trays[self.current_tray_index]
        # nuovo ordine deciso dall'utente
        sorted_idx = [
            lbl.cap_info["num_caps"]
            for lbl in sorted(tray, key=lambda l: l.x())
        ]

        real_tray = self.parent_test.labels[self.current_tray_index]
        idx2lbl   = {lbl.cap_info["num_caps"]: lbl for lbl in real_tray}

        new_row = [idx2lbl[i] for i in sorted_idx]

        # aggiorno il model
        self.parent_test.labels[self.current_tray_index] = new_row

        # aggiorno la proprietà col di ogni capsula
        for new_col, lbl in enumerate(new_row):
            lbl.col = new_col

        # refresh della UI
        self.parent_test.update_ui()


            

    def _add_restart_button(self):
        """Crea il bottone 'Esegui nuovamente' sotto i vassoi."""
        from PyQt5.QtWidgets import QPushButton

        btn = QPushButton("Esegui nuovamente", self.parent_test)
        self.parent_test.restart_btn = btn

        tot_rows = len(self.parent_test.shuffled_colors) + 4
        self.parent_test.layout.addWidget(
            btn, tot_rows, 0, 1, len(self.parent_test.shuffled_colors[0])
        )

        def restart():
            self.parent_test.close()
            seq = FarnsworthSequentialDialog(self.parent_test)
            seq.exec_()

        btn.clicked.connect(restart)




# -----------------------------------------------------------------------------
#  HOME (bottone 1 = semplice, bottone 2 = esperto)
# -----------------------------------------------------------------------------
class HomeWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FM100 Hue - Home")
        self.setFixedSize(480, 280)

        layout = QVBoxLayout()
        layout.setSpacing(50)

        self.diagnosi_btn = QPushButton("Esegui il test")
        self.analisi_btn = QPushButton("Analisi Risultati (Modalità Esperto)")

        self.diagnosi_btn.clicked.connect(lambda: self.run_test(mode_expert=False, visible_number=False)) # simple mode
        self.analisi_btn.clicked.connect(lambda: self.run_test(mode_expert=True, visible_number=True))    # expert mode

        layout.addStretch()
        layout.addWidget(self.diagnosi_btn)
        layout.addWidget(self.analisi_btn)
        layout.addStretch()
        self.setLayout(layout)

    # --------------------
    def run_test(self, mode_expert = True, visible_number = True):
        self.test_window = FarnsworthTest(mode_expert = mode_expert, visible_number = visible_number)
        self.test_window.show()
        self.close()





# if __name__ == "__main__":
    
#     # 1) Esegui Farnsworth wrapper
#     wrapper = FarnsworthTestWrapper()
#     results = wrapper.run_test_and_get_correction()
    
#     # Ottieni matrix Farnsworth e angolo
#     pca_matrix = results.get("Matrix Correction", np.eye(2))
#     conf_angle = results.get("Confusion Angle (degrees)", 0.0)
#     print("[INFO] Farnsworth PCA:", pca_matrix)
#     print("[INFO] Confusion angle:", conf_angle)


# -----------------------------------------------------------------------------
#  MAIN
# -----------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("FM100 Hue Test - Launcher")

    home = HomeWindow()
    home.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()