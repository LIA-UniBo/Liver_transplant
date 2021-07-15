import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
from tensorflow.keras.metrics import TrueNegatives, TruePositives, FalseNegatives, FalsePositives
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

seed = 42

def plot_pie (series: pd.Series, show_plot: bool):
    if show_plot:
        vc = series.value_counts()
        plt.pie(vc, labels=vc.index, autopct='%.1f%%');

def plot_hist (series: pd.Series, show_plot: bool):
    if show_plot:
        plt.hist(series, bins=20, edgecolor='black');
        
def plot_barh (series: pd.Series, show_plot: bool):
    if show_plot:
        vc = series.value_counts(normalize=True, ascending=True)
        plt.barh(vc.index, vc.values, height=0.5, color='orange');

def plot_barv_date (series: pd.Series, show_plot: bool):
    if show_plot:
        vc = series.dt.year.value_counts()
        plt.bar(vc.index, vc.values, color='green', edgecolor='black');
        
def get_freq_nan_std (df: pd.DataFrame, column_name: str):
    return df[column_name].isna().sum() / len(df)

def get_freq_nan_tipo_combinato (df: pd.DataFrame):
    return ((df['iscriz_tx_comb'] == 'SI') & (pd.isna(df['tipo_combinato']))).sum() / len(df)

def get_freq_nan_data_decesso (df: pd.DataFrame):    
    return ((df['causa_uscita'] == 'Deceduto iscritto in lista') & (pd.isna(df['data_decesso']))).sum() / len(df)

def get_freq_nan_tx (df: pd.DataFrame, column_name: str):
    tx = ['TX nel centro di iscrizione',
          'Trapiantato da VIVENTE in questo Centro',
          'Trapianto in altra sede non specificata',
          'Trapianto AIRT fuori Regione']
    
    return ((df['causa_uscita'].isin(tx)) & (pd.isna(df[column_name]))).sum() / len(df)

def get_min_value ():
    return [
        0,        # pers_sesso
        0,        # pers_eta_ingresso_in_lista
        0,        # tot_tx_organo
        1,        # tx_fegato_corrente
        0,        # tx_cuore
        0,        # tx_intestino
        0,        # tx_pancreas
        0,        # tx_rene
        0,        # ric_HBsAg
        0,        # ric_DELTA
        0,        # ric_HCV
        0,        # ric_HCV_RNA
        0,        # ric_HIV
        0,        # Peso
        0,        # Altezza
        0,        # BMI
        0,        # chirurgia_addom
        0,        # trombosi_portale
        0,        # TIPS
        0,        # ric_diabete
        0,        # HCC
        0,        # MELD_base
        0,        # dialisi_ultimi_15gg
        1,        # UNOS
        1,        # CHILD
        0,        # ric_ospedalizzazione_al_tx
        0,        # organo_trapiantato
        0,        # ischemia_fredda
        0,        # ric_ripresa_funzionale
        0,        # donatore_eta
        0,        # donatore_sesso
        0,        # donatore_peso
        0,        # donatore_altezza
        0,        # donatore_BMI
        0,        # donatore_cardiopatia
        0,        # donatore_ipertensione
        0,        # donatore_diabete
        0,        # donatore_diabete_insulinodipendente
        0,        # donatore_dislipidemie
        0,        # donatore_sodio_avg
        0,        # donatore_AST_avg
        0,        # donatore_ALT_avg
        0,        # donatore_bilirubinaTot_avg
        0,        # donatore_fosfatasi_avg
        0,        # donatore_gammaGT_avg
        0,        # donatore_arresto_cardiaco
        0,        # donatore_hcv
        0,        # donatore_HBsAg
        0,        # donatore_HBsAb
        0,        # donatore_antiCoreTot
        0,        # donatore_hbv_dna
        0,        # donatore_hcv_rna
        0,        # donatore_steatosiMacro
        1,        # donatore_livelloRischioPreLT
        0,        # diagnosi_epatocarcinoma
        0,        # diagnosi_cirrosi_hcv
        0,        # diagnosi_cirrosi_laennec
        0,        # diagnosi_ritrapianto_rigetto
        0,        # diagnosi_altro
        0,        # donatore_caus_decesso_emorragia
        0,        # donatore_caus_decesso_trauma
        0,        # donatore_caus_decesso_encefalopatia
        0,        # donatore_caus_decesso_ictus
        0         # donatore_caus_decesso_altro
    ]
    
def get_max_value ():
    return [
        1,        # pers_sesso
        np.inf,   # pers_eta_ingresso_in_lista
        np.inf,   # tot_tx_organo
        np.inf,   # tx_fegato_corrente
        1,        # tx_cuore
        1,        # tx_intestino
        1,        # tx_pancreas
        1,        # tx_rene
        1,        # ric_HBsAg
        1,        # ric_DELTA
        1,        # ric_HCV
        1,        # ric_HCV_RNA
        1,        # ric_HIV
        np.inf,   # Peso
        np.inf,   # Altezza
        np.inf,   # BMI
        1,        # chirurgia_addom
        2,        # trombosi_portale
        1,        # TIPS
        1,        # ric_diabete
        1,        # HCC
        np.inf,   # MELD_base
        1,        # dialisi_ultimi_15gg
        4,        # UNOS
        11,       # CHILD
        2,        # ric_ospedalizzazione_al_tx
        1,        # organo_trapiantato
        np.inf,   # ischemia_fredda
        1,        # ric_ripresa_funzionale
        np.inf,   # donatore_eta
        1,        # donatore_sesso
        np.inf,   # donatore_peso
        np.inf,   # donatore_altezza
        np.inf,   # donatore_BMI
        1,        # donatore_cardiopatia
        1,        # donatore_ipertensione
        1,        # donatore_diabete
        1,        # donatore_diabete_insulinodipendente
        1,        # donatore_dislipidemie
        np.inf,   # donatore_sodio_avg
        np.inf,   # donatore_AST_avg
        np.inf,   # donatore_ALT_avg
        np.inf,   # donatore_bilirubinaTot_avg
        np.inf,   # donatore_fosfatasi_avg
        np.inf,   # donatore_gammaGT_avg
        1,        # donatore_arresto_cardiaco
        1,        # donatore_hcv
        1,        # donatore_HBsAg
        1,        # donatore_HBsAb
        1,        # donatore_antiCoreTot
        1,        # donatore_hbv_dna
        1,        # donatore_hcv_rna
        np.inf,   # donatore_steatosiMacro
        4,        # donatore_livelloRischioPreLT
        1,        # diagnosi_epatocarcinoma
        1,        # diagnosi_cirrosi_hcv
        1,        # diagnosi_cirrosi_laennec
        1,        # diagnosi_ritrapianto_rigetto
        1,        # diagnosi_altro
        1,        # donatore_caus_decesso_emorragia
        1,        # donatore_caus_decesso_trauma
        1,        # donatore_caus_decesso_encefalopatia
        1,        # donatore_caus_decesso_ictus
        1         # donatore_caus_decesso_altro
    ]

def get_cost (y_true, y_pred, cost_matrix: dict):
    # cost_matrix structure: {'TN': TN cost, TP: TP cost, FN: FN cost, FP: FP cost'}
    
    cost = 0
    
    for t, p in zip(y_true, y_pred):
        if t == 0 and p == 0:
            # TN
            cost += cost_matrix['TN']
        elif t == 1 and p == 1:
            # TP
            cost += cost_matrix['TP']
        elif t == 0 and p == 1:
            # FP
            cost += cost_matrix['FP']
        elif t == 1 and p == 0:
            # FN
            cost += cost_matrix['FN']
        else:
            raise ValueError("{} or {} is a value not recognized. It should be 0 or 1!".format(t, p))
            
    return cost

def get_cost_tf (y_true, y_pred, cost_matrix: dict):
    # cost_matrix structure: {'TN': TN cost, TP: TP cost, FN: FN cost, FP: FP cost'}
    
    cost = 0
    
    tn = tf.keras.metrics.TrueNegatives()
    tn.update_state(y_true, y_pred)
    cost += tn.result().numpy() * cost_matrix['TN']
    
    tp = tf.keras.metrics.TruePositives()
    tp.update_state(y_true, y_pred)
    cost += tp.result().numpy() * cost_matrix['TP']
    
    fn = tf.keras.metrics.FalseNegatives()
    fn.update_state(y_true, y_pred)
    cost += fn.result().numpy() * cost_matrix['FN']
    
    fp = tf.keras.metrics.FalsePositives()
    fp.update_state(y_true, y_pred)
    cost += fp.result().numpy() * cost_matrix['FP']
                
    return cost

class CostMetric (tf.keras.metrics.Metric):
    def __init__ (self, cost_matrix: dict, **kwargs):
        super(CostMetric, self).__init__(name='cost', **kwargs)

        self.cost = 0
        self.cost_matrix = cost_matrix
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.cost += get_cost_tf(y_true, y_pred, self.cost_matrix)
        
    def reset_states(self):
        self.cost = 0
        
    def result(self):
          return self.cost

class NeuralNetwork (BaseEstimator, ClassifierMixin):
    def __init__(self,
                 cost_matrix: dict,
                 lr = 0.0001,
                 epochs = 10,
                 batch_size = 8,
                 input_dim = 64,
                 output_dim = 1,
                 hidden_dim = [32, 10],
                 class_weight = {0: 1, 1: 1}):

        self.cost_matrix = cost_matrix
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.class_weight = class_weight
        
        if len(hidden_dim) < 1:
            raise ValueError("List hidden_dim must have at least 1 element!")
        
        self.model = Sequential()
        self.model.add(Dense(hidden_dim[0], input_dim=input_dim, activation='relu'))
        for h in hidden_dim[1:]:
            self.model.add(Dense(h, activation='relu'))
        self.model.add(Dense(output_dim, activation='sigmoid'))
                
        #self.callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

        self.model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=[CostMetric(self.cost_matrix)],
                          run_eagerly=True)
                           
    def fit(self, X, y):
        tf.random.set_seed(42)
        #return self.model.fit(X, y, epochs=self.epochs, callbacks=[self.callback])
        return self.model.fit(X, y,
                              epochs=self.epochs,
                              #validation_split=0.2,
                              class_weight=self.class_weight)
        
    def predict(self, X):
        return self.model.predict(X)