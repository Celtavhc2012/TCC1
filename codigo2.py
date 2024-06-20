import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.simulation import playout

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np

# Função para treinar um modelo SVM com dados do log
def train_svm_model(event_log):
    # Extrair atributos do log
    data, labels = extract_features(event_log)
    
    # Codificar labels para SVM
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Dividir dados em conjunto de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(data, encoded_labels, test_size=0.2, random_state=42)
    
    # Criar e treinar modelo SVM
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    
    # Avaliar o modelo
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia do modelo SVM: {accuracy}')
    
    return svm_model

# Função para extrair features do log de eventos
def extract_features(event_log):
    case_durations = case_statistics.get_all_case_durations(event_log)
    
    data, labels = [], []
    
    for trace in event_log:
        trace_id = trace.attributes['concept:name']
        start_time = trace[0]['time:timestamp']
        end_time = trace[-1]['time:timestamp']
        case_duration = (end_time - start_time).total_seconds()  # Duração em segundos
        data.append([len(trace), case_duration])
        labels.append(trace[0]['concept:name'])
    
    return np.array(data), np.array(labels)

# Função para simular o processo usando o modelo treinado
def simulate_process(event_log, svm_model, num_simulations=5):
    for i in range(num_simulations):
        playout_log = playout.apply(event_log)
        simulated_log = log_converter.apply(playout_log, variant=log_converter.Variants.TO_EVENT_LOG)
        event_log = simulated_log  # Atualiza o log para a próxima simulação
    
    return event_log

# Carregar o log de eventos
log_path = 'C:/Users/willi/Desktop/Aulas e atv 9º Periodo/TCC/implementação/bpi_volvo.xes'
event_log = xes_importer.apply(log_path)

# Treinar modelo SVM com o log de eventos
svm_model = train_svm_model(event_log)

# Simular o processo várias vezes usando o modelo treinado
event_log_final = simulate_process(event_log, svm_model, num_simulations=5)

# Salvar o log simulado
pm4py.write_xes(event_log_final, 'C:/Users/willi/Desktop/Aulas e atv 9º Periodo/TCC/implementação/log_simulado.xes')
