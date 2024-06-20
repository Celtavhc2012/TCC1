import os
import numpy as np  # Import NumPy with alias 'np'
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

def simulate_new_log(svm_model, scaler, original_log):
    """
    Função fictícia para simular um novo log com base no modelo SVM treinado.
    Esta função deve ser implementada de acordo com suas necessidades específicas.
    """
    # Implementação fictícia
    return None

def create_petri_net_from_log(log):
    """
    Cria uma rede Petri simples a partir do log.
    Esta função é apenas um exemplo e deve ser adaptada conforme necessário.
    """
    net = PetriNet("SimplePetriNet")

    # Criação dos lugares
    place1 = PetriNet.Place("Place1")
    place2 = PetriNet.Place("Place2")
    net.places.add(place1)
    net.places.add(place2)

    # Criação das transições
    transition1 = PetriNet.Transition("Transition1", "Transition1")
    transition2 = PetriNet.Transition("Transition2", "Transition2")
    net.transitions.add(transition1)
    net.transitions.add(transition2)

    # Criação dos arcos
    petri.utils.add_arc_from_to(place1, transition1, net)
    petri.utils.add_arc_from_to(transition1, place2, net)
    petri.utils.add_arc_from_to(place2, transition2, net)
    petri.utils.add_arc_from_to(transition2, place1, net)

    initial_marking = Marking()
    initial_marking[place1] = 1
    final_marking = Marking()
    final_marking[place2] = 1

    return net, initial_marking, final_marking

def visualize_petri_net(net, initial_marking, final_marking):
    """
    Visualiza a rede Petri.
    """
    gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    pn_visualizer.view(gviz)
    pn_visualizer.save(gviz, "petri_net.png")

def main():
    # Use o caminho absoluto para o arquivo XES
    file_path = 'C:/Users/willi/Desktop/Aulas e atv 9º Periodo/TCC/implementação/log1.xes'

    # Verifica se o arquivo existe
    if not os.path.isfile(file_path):
        raise Exception(f"File does not exist: {file_path}")

    # Lê o arquivo XES
    log = xes_importer.apply(file_path)

    # Converte o log para um DataFrame
    df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)

    # Separação de colunas numéricas e categóricas
    numerical_cols = ['time:timestamp']  # Ajuste as colunas numéricas
    categorical_cols = ['concept:name']  # Ajuste as colunas categóricas

    # Normalização de colunas numéricas
    X_train_num = df[numerical_cols]
    X_test_num = df[numerical_cols]
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_num)
    X_test_num = scaler.transform(X_test_num)

    # Carregamento e processamento das colunas-alvo (ajuste de acordo com o seu caso)
    target_column = 'concept:name'  # Substitua por sua coluna de destino

    # Verificação da existência da coluna alvo
    if target_column not in df.columns:
        raise KeyError(f"Column '{target_column}' not found in DataFrame.")

    y = df[target_column]

    # Processamento de colunas categóricas (exemplo com Label Encoder)
    encoder = LabelEncoder()
    X_train_cat = df[categorical_cols]
    X_test_cat = df[categorical_cols]
    X_train_cat = encoder.fit_transform(X_train_cat.values.ravel())
    X_test_cat = encoder.transform(X_test_cat.values.ravel())  # Corrigido fit_ para fit_transform

    # Combinação dos dados processados
    X_train = np.concatenate([X_train_num, X_train_cat.reshape(-1, 1)], axis=1)
    X_test = np.concatenate([X_test_num, X_test_cat.reshape(-1, 1)], axis=1)

    # Divisão dos dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X_train, y, test_size=0.3, random_state=42)

    # Treinamento do modelo SVM
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)

    # Simulação de um novo log (exemplo fictício)
    new_log = simulate_new_log(svm_model, scaler, log)

    # Criação e visualização de uma rede Petri
    simple_petri_net, initial_marking, final_marking = create_petri_net_from_log(df)
    visualize_petri_net(simple_petri_net, initial_marking, final_marking)

if __name__ == "__main__":
    main()