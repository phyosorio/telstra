import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import math
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
import re

def encode_cat(df,column):
    '''
    Codifica las etiquetas con valores entre 0 y n_clases
    '''
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    return df

def load_data():
    '''
    Carga los datos originales
    '''
    train_df = pd.read_csv("./dataset/train.csv")
    test_df = pd.read_csv("./dataset/test.csv")
    severity_type_df = pd.read_csv("./dataset/severity_type.csv")
    log_feature_df = pd.read_csv("./dataset/log_feature.csv")
    event_type_df = pd.read_csv("./dataset/event_type.csv")
    resource_type_df = pd.read_csv("./dataset/resource_type.csv")
    return train_df, test_df, severity_type_df, log_feature_df, event_type_df, resource_type_df

def plot_count_class(df,column):
    '''
    Grafica los datos separados por clases
    '''
    a = df[column].value_counts()
    p_class = []
    for i, v in enumerate(a):
        prop = v/sum(a)
        p_class.append(a[i]/sum(a))
    plt.figure(figsize=(10,4))
    ax = sns.countplot(x = column, data = df)
    ax.text(0-0.18,2000, str(round(p_class[0]*100,2))+'%', fontsize=20, color='white')
    ax.text(1-0.18,1000, str(round(p_class[1]*100,2))+'%', fontsize=20, color='white')
    ax.text(2-0.18,200, str(round(p_class[2]*100,2))+'%', fontsize=20, color='white')
    ax.tick_params(labelsize=15)
    ax.set_xlabel(ax.get_xlabel(), fontsize = 20)
    ax.set_ylabel(ax.get_ylabel(), fontsize = 20)
    plt.show()

def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]
    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss


def cv_rf(X, y, model):
    '''
    K-fold validation con clasificador tipo model
    '''

    cv = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
    f1 = []
    loss =[]
    acc = []
    for (train, test), i in zip(cv.split(X, y), range(5)):
        model.fit(X.iloc[train], np.ravel(y.iloc[train]))
        y_pred=model.predict(X.iloc[test])
        f1.append(f1_score(np.ravel(y.iloc[test]), y_pred, average='macro'))
        acc.append(accuracy_score(np.ravel(y.iloc[test]), y_pred))
        y_pred_prob=model.predict_proba(X.iloc[test])
        loss.append(multiclass_log_loss(np.ravel(y.iloc[test]), y_pred_prob, eps=1e-15))
    print('f1: ', np.mean(f1))
    print('score: ', np.mean(loss))
    print('acc: ', np.mean(acc))

def add_elm_nelem(df,id_loc, column, tipo, pattern):
    # Uniremos el dataset con el df que solo contiene id y location

    df = df.merge(id_loc, on= 'id', how = 'left')
    # Agruparemos los datos por id y locacion, esto para saber que tipos salen en cada registro
    # Agruparemos el numero de tipos y los concatenaremos para no perder registro de
    # cuales fueron
    id_loc_sum = df.groupby(['id', 'location'], as_index=False).sum()
    id_loc_count = df.groupby(['id', 'location'], as_index=False).count()
    id_loc_count.rename(columns = {column:column+'_count'}, inplace = True)
    # Como los dataset tienen columnas en comun, las quitamos
    df.drop(['location', column], axis = 1, inplace=True)
    id_loc_sum.drop('location', axis = 1, inplace=True)
    id_loc_count.drop('location', axis = 1, inplace=True)
    # Integramos el dataset.
    df = df.merge(id_loc_sum, on = 'id', how = 'left')
    df = df.merge(id_loc_count, on = 'id', how = 'left')
    # Como ya juntamos todos los eventos por id, tenemos multiples registros repetidos,
    # asi que los quitamos.
    df.drop_duplicates(inplace=True)
    df = df.reset_index().drop('index',axis = 1)
    # Ahora recuperamos los tipos individuales para cada registro. Como cada tipo se concateno
    # 'algo1', 'algo20' -> 'algo1algo20', es necesario encontrar cada evento

    df = encontrar_individuales(df, column, tipo, pattern)
    return df

def encontrar_individuales(df, column, tipo, pattern):
    '''
    Permite encontrar que eventos fueron utilizados en cada registro de forma individual
    Requiere:
                Dataset: df
                Columna donde fueron concatenados los items: column(str)
                Tipo de dataset: tipo(str)
                Patron con el que fue codificado: pattern(str)

    Salidas:
                Dataset concatenado
                Dataset + n columnas donde vale 1 si el item i fue utilizado en el registro o 0 si no
    '''

    incidencias = [re.findall(pattern, item) for item in df[column]]
    df[tipo] = incidencias
    flat_list = [item for sublist in incidencias for item in sublist]
    indi = []
    for i in incidencias:
        for j in i:
            if j not in indi:
                indi.append(j)
    zero_data = np.zeros(shape=(len(df),len(indi)))
    extra_column = pd.DataFrame(zero_data, columns=indi)
    n_col_left = df.columns.value_counts().sum()
    df = pd.concat([df,extra_column], axis =1)

    for i,v in enumerate(incidencias):
        for j in v:
            df.iloc[i,indi.index(j)+n_col_left] = 1
    df = df.drop([column, tipo], axis = 1)
    return df

# Leer los datos
train_df, test_df, severity_type_df, log_feature_df, event_type_df, resource_type_df = load_data()
sub_df = test_df['id']
# Grafica los datos separados por clases
#plot_count_class(train_df,'fault_severity')

# Combinar los datos de train y test en un solo dataframe
train_df['source'] = 'train'
test_df['source'] = 'test'
df = pd.concat([train_df, test_df])
df.location = df.location.str.replace('location ','').astype('int')
df_id_loc = df.loc[:,['id', 'location']] # solo id y locaciones

# calculo de la frecuencia de eventos por su locación
#df['ev_freq'] =(event_type_df.merge(df, on='id', how='left')).groupby('event_type')['location'].transform('count')

# El siguiente dataset que se investigara sera event_type. Lo primero que haremos es simplifar los nombres
# cambienado 'event_type ' por 'e'
event_type_df.event_type = event_type_df.event_type.str.replace('event_type ','e')
# Vamos a calcular el numero de eventos diferentes y las combinaciones.
event_type_df = add_elm_nelem(event_type_df,df_id_loc, 'event_type', 'events', '(e[0-9]*)')


df = df.merge(event_type_df, on = 'id',how='left')

# Calculamos algo de estadística de los eventos por locacion
#df['ev_min_loc']=((df.groupby(['location']))['event_type_count'].transform('min'))
#df['ev_max_loc']=((df.groupby(['location']))['event_type_count'].transform('max'))
#df['ev_mean_loc']=((df.groupby(['location']))['event_type_count'].transform('mean'))


# Ahora vamos con resource_type
# Cambiamos nuevamente el nombre de cada recurso
resource_type_df.resource_type = resource_type_df.resource_type.str.replace('resource_type ','r')
resource_type_df = add_elm_nelem(resource_type_df,df_id_loc, 'resource_type', 'resources', '(r[0-9]*)')
df = df.merge(resource_type_df, on = 'id',how='left')
df['ev_freq_loc'] =df.groupby('location')['event_type_count'].transform('count')
#df['res_freq_loc'] =df.groupby('location')['resource_type_count'].transform('count')
#df['ev_min_loc']=((df.groupby(['location']))['event_type_count'].transform('min'))
#df['ev_max_loc']=((df.groupby(['location']))['event_type_count'].transform('max'))
#df['ev_mean_loc']=((df.groupby(['location']))['event_type_count'].transform('mean'))

# El rango del volúmen es muy grande y se ve que su distribución tiene sesgo a la derecha, por lo que lo transformaremos empleando logaritmo.
log_feature_df['volume'] = log_feature_df.volume.transform(lambda x: np.log(x)+1)


# Distribuimos el volúmen a lo largo de cada tipo de feature y luego calculamos algunos valores estadísticos
a = (pd.get_dummies(log_feature_df, 'log_feature').iloc[:,2:]).multiply(log_feature_df['volume'],axis = 0)
b = pd.concat([log_feature_df, a], axis=1)
b = b.groupby(['id']).sum().reset_index()
b.drop('volume', axis =1, inplace= True)
b['v_total'], b['v_max'], b['v_min'] , b['v_std'], b['v_median'], b['log_count']  = b.iloc[:,1:].replace(0, np.NaN).sum(axis = 1),b.iloc[:,1:].replace(0, np.NaN).max(axis = 1), b.iloc[:,1:].replace(0, np.NaN).min(axis = 1), b.replace(0, np.NaN).iloc[:,1:].std(axis = 1), b.replace(0, np.NaN).iloc[:,1:].median(axis = 1), b.iloc[:,1:].astype(bool).sum(axis=0)
b = b.fillna(value = 0)
df = df.merge(b, on = 'id',how='left')
df['log_freq_loc'] =df.groupby('location')['log_count'].transform('count')
df['vmax_loc']=((df.groupby(['location']))['v_max'].transform('max'))
df['vmin_loc']=((df.groupby(['location']))['v_min'].transform('min'))
df['vstd_loc']=((df.groupby(['location']))['v_min'].transform('std'))

# Por si cambiamos accidentalmente la lase del df al rellenar los NaN en algun punto
df = df.fillna(value = 0)
df[df['source']=='test']['fault_severity'] == np.NaN


# Separamos el conjunto en los de entrenamiento y prueba
severity_type_df1 = severity_type_df.merge(df.loc[:,['id','location']], on = 'id', how = 'left')

severity_type_df1 = severity_type_df1.drop('severity_type',axis =1)
severity_type_df1['index_sev'] = severity_type_df1.groupby('location')['id'].transform(lambda x: np.arange(x.shape[0])+1)
severity_type_df1['index_sev_inv'] = severity_type_df1.groupby('location')['id'].transform(lambda x: (np.arange(x.shape[0])+1)[::-1])
severity_type_df1['diff_index'] = severity_type_df1['index_sev']-severity_type_df1['index_sev_inv']
severity_type_df1.drop('location', axis = 1, inplace = True)


df = df.merge(severity_type_df1, on ='id', how = 'left')
print(df)


test_df = df[df['source']=='test']
test_df = test_df.reset_index()
test_df.drop(['index', 'id'], axis = 1, inplace = True)
train_df = df[df['source']=='train']
train_df = train_df.reset_index()
train_df.drop('index', axis = 1, inplace = True)
test_df.drop(['source', 'fault_severity'], axis =1, inplace=True)
train_df.drop(['source', 'id'], axis =1, inplace=True)

# Opcional, se puede entrenar submuestreando las clases al tamaño de la clase menor.

# count_class_0, count_class_1, count_class_2 = train_df.fault_severity.value_counts()
#df_class_0 = train_df[train_df['fault_severity']==0]
#df_class_1 = train_df[train_df['fault_severity']==1]
#df_class_2 = train_df[train_df['fault_severity']==2]
#df_class_0_under = df_class_0.sample(count_class_2)
#df_class_1_under = df_class_1.sample(count_class_2)
#df_test_under = pd.concat([df_class_0_under, df_class_1_under, df_class_2], axis=0)
#df_test_under.fault_severity.value_counts().plot(kind='bar', title='Count (target)')
#X = df_test_under.loc[:, df_test_under.columns!='fault_severity']
#y = df_test_under.loc[:, df_test_under.columns=='fault_severity']

# Separamos el dataset en características y blanco

X = train_df.loc[:, train_df.columns!='fault_severity']
y = train_df.loc[:, train_df.columns=='fault_severity']

#Utilizamos el clasificador RandomForestClassifier y validación cruzada
rf=RandomForestClassifier()
cv_rf(X, y, rf)

# Para entrenar el modelo, empleamos la tecnica de dividir los datos de entrenamiento en prueba y validación.
# Para el modelo final, se emplean todos los datos.
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, stratify = y)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)



# Entrenamos un modelo con XGBClassifier y otro con RandomForestClassifier
xgb_model2 = xgb.XGBClassifier(objective= 'multi:softprob', n_estimators = 1000, subsample  = 0.8, max_depth =3)
eval_set  = [(X_train,y_train), (X_test,y_test)]
xgb_model2.fit(X_train, y_train, eval_set=eval_set,
        eval_metric="mlogloss", early_stopping_rounds=100)

xgb_pred = xgb_model2.predict(X_test)
cr = classification_report(y_test,xgb_pred)
print(cr)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, xgb_pred)


# Vemos que caracteristicas son las mejores para cada clasificador

sorted_idx = xgb_model2.feature_importances_.argsort()
sorted_idx = sorted_idx[-10:]
plt.barh(X_train.columns[sorted_idx], xgb_model2.feature_importances_[sorted_idx])
plt.xlabel("xgbF Top 10 Feature Importance")
plt.show()




X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, stratify = y)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# Generamos el archivo con las prediciones para evaluar el modelo en el sitio.

# Buscamos optimizar el clasifiador RandomForestClassifier

#from sklearn.model_selection import GridSearchCV
#rf_clf = RandomForestClassifier()
#param_grid = {'max_features':[35,37, 40,42, 45 ],
#             'min_samples_split':[20],
#             'n_estimators': [1500]}
#grid_search = GridSearchCV(rf_clf, param_grid, n_jobs=-1, cv = 5, scoring = 'loss')

predict_test=xgb_model2.predict_proba(test_df)
pred_df=pd.DataFrame(predict_test,columns=['predict_0', 'predict_1', 'predict_2'])
submission=pd.concat([sub_df,pred_df],axis=1)
submission.to_csv('script_pred_atom.csv',index=False,header=True)











#estimators = []
#estimators.append(('logistic',lr))
#estimators.append(('rf',rf))
#estimators.append(('dt',dt))
#estimators.append(('xgb',xgb_model2))
#estimators.append(('ada',ada))
#ensemble = VotingClassifier(estimators, voting = 'soft', n_jobs = -1, )
#ensemble.fit(X_train,y_train)
#ensemble_pred = ensemble.predict(X_test)
#cr = classification_report(y_test,ensemble_pred)
#print(cr)
