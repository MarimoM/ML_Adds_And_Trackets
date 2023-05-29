# Read and preprocess data
import pandas as pd
import numpy as np
df_nuorodos = pd.read_csv('nuorodos_final.csv')
df_grafas = pd.read_csv('grafo_bendras.csv')
df_scraped = pd.read_csv('magistras_2.csv', dtype={'Domenas': 'str', 
                                                   'URL': 'str',
                                                   'URL_ilgis': 'str', 
                                                   'Tipas': 'str',
                                                   'Klase': 'str', 
                                                   'Gautu_issiustu_santykis': 'str',
                                                   'Gauti_slapukai': 'str', 
                                                   'Issiusti slapukai': 'str',
                                                   'Slapuku_dydis': 'str',
                                                   'Slapuku_trukme': 'str',
                                                   'Treciuju_saliu': 'str',
                                                   'Subdomenas': 'str',
                                                   'Parametru_sk': 'str', 
                                                   'Parametrai': 'str'})
df_nuorodos.rename(columns = {'kelias':'Request_URL'}, inplace = True)
df_grafas.rename(columns = {'domenas....domain':'Domenas'}, inplace = True)
df_scraped = df_scraped[df_scraped['URL'].notna() & df_scraped['URL_ilgis'].notna() & df_scraped['Domenas'].notna() & df_scraped['Klase'].notna()]
df_scraped = df_scraped[df_scraped['URL_ilgis'].str.len() <= 5]
df_scraped = df_scraped[df_scraped['Gauti_slapukai'].str.len() <= 4]
df_nuorodos = df_nuorodos.drop_duplicates(subset=['URL'], keep='last')
df_grafas = df_grafas.drop_duplicates(keep='last')
df_grafas = df_grafas.replace(np.nan, 0)
df_scraped = df_scraped.replace(np.nan, 0)
df_nuorodos = df_nuorodos.replace(np.nan, 0)
df_grafas = df_grafas.round({'v_n_ratio': 3})


# In[232]:


santykis = []
for i, j in df_scraped[['Gauti_slapukai', 'Issiusti slapukai']].values:
    gauti = int(i)
    issiusti = int(j)
    if gauti != 0 and issiusti != 0:
        santykis.append(round(gauti/issiusti, 3))
    else:
        santykis.append(0)
df_scraped['Gautu_issiustu_santykis'] = santykis
df_scraped = df_scraped.drop(['Gautu_issiustu_dantykis'], axis=1)
df_scraped['Gautu_issiustu_santykis'] = df_scraped['Gautu_issiustu_santykis'].astype('float')


# In[233]:


df_scraped['Gauti_slapukai'] = df_scraped['Gauti_slapukai'].astype('float')
df_scraped['Issiusti slapukai'] = df_scraped['Issiusti slapukai'].astype('float')
df_scraped['Slapuku_dydis'] = df_scraped['Slapuku_dydis'].astype('float')
df_scraped['Parametru_sk'] = df_scraped['Parametru_sk'].astype('float')

df_nuorodos['vaiku_sk'] = df_nuorodos['vaiku_sk'].astype('float')
df_nuorodos['broliai_seserys'] = df_nuorodos['broliai_seserys'].astype('float')
df_nuorodos['gylis'] = df_nuorodos['gylis'].astype('float')

df_grafas['grafo_dydis'] = df_grafas['grafo_dydis'].astype('float')
df_grafas['nuorodu_kiekis'] = df_grafas['nuorodu_kiekis'].astype('float')
df_grafas['virsuniu_kiekis'] = df_grafas['virsuniu_kiekis'].astype('float')
df_grafas['v_n_ratio'] = df_grafas['v_n_ratio'].astype('float')
df_grafas['briaunos'] = df_grafas['briaunos'].astype('float')


# In[234]:


duom3 = pd.merge(df_scraped, df_grafas, how='inner', on = 'Domenas')
duom3 = pd.merge(duom3, df_nuorodos, how='inner', on = 'URL')


# In[235]:


duom3 = duom3.drop_duplicates('URL')


# In[236]:


def get_url_tevo_atributai_list(df):
    tevo_atributai_list = []
    for i in df["tevo_atributai"].values:
        # split with space
        splitted = i.split()
        for atr in splitted:
            if atr not in tevo_atributai_list:
                tevo_atributai_list.append(atr)
    return tevo_atributai_list

def get_tevo_atr_feature_array(tevu_atributu_list, df):
    t_atr_features = []
    len_atributai = len(tevu_atributu_list)
    for i in df["tevo_atributai"].values:
        nulinis = [0] * len_atributai
        splitted = i.split()
        for atr in splitted:
            index = np.where(np.asarray(tevu_atributu_list) == atr)[0][0]
            nulinis[index] = 1
        t_atr_features.append(nulinis)
    return t_atr_features


# In[237]:


tevo_atributai = get_url_tevo_atributai_list(duom3)
duom3['Tevo_atributai_list'] = get_tevo_atr_feature_array(tevo_atributai, duom3)
duom3 = duom3.drop(['tevo_atributai'], axis=1)


# In[238]:


def get_tevo_kelio_atributus(df):
    tevo_atributai_list = []
    for i in df["tevas"].values:
        splitted = i.split('/')
        for atr in splitted:
            if atr not in tevo_atributai_list:
                tevo_atributai_list.append(atr)
    return tevo_atributai_list

def get_tevo_kelio(tevo_kelias_list, df):
    t_atr_features = []
    len_atributai = len(tevo_kelias_list)
    for i in df["tevas"].values:
        nulinis = [0] * len_atributai
        splitted = i.split('/')
        for atr in splitted:
            index = np.where(np.asarray(tevo_kelias_list) == atr)[0][0]
            nulinis[index] = 1
        t_atr_features.append(nulinis)
    return t_atr_features


# In[239]:


tevo_kelias_list = get_tevo_kelio_atributus(duom3)
duom3['Tevo_kelias'] = get_tevo_kelio(tevo_kelias_list, duom3)
duom3 = duom3.drop(['tevas'], axis=1)


# In[240]:


def get_url_atributai_list(df):
    atributai_list = []
    for i in df["url_atributas"].values:
        # split with space
        splitted = i.split()
        for atr in splitted:
            if atr not in atributai_list:
                atributai_list.append(atr)
    return atributai_list

def get_atr_feature_array(atributu_list, df):
    atr_features = []
    len_atributai = len(atributu_list)
    for i in df["url_atributas"].values:
        nulinis = [0] * len_atributai
        splitted = i.split()
        for atr in splitted:
            index = np.where(np.asarray(atributu_list) == atr)[0][0]
            nulinis[index] = 1
        atr_features.append(nulinis)
    return atr_features


# In[241]:


atributu_list = get_url_atributai_list(duom3)
duom3['URL_atributai'] = get_atr_feature_array(atributu_list, duom3)
duom3 = duom3.drop(['url_atributas'], axis=1)


# In[242]:


def treciuju_saliu_f(df):
    treciuju_saliu = []
    for i in df["Treciuju_saliu"].values:
        if i is None:
            treciuju_saliu.append(0.0)
        elif str(i) == 'TRUE':
            treciuju_saliu.append(1.0)
        else:
            treciuju_saliu.append(0.0)
    return treciuju_saliu


# In[243]:


duom3['Treciuju_saliu_num'] = treciuju_saliu_f(duom3)
duom3 = duom3.drop(['Treciuju_saliu'], axis=1)


# In[244]:


lower_tipas = duom3['Tipas'].str.lower().unique()
for i in lower_tipas:
    if str(i)[len(str(i))-1:] == ';' or len(str(i)) > 100:
        lower_tipas = lower_tipas[lower_tipas != i]
    try:
        float(str(i))
        lower_tipas = lower_tipas[lower_tipas != i]
    except ValueError:
        continue

tipas_numeric = []
for i in duom3['Tipas'].values:
    if str(i)[len(str(i))-1:] == ';':
        tipas = str(i)[:len(str(i))-1]
        index = np.where(lower_tipas == tipas)[0][0]
        tipas_numeric.append(float(index))
        continue
    try:
        index = np.where(lower_tipas == str(i))[0][0]
        tipas_numeric.append(float(index))
    except IndexError:
        tipas_numeric.append(-1.0)
duom3['Tipas_numeric'] = tipas_numeric
duom3 = duom3.drop(['Tipas'], axis=1)


# In[245]:


def gyliai_f(df):
    gyliai = []
    for i in df["gylis"].values:
        if i is None:
            gyliai.append(0)
        else:
            gyliai.append(len(str(i)))
    return gyliai

def broliai_seserys_f(df):
    broliai_seserys = []
    for i in df["broliai_seserys"].values:
        if i is None:
            broliai_seserys.append(0)
        else:
            broliai_seserys.append(len(str(i)))
    return broliai_seserys

duom3['Gylis'] = gyliai_f(duom3)
duom3 = duom3.drop(['gylis'], axis=1)
duom3['Vaikai'] = duom3['vaiku_sk']
duom3 = duom3.drop(['vaiku_sk'], axis=1)
duom3['Broliai_seserys'] = broliai_seserys_f(duom3)
duom3 = duom3.drop(['broliai_seserys'], axis=1)


# In[246]:


def get_subdomain_vector(df):
    df['Subdomain_vectorized'] = None
    for i, index in zip(df["Subdomenas"], df.index.values):
        vektor = []
        if i == 0:
             vektor.append([0])
        try:
            for x in i:
                vektor.append(ord(x))
        except TypeError:
            vektor.append(i)
        df.at[index, 'Subdomain_vectorized'] = vektor
    return df

def get_domain_vector(df):
    df['Domenas_vectorized'] = None
    for i, index in zip(df["Domenas"], df.index.values):
        vektor = []
        try:
            for x in i:
                vektor.append(ord(x))
        except TypeError:
            vektor.append(i)
        df.at[index, 'Domenas_vectorized'] = vektor
    return df

def get_param_vektor(df):
    param_vektor = []
    df['Parametrai_vectorized'] = None
    for i, index in zip(df["Parametrai"], df.index.values):
        vektor = []
        if i == None:
            vektor.append(0)
        else:
            try:
                for x in i:
                    vektor.append(ord(x))
            except TypeError:
                vektor.append(i)
        df.at[index, 'Parametrai_vectorized'] = vektor
    return df


# In[247]:


get_subdomain_vector(duom3)
get_domain_vector(duom3)
get_param_vektor(duom3)
duom3 = duom3.drop(['Subdomenas'], axis=1)
duom3 = duom3.drop(['Domenas'], axis=1)
duom3 = duom3.drop(['Parametrai'], axis=1)


# In[248]:


def create_word_vek(x):
    for i in x:
        word.append(ord(i))
    return word

# If URL length < 200 add 0 in the beginning
def drop_length(X_1, max_length = 100):
    import numpy as np
    duomenis_3_modeliui = []
    for i in X_1:
        if len(i) < max_length:
            kiek_reikia = max_length - len(i)
            arr0 = np.array([0] * kiek_reikia)
            arr0 = np.hstack((arr0, i))
            duomenis_3_modeliui.append(arr0)
        elif len(i) == max_length:
            duomenis_3_modeliui.append(i)
        else:
            duomenis_3_modeliui.append(i[0:max_length])
    return duomenis_3_modeliui

X_1 = []
len_schema = 9
for x, index in zip(duom3["URL"], duom3.index.values):
    x = str(x)
    word = []
    if len(x) <= 200:
        word = create_word_vek(x)
    else:
        flat_list_subdom = duom3['Subdomain_vectorized'][index]
        if len(x) - 9 <= 200:
            x = x[len_schema:len(x)-len_schema-1]
            word = create_word_vek(x)
        elif (len(x) - 9 - len(flat_list_subdom)) <= 200:
            len_dom_schema = len(flat_list_subdom) + len_schema
            x = x[len_dom_schema:len(x)-len_dom_schema-1]
            word = create_word_vek(x)
        else:
            len_dom_schema = len(flat_list_subdom) + len_schema
            x = x[len_dom_schema:(199+len_dom_schema+1)]
            word = create_word_vek(x)
    X_1.append(word)
duom3['URL_vectorized'] = drop_length(X_1)


# In[249]:


duom3 = duom3.drop(['URL'], axis=1)


# In[250]:


def get_klase_numeric(df):
    klase = []
    for i in df["Klase"].values:
        if i == 'NOTCLEAN':
            klase.append(1)
        else:
            klase.append(0)
    return klase


# In[251]:


duom3['Klase_numeric'] = get_klase_numeric(duom3)
duom3 = duom3.drop(['Klase'], axis=1)


# In[257]:


sub_val = duom3['Subdomain_vectorized'].values.tolist()
duom3['Subdomain_vectorized_length'] = drop_length(sub_val, max_length=50)
param_val = duom3['Parametrai_vectorized'].values.tolist()
duom3['Parametrai_vectorized_length'] = drop_length(param_val, max_length=100)
dom_val = duom3['Domenas_vectorized'].values.tolist()
duom3['Domenas_vectorized_length'] = drop_length(dom_val, max_length=45)


# In[253]:


duom3 = duom3.drop(['Subdomain_vectorized'], axis=1)
duom3 = duom3.drop(['Parametrai_vectorized'], axis=1)
duom3 = duom3.drop(['Domenas_vectorized'], axis=1)


# In[254]:


duom3 = duom3.drop(['Unnamed: 14'], axis=1)


# In[255]:


duom3 = duom3.drop(['v_n_ratio'], axis=1)


# In[29]:


df_modeliui3 = pd.DataFrame(data = [duom3['nuorodu_kiekis'], duom3['Parametru_sk'], duom3['Tipas_numeric'], 
                                    duom3['Gylis'], 
                                    duom3['Vaikai'], duom3['Broliai_seserys'], duom3['briaunos'],
                                    duom3['Treciuju_saliu_num'], duom3['URL_atributai'], duom3['Tevo_kelias'], 
                                    duom3['Tevo_atributai_list'], duom3['virsuniu_kiekis'], duom3['grafo_dydis'], 
                                    duom3['Gautu_issiustu_santykis'], duom3['Slapuku_trukme'],
                                    duom3['Slapuku_dydis'], duom3['Issiusti slapukai'], duom3['Gauti_slapukai'], 
                                    duom3['URL_ilgis'], duom3['URL_vectorized'], duom3['Parametrai_vectorized_length'], 
                                    duom3['Domenas_vectorized_length'], duom3['Subdomain_vectorized_length'],])


# In[30]:


df_modeliui1 = pd.DataFrame(data = [duom3['URL_vectorized']])


# In[31]:


df_modeliui2 = pd.DataFrame(data = [duom3['Parametru_sk'], duom3['Tipas_numeric'],
                                    duom3['Treciuju_saliu_num'], duom3['URL_atributai'],
                                    duom3['Gautu_issiustu_santykis'], duom3['Slapuku_trukme'],
                                    duom3['Slapuku_dydis'], duom3['Issiusti slapukai'], duom3['Gauti_slapukai'], 
                                    duom3['URL_ilgis'], duom3['URL_vectorized'], duom3['Parametrai_vectorized_length'], 
                                    duom3['Domenas_vectorized_length'], duom3['Subdomain_vectorized_length'],])


# In[32]:


duomenys_3_modeliui = []
for column in df_modeliui3:
    duomenys_3_modeliui.append(df_modeliui3[column].to_numpy())
unlist_X3 = []
counter = 0
for i in duomenys_3_modeliui:
    concatenated = np.concatenate(([np.asarray(duomenys_3_modeliui[counter][0], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][1], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][2], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][3], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][4], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][5], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][6], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][7], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][8], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][9], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][10], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][11], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][12], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][13], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][14], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][15], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][16], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][17], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][18], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][19], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][20], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][21], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][22], dtype=np.float32)
                                   ]), axis=None)
    unlist_X3.append(concatenated)
    counter = counter + 1


# In[242]:


duomenys_2_modeliui = []
for column in df_modeliui2:
    duomenys_2_modeliui.append(df_modeliui2[column].to_numpy())
unlist_X2 = []
counter = 0
for i in duomenys_2_modeliui:
    concatenated = np.concatenate(([np.asarray(duomenys_2_modeliui[counter][0], dtype=np.float32),
                                    np.asarray(duomenys_2_modeliui[counter][1], dtype=np.float32),
                                    np.asarray(duomenys_2_modeliui[counter][2], dtype=np.float32),
                                    np.asarray(duomenys_2_modeliui[counter][3], dtype=np.float32),
                                    np.asarray(duomenys_2_modeliui[counter][4], dtype=np.float32),
                                    np.asarray(duomenys_2_modeliui[counter][5], dtype=np.float32),
                                    np.asarray(duomenys_2_modeliui[counter][6], dtype=np.float32),
                                    np.asarray(duomenys_2_modeliui[counter][7], dtype=np.float32),
                                    np.asarray(duomenys_2_modeliui[counter][8], dtype=np.float32),
                                    np.asarray(duomenys_2_modeliui[counter][9], dtype=np.float32),
                                    np.asarray(duomenys_2_modeliui[counter][10], dtype=np.float32),
                                    np.asarray(duomenys_2_modeliui[counter][11], dtype=np.float32),
                                    np.asarray(duomenys_2_modeliui[counter][12], dtype=np.float32),
                                    np.asarray(duomenys_2_modeliui[counter][13], dtype=np.float32)
                                   ]), axis=None)
    unlist_X2.append(concatenated)
    counter = counter + 1


# In[243]:


duomenys_1_modeliui = []
for column in df_modeliui1:
    duomenys_1_modeliui.append(df_modeliui1[column].to_numpy())
unlist_X1 = []
counter = 0
for i in duomenys_1_modeliui:
    concatenated = np.concatenate((np.asarray(duomenys_1_modeliui[counter][0])), axis=None)
    unlist_X1.append(concatenated)
    counter = counter + 1


# In[33]:


atsakymai_3_modeliui = []
for i, index in zip(duom3["Klase_numeric"], duom3.index.values):
    if i == 0:
        atsakymai_3_modeliui.append(0)
    else:
        atsakymai_3_modeliui.append(1)


# In[34]:


from sklearn.model_selection import train_test_split
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(unlist_X3, atsakymai_3_modeliui, test_size=0.1, shuffle=True)


# In[204]:


from sklearn.model_selection import train_test_split
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(unlist_X2, atsakymai_3_modeliui, test_size=0.1, shuffle=True)


# In[206]:


from sklearn.model_selection import train_test_split
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(unlist_X1, atsakymai_3_modeliui, test_size=0.1, shuffle=True)


# In[35]:


# Model 3
from tensorflow.keras.layers import Dense, BatchNormalization
import tensorflow as tf
model3 = tf.keras.Sequential([
    BatchNormalization(),
    tf.keras.layers.Dense(1024, activation='relu', name = 'dense0'),
    BatchNormalization(),
    tf.keras.layers.Dense(512, activation='relu', name = 'dense1'),
    BatchNormalization(),
    tf.keras.layers.Dense(256, activation='relu', name = 'dense2'),
    tf.keras.layers.Dense(1, activation='sigmoid', name = 'output'),
])


# In[245]:


# Model 2
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization
model2 = tf.keras.Sequential([
    BatchNormalization(),
    tf.keras.layers.Dense(1024, activation='relu', name = 'dense0'),
    BatchNormalization(),
    tf.keras.layers.Dense(512, activation='relu', name = 'dense1'),
    BatchNormalization(),
    tf.keras.layers.Dense(264, activation='relu', name = 'dense2'),
    BatchNormalization(),
    tf.keras.layers.Dense(1, activation='sigmoid', name = 'output'),
])


# In[246]:


# Model 1
import tensorflow as tf
model1 = tf.keras.Sequential([
    tf.keras.layers.Embedding(200, 32, input_length=200),
    tf.keras.layers.Masking(),
    tf.keras.layers.Flatten(input_shape=(200, 32)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])


# In[680]:


def model_builder(hp):
    model = tf.keras.Sequential()
    hp_units = hp.Int('units', min_value=64, max_value=128, step=32)
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model


# In[703]:


import keras_tuner as kt
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='random_directory',
                     project_name='model_3_tuning')
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
tuner.search(np.asarray(X_train_3), np.array(y_train_3), epochs=10, validation_split=0.2, callbacks=[stop_early])
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)
history = model.fit(np.asarray(X_train_3), np.array(y_train_3), epochs=10, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))


# In[687]:


eval_result = model.evaluate(np.asarray(X_test_3), np.array(y_test_3))


# In[36]:


opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)
model3.compile(loss = "binary_crossentropy", optimizer = opt, metrics="acc")
history3 = model3.fit(np.asarray(unlist_X3), np.array(atsakymai_3_modeliui), batch_size=256, epochs=10, verbose=1,  validation_split=0.2)


# In[40]:


opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model3.compile(loss = "binary_crossentropy", optimizer = opt, metrics="acc")
history3 = model3.fit(np.asarray(unlist_X3), np.array(atsakymai_3_modeliui), batch_size=256, epochs=10, verbose=1,  validation_split=0.2)


# In[250]:


opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model2.compile(loss = "binary_crossentropy", optimizer = opt, metrics="acc")
history2 = model2.fit(np.asarray(unlist_X2), np.array(atsakymai_3_modeliui), batch_size=256, epochs=10, verbose=1,  validation_split=0.2)


# In[251]:


opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model1.compile(loss = tf.keras.losses.BinaryCrossentropy(), optimizer = opt, metrics="acc")
history1 = model1.fit(np.array(unlist_X1), np.array(atsakymai_3_modeliui), batch_size=256, epochs=10, validation_split = 0.2, verbose=1)


# In[94]:


# Confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def klasifikavimo_matrica(classes_x, y_test):
    con_mat = tf.math.confusion_matrix(labels=y_test, predictions=classes_x).numpy()
    cm = confusion_matrix(y_test, classes_x)
    labels = ["Clean", "Tracker"]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
    return cm

def testavimas(model, X_test):
    predict_x = model.predict(np.array(X_test))
    classes_x = []
    for i in predict_x:
        if i >= 0.5:
            classes_x.append(1)
        else:
            classes_x.append(0)
    return classes_x, predict_x

def parametrai(cm):
    precision = cm[1][1]/(cm[1][1]+cm[1][0])
    recall = cm[0][0]/(cm[0][0]+cm[1][0])
    acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[1][0]+cm[0][1])
    fp = (cm[0][1]/(cm[0][1] + cm[1][1]))
    fn = (cm[1][0]/(cm[1][0] + cm[0][0]))
    print(f'Precision: {precision}')
    print(f'Recall: {recall}') 
    print(f'Accuracy: {acc}')
    print(f'False positive rate: {fp}')
    print(f'False negative rate: {fn}')
    
# ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

def ROC_kreive(classes_x, y_test):
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, classes_x)
    auc_keras = auc(fpr_keras, tpr_keras)
    print(f'FPR: {fpr_keras}, TPR: {tpr_keras}')
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='AUC = {:.3f}'.format(auc_keras))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    
# Learning curves
import matplotlib.pyplot as plt
def mokymosi_kreives(history):
    # Accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['acc', 'val_acc'], loc='upper left')
    plt.show()
    # Loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.show()


# In[710]:


class_x, predict_x = testavimas(model3, X_test_3)
cm = klasifikavimo_matrica(class_x, y_test_3)
parametrai(cm)
mokymosi_kreives(history3)
ROC_kreive(class_x, y_test_3)


# In[464]:


class_x, predict_x = testavimas(model2, X_test_2)
cm = klasifikavimo_matrica(class_x, y_test_2)
parametrai(cm)
mokymosi_kreives(history2)
ROC_kreive(class_x, y_test_2)


# In[404]:


class_x, predict_x = testavimas(model1, X_test_1)
cm = klasifikavimo_matrica(class_x, y_test_1)
parametrai(cm)
mokymosi_kreives(history1)
ROC_kreive(class_x, y_test_1)


# In[45]:

# Random forest for most important feature selection
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 10, verbose=1)
fitted = rf.fit(X_train_3, y_train_3);



# In[ ]:


df_modeliui3 = pd.DataFrame(data = [duom3['nuorodu_kiekis'], duom3['Parametru_sk'], duom3['Tipas_numeric'], 
                                    duom3['Gylis'], 
                                    duom3['Vaikai'], duom3['Broliai_seserys'], duom3['briaunos'],
                                    duom3['Treciuju_saliu_num'], duom3['URL_atributai'], duom3['Tevo_kelias'], 
                                    duom3['Tevo_atributai_list'], duom3['virsuniu_kiekis'], duom3['grafo_dydis'], 
                                    duom3['Gautu_issiustu_santykis'], duom3['Slapuku_trukme'],
                                    duom3['Slapuku_dydis'], duom3['Issiusti slapukai'], duom3['Gauti_slapukai'], 
                                    duom3['URL_ilgis'], duom3['URL_vectorized'], duom3['Parametrai_vectorized_length'], 
                                    duom3['Domenas_vectorized_length'], duom3['Subdomain_vectorized_length'],])


# In[47]:


import matplotlib.pyplot as plt
from statistics import mean
reg = fitted
lst = np.append(reg.feature_importances_[0:7], mean(list(reg.feature_importances_[7:162])))
lst = np.append(lst,  mean(list(reg.feature_importances_[162:212])))
lst = np.append(lst,  mean(list(reg.feature_importances_[212:454])))
lst = np.append(lst, reg.feature_importances_[454:461])
lst = np.append(lst,  mean(list(reg.feature_importances_[461:661])))
lst = np.append(lst,  mean(reg.feature_importances_[661:861]))
lst = np.append(lst,  mean(list(reg.feature_importances_[861:896])))
lst = np.append(lst,  mean(list(reg.feature_importances_[896:936])))
plt.bar(range(0, len(lst)), lst)
plt.xticks(range(0, len(lst)), ['Nuorodų skaičius', 'Parametrų skaičius', 'Šaltinio tipas', 
                                'Nuorodos gylis', 'Vaikų skaičius', 'Brolių ir seserų skaičius', 
                                'Trečiųjų šalių', 'Nuorodos atributai', 'Tėvo kelias',
                                'Tėvo atributai', 'Viršūnių skaičius', 'Grafo dydis', 
                                'Slapukų santykis', 'Slapukų trūkmė', 'Slapukų dydis', 
                                'Gautų slapukų dydis', 'Nuorodos ilgis', 'Nuoroda', 
                                'Parametrai', 'Domenas', 'Subdomenas' ], rotation=90)
plt.show()


# In[264]:

# Correlation matrix
import pandas as pd
import seaborn as sns
duom_chanded_names = duom3

duom_chanded_names.rename(columns = {'Gauti_slapukai':'Received cookies'}, inplace = True)
duom_chanded_names.rename(columns = {'Issiusti slapukai':'Send cookies'}, inplace = True)
duom_chanded_names.rename(columns = {'Parametru_sk':'Parameters number'}, inplace = True)
duom_chanded_names.rename(columns = {'Gautu_issiustu_santykis':'Received send cookie rate'}, inplace = True)
duom_chanded_names.rename(columns = {'grafo_dydis':'Graph size'}, inplace = True)
duom_chanded_names.rename(columns = {'nuorodu_kiekis':'Reference number'}, inplace = True)
duom_chanded_names.rename(columns = {'virsuniu_kiekis':'Vertex number'}, inplace = True)
duom_chanded_names.rename(columns = {'briaunos':'Edge number'}, inplace = True)
duom_chanded_names.rename(columns = {'Treciuju_saliu_num':'Third party'}, inplace = True)
duom_chanded_names.rename(columns = {'Tipas_numeric':'Source type'}, inplace = True)
duom_chanded_names.rename(columns = {'Gylis':'URL node depth'}, inplace = True)
duom_chanded_names.rename(columns = {'Vaikai':'Children number'}, inplace = True)
duom_chanded_names.rename(columns = {'Broliai_seserys':'Sibling number'}, inplace = True)
duom_chanded_names.rename(columns = {'Slapuku_dydis':'Cookie size'}, inplace = True)
duom_chanded_names.rename(columns = {'Slapuku_trukme':'Cookie duration'}, inplace = True)
duom_chanded_names.rename(columns = {'URL_atributai':'URL attributes'}, inplace = True)
duom_chanded_names.rename(columns = {'URL_ilgis':'URL length'}, inplace = True)
duom_chanded_names.rename(columns = {'URL_vectorized':'URL'}, inplace = True)
duom_chanded_names.rename(columns = {'Parametrai_vectorized_length':'Parameters'}, inplace = True)
duom_chanded_names.rename(columns = {'Domenas_vectorized_length':'Domain'}, inplace = True)
duom_chanded_names.rename(columns = {'Subdomain_vectorized_length':'Subdomain'}, inplace = True)
duom_chanded_names.rename(columns = {'Klase_numeric':'Label'}, inplace = True)

duom_chanded_names.rename(columns = {'Tevo_kelias':'Father path'}, inplace = True)
duom_chanded_names.rename(columns = {'Tevo_atributai_list':'Father attributes'}, inplace = True)

duom_chanded_names = duom_chanded_names.drop(['Third party'], axis=1)
duom_chanded_names = duom_chanded_names.drop(['Source type'], axis=1)
duom_chanded_names = duom_chanded_names.drop(['Label'], axis=1)

corr_matrix = duom_chanded_names.corr()
sns.heatmap(corr_matrix, center=0)


# ## Training NN with most important parameters

# In[696]:


df_modeliui3 = pd.DataFrame(data = [duom3['nuorodu_kiekis'], duom3['Parametru_sk'], duom3['Tipas_numeric'],
                                    duom3['Broliai_seserys'], duom3['Treciuju_saliu_num'], duom3['virsuniu_kiekis'], 
                                    duom3['grafo_dydis'], duom3['Slapuku_dydis'], duom3['Gauti_slapukai'], 
                                    duom3['URL_ilgis'], duom3['URL_vectorized'], duom3['Domenas_vectorized_length'], 
                                    duom3['Subdomain_vectorized_length'],])
duomenys_3_modeliui = []
for column in df_modeliui3:
    duomenys_3_modeliui.append(df_modeliui3[column].to_numpy())
unlist_X = []
counter = 0
for i in duomenys_3_modeliui:
    concatenated = np.concatenate(([np.asarray(duomenys_3_modeliui[counter][0], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][1], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][2], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][3], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][4], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][5], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][6], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][7], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][8], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][9], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][10], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][11], dtype=np.float32),
                                    np.asarray(duomenys_3_modeliui[counter][12], dtype=np.float32)
                                   ]), axis=None)
    unlist_X.append(concatenated)
    counter = counter + 1

atsakymai_3_modeliui = []
for i, index in zip(duom3["Klase_numeric"], duom3.index.values):
    if i == 0:
        atsakymai_3_modeliui.append(0)
    else:
        atsakymai_3_modeliui.append(1)


# In[697]:


from sklearn.model_selection import train_test_split
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(unlist_X, atsakymai_3_modeliui, test_size=0.1, shuffle=True)
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization
model3 = tf.keras.Sequential([
    BatchNormalization(),
    tf.keras.layers.Dense(1024, activation='relu', name = 'dense0'), 
    BatchNormalization(),
    tf.keras.layers.Dense(512, activation='relu', name = 'dense1'),
    tf.keras.layers.Dense(264, activation='relu', name = 'dense2'),
    tf.keras.layers.Dense(1, activation='sigmoid', name = 'output'),
])
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model3.compile(loss = "binary_crossentropy", optimizer = opt, metrics="acc")
history3 = model3.fit(np.asarray(X_train_3), np.array(y_train_3), batch_size=512, epochs=20, verbose=1,  validation_split=0.2)


# In[699]:


class_x, predict_x = testavimas(model3, X_test_3)
cm = klasifikavimo_matrica(class_x, y_test_3)
parametrai(cm)
mokymosi_kreives(history3)
ROC_kreive(class_x, y_test_3)


# ## Cross-vaidation

# In[700]:


from sklearn.model_selection import KFold
inputs = np.concatenate((X_train_3, X_test_3), axis=0)
targets = np.concatenate((y_train_3, y_test_3), axis=0)
kfold = KFold(n_splits=5, shuffle=True)
fold_no = 1
acc_per_fold=[]
loss_per_fold=[]

for train, test in kfold.split(inputs, targets):
    model = tf.keras.Sequential([
    BatchNormalization(),
    tf.keras.layers.Dense(1024, activation='relu', name = 'dense0'), 
    BatchNormalization(),
    tf.keras.layers.Dense(512, activation='relu', name = 'dense1'),
    tf.keras.layers.Dense(264, activation='relu', name = 'dense2'),
    tf.keras.layers.Dense(1, activation='sigmoid', name = 'output')])
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss = "binary_crossentropy", optimizer = opt, metrics="acc")
    print(f'Training for fold {fold_no} ...')
    history = model.fit(inputs[train], targets[train], batch_size=512, epochs=20, verbose=0,  validation_split=0.2)
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%, val_acc: {history.history["val_acc"][19]} val_loss: {history.history["val_loss"][19]}')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    fold_no = fold_no + 1
    


# In[211]:


rolling3_tuned = np.loadtxt('rolling_test_3_tuned')
rolling_answers = np.loadtxt('rolling_test_3_answers')
rolling2 = np.loadtxt('rolling_test_2')
rolling1 = np.loadtxt('rolling_test_1')
rolling3 = np.loadtxt('rolling_test_3')
rolling_answers = np.loadtxt('rolling_test_3_answers')


# In[264]:


# Rolling forecast
class_x, predict_x = testavimas(model3, rolling3_tuned)
cm = klasifikavimo_matrica(class_x, rolling_answers)
parametrai(cm)
mokymosi_kreives(history3)
ROC_kreive(class_x, rolling_answers)


# In[254]:


# Rolling not tuned
class_x, predict_x = testavimas(model3, rolling3)
cm = klasifikavimo_matrica(class_x, rolling_answers)
parametrai(cm)

