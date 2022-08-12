import pickle
import inflection
import numpy as np
import pandas as pd
import datetime
import math


class Rossmann(object):

    def __init__(self):
        self.homepath = r''

        self.competition_distance_scaler = pickle.load(open(r'parameters/competition_distance_scaler.pkl', 'rb'))
        self.competition_open_since_scaler = pickle.load(open(r'parameters/competition_open_since_year_scaler.pkl', 'rb'))
        self.competition_month_since_scaler = pickle.load(open(r'parameters/competition_time_month_scaler.pkl', 'rb'))
        self.promo_time_week_scaler = pickle.load(open(r'parameters/promo_time_week_scaler.pkl', 'rb'))
        self.promo2_since_year_scaler = pickle.load(open(r'parameters/promo2_since_year_scaler.pkl', 'rb'))
        self.store_type_label_encoder = pickle.load(open(r'parameters/store_type_label_encoder.pkl', 'rb'))
        self.year_scaler = pickle.load(open(r'parameters/year_scaler.pkl', 'rb'))
        self.model_selected_columns = ['store',
                                       'promo',
                                       'store_type',
                                       'assortment',
                                       'competition_distance',
                                       'competition_open_since_year',
                                       'promo2',
                                       'promo2_since_year',
                                       'competition_time_month',
                                       'promo_time_week',
                                       'competition_open_since_month_sin',
                                       'competition_open_since_month_cos',
                                       'month_cos',
                                       'day_of_week_sin',
                                       'day_of_week_cos',
                                       'promo2_since_week_sin',
                                       'promo2_since_week_cos',
                                       'week_of_year_cos',
                                       'day_sin',
                                       'day_cos']

    def data_cleaning(self, df_1):
        # Tudo será passado para minúsculo e com undersocre

        old_columns_name = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo',
                            'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
                            'CompetitionDistance', 'CompetitionOpenSinceMonth',
                            'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
                            'Promo2SinceYear', 'PromoInterval']

        snakecase = lambda x: inflection.underscore(x)

        cols_new = list(map(snakecase, old_columns_name))

        df_1.columns = cols_new

        # Convertendo variável de data para datetime
        format_time = '%Y-%m-%d'
        df_1['date'] = pd.to_datetime(df_1['date'])

        # Quando Competition Distance for vazio, isso quer dizer que não nenhum de competidor por perto daquela loja. Para imputação será utilizado um valor alto.
        df_1['competition_distance'] = df_1['competition_distance'].apply(lambda x: 200000.0 if math.isnan(x) else x)

        # Para os valores referentes as datas, caso o mesmo seja vazio, assumir a data na qual a lojo foi aberta.]
        df_1['competition_open_since_month'] = df_1.apply(
            lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else x[
                'competition_open_since_month'], axis=1)

        df_1['competition_open_since_year'] = df_1.apply(
            lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else x['competition_open_since_year'],
            axis=1)

        df_1['promo2_since_year'] = df_1.apply(
            lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else x['promo2_since_year'], axis=1)

        df_1['promo2_since_week'] = df_1.apply(
            lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else x['promo2_since_week'], axis=1)

        # Creates a month dictionary with all the month matching a number
        month_map = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep',
                     10: 'Oct', 11: 'Nov', 12: 'Dec'}

        # Them fills the 'promo_interval column' with 0
        df_1['promo_interval'].fillna(0, inplace=True)

        # Creates a month map column by mapping the date variable with the dictionary created above
        df_1['month_map'] = df_1['date'].dt.month.map(month_map)

        # Creates a new columns 'is_promo' indicating if a promo is happening
        df_1['is_promo'] = df_1[['promo_interval', 'month_map']].apply(
            lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0,
            axis=1)

        ###  1.6 - Change Data Types and Remove Columns

        df_1['competition_open_since_month'] = df_1['competition_open_since_month'].astype(int)
        df_1['competition_open_since_year'] = df_1['competition_open_since_year'].astype(int)
        df_1['promo2_since_year'] = df_1['promo2_since_year'].astype(int)
        df_1['promo2_since_week'] = df_1['promo2_since_week'].astype(int)

        return df_1

    def feature_engineering(self, df_2):
        # Cria coluna de anos
        df_2['year'] = df_2['date'].dt.year

        # Criar coluna de mes
        df_2['month'] = df_2['date'].dt.month

        # Criar coluna de dia
        df_2['day'] = df_2['date'].dt.day

        # Criar coluna de semanda do ano
        df_2['weekofyear'] = df_2['date'].dt.isocalendar().week

        # Como existe somente a variável de mês e ano na qual o concorrente abriu, uma coluna com a data completa de quando o concorrente abriu será feita;
        df_2['competition_since'] = df_2.apply(
            lambda x: datetime.datetime(year=x['competition_open_since_year'], month=x['competition_open_since_month'],
                                        day=1), axis=1)

        # Como existe somente a variável de mês e ano na qual o concorrente abriu, uma coluna com a data completa de quando o concorrente abriu será feita;
        df_2['competition_time_month'] = df_2.apply(lambda x: int((x['date'] - x['competition_since']).days / 30),
                                                    axis=1)

        # Como existe somente a variável de mês e ano na qual o concorrente abriu, uma coluna com a data completa de quando o concorrente abriu será feita;
        df_2['promo_since'] = df_2.apply(lambda x: str(x['promo2_since_year']) + '-' + str(x['promo2_since_week']),
                                         axis=1)
        df_2['promo_since'] = df_2['promo_since'].apply(
            lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days=7))

        # Como existe somente a variável de mês e ano na qual o concorrente abriu, uma coluna com a data completa de quando o concorrente abriu será feita;
        df_2['promo_time_week'] = df_2.apply(lambda x: int((x['date'] - x['promo_since']).days / 7), axis=1)

        # Converte 'assortment' de letras para
        df_2['assortment'] = df_2.apply(
            lambda x: 'basic' if (x['assortment'] == 'a') else 'extra' if (x['assortment'] == 'b')
            else 'extended' if (x['assortment'] == 'c') else 0, axis=1)

        # Converte 'holyday' de letras para os tipos de feriado
        df_2['state_holiday'] = df_2.apply(
            lambda x: 'public' if (x['state_holiday'] == 'a') else 'easter' if (x['state_holiday'] == 'b')
            else 'christmas' if (x['state_holiday'] == 'c') else 'regular', axis=1)

        ### 2.4 - Removendo colunas e Selecionando dados

        # Selecionando
        df_2 = df_2[(df_2['open'] != 0)]

        # cols_drop = ['open', 'promo_interval']
        cols_drop = ['open', 'promo_interval', 'month_map']
        df_2 = df_2.drop(cols_drop, axis=1)

        return df_2

    def data_preparation(self, df_4):
        # Normalizar os valores para esta variável que contem muitos outliers, entao robust scaler será utilizado
        df_4['competition_distance'] = self.competition_distance_scaler.fit_transform(df_4[['competition_distance']].values)

        # Normalizar os valores para esta variável que contem muitos outliers, entao robust scaler será utilizado
        df_4['competition_open_since_year'] = self.competition_open_since_scaler.fit_transform(
            df_4[['competition_open_since_year']].values)

        # Normalizar os valores para esta variável que contem muitos outliers, entao robust scaler será utilizado
        df_4['promo2_since_year'] = self.promo2_since_year_scaler.fit_transform(df_4[['promo2_since_year']].values)

        # Normalizar os valores para esta variável que contem muitos outliers, entao robust scaler será utilizado
        df_4['year'] = self.year_scaler.fit_transform(df_4[['year']].values)

        # Normalizar os valores para esta variável que contem muitos outliers, entao robust scaler será utilizado
        df_4['competition_time_month'] = self.competition_month_since_scaler.fit_transform(
            df_4[['competition_time_month']].values)

        # Normalizar os valores para esta variável que contem muitos outliers, entao robust scaler será utilizado
        df_4['promo_time_week'] = self.promo_time_week_scaler.fit_transform(df_4[['promo_time_week']].values)

        # state_holiday - One Hot Encoding
        df_4 = pd.get_dummies(df_4, prefix=['state_holiday'], columns=['state_holiday'])

        # store_type - Label Encoding
        df_4['store_type'] = self.store_type_label_encoder.fit_transform(df_4['store_type'])

        # assortment - Ordinal Encoding - utilizando um dicionário
        assortment_dict = {'basic': 1, 'extra': 2, 'extended': 3}
        df_4['assortment'] = df_4['assortment'].map(assortment_dict)

        #### 4.2.2 - Response Variable Enconding

        #df_4['sales'] = np.log1p(df_4['sales'])

        #### 4.2.3 - Circular Encoding

        # Todas as variáveis circulares como dia e mês podem ser convertidas em senos e cossenos com os angulos definidos a partir da teoria do arco de circuferencia
        # Um arco completo possui 360 graus ou 2*pi radianos
        # Utilizando a regra de três é possível deduzir o angulo a partir do comprimento de arco utilizado  alpha = (2*pi/n)*(comprimento de arco)
        # n - número de divisoes dentro do circulo de 360

        # Mes
        df_4['competition_open_since_month_sin'] = df_4['competition_open_since_month'].apply(
            lambda x: np.sin((2 * np.pi / 12) * x))
        df_4['competition_open_since_month_cos'] = df_4['competition_open_since_month'].apply(
            lambda x: np.cos((2 * np.pi / 12) * x))
        df_4['month_sin'] = df_4['month'].apply(lambda x: np.sin((2 * np.pi / 12) * x))
        df_4['month_cos'] = df_4['month'].apply(lambda x: np.cos((2 * np.pi / 12) * x))

        # Dia da semana
        df_4['day_of_week_sin'] = df_4['day_of_week'].apply(lambda x: np.sin((2 * np.pi / 7) * x))
        df_4['day_of_week_cos'] = df_4['day_of_week'].apply(lambda x: np.cos((2 * np.pi / 7) * x))

        # Semana do ano
        df_4['promo2_since_week_sin'] = df_4['promo2_since_week'].apply(lambda x: np.sin((2 * np.pi / 52) * x))
        df_4['promo2_since_week_cos'] = df_4['promo2_since_week'].apply(lambda x: np.cos((2 * np.pi / 52) * x))
        df_4['week_of_year_sin'] = df_4['weekofyear'].apply(lambda x: np.sin((2 * np.pi / 52) * x))
        df_4['week_of_year_cos'] = df_4['weekofyear'].apply(lambda x: np.cos((2 * np.pi / 52) * x))

        # Dia
        df_4['day_sin'] = df_4['day'].apply(lambda x: np.sin((2 * np.pi / 31) * x))
        df_4['day_cos'] = df_4['day'].apply(lambda x: np.cos((2 * np.pi / 31) * x))

        # Dropando colunas que foram convertidas
        df_4.drop(
            columns=['day', 'weekofyear', 'promo2_since_week', 'day_of_week', 'month', 'competition_open_since_month',
                     'competition_since', 'promo_since'], inplace=True)

        return df_4

    def get_prediction(self, model, test_raw, df_5):
        model_data = df_5.loc[:, self.model_selected_columns]

        X = model_data.values

        test_raw['prediction'] = np.expm1(model.predict(X))
        # E necessario retornar o arquivo JSON porque é o padrao de comunicacao entre sistemas

        return test_raw.to_json(date_format='iso', orient='records')


