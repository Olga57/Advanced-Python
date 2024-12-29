import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
from sklearn.linear_model import LinearRegression
import time

# Функция для получения названия сезона
def get_season_name(season_number):
    seasons = {1: 'Зима', 2: 'Весна', 3: 'Лето', 4: 'Осень'}
    return seasons.get(season_number, 'Неизвестный сезон')

# Функция для получения текущей температуры
def get_current_temperature(city, api_key):
    response = requests.get(f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}")
    if response.status_code == 200:
        current_weather = response.json()
        current_temp = current_weather['main']['temp'] - 273.15  
        return current_temp
    else:
        raise Exception("Ошибка получения данных: " + response.json().get("message", "Неизвестная ошибка"))

# Функция для анализа данных по одному городу
def analyze_city(city_data):
    city_data = city_data.copy()
    city_data['season'] = city_data['timestamp'].dt.month % 12 // 3 + 1  

    # Вычисляем скользящее среднее и стандартное отклонение
    city_data['rolling_mean'] = city_data['temperature'].rolling(window=30).mean()
    city_data['rolling_std'] = city_data['temperature'].rolling(window=30).std()

    # Выявление аномалий
    anomalies = city_data[(city_data['temperature'] > city_data['rolling_mean'] + 2 * city_data['rolling_std']) |
                          (city_data['temperature'] < city_data['rolling_mean'] - 2 * city_data['rolling_std'])]

    # Группировка по сезонам и расчет mean и std
    season_profile = city_data.groupby('season')['temperature'].agg(['mean', 'std']).reset_index()

    # Тренд
    X = np.arange(len(city_data)).reshape(-1, 1)
    y = city_data['temperature'].values
    model = LinearRegression().fit(X, y)
    trend = model.coef_[0]

    # Средняя, минимальная и максимальная температура
    average_temp = city_data['temperature'].mean()
    min_temp = city_data['temperature'].min()
    max_temp = city_data['temperature'].max()

    return {
        'city': city_data['city'].iloc[0],
        'average_temp': average_temp,
        'min_temp': min_temp,
        'max_temp': max_temp,
        'season_profile': season_profile,
        'trend': trend,
        'anomalies': anomalies
    }

# Заголовок приложения
st.title("Анализ исторических данных о температуре")

# Загрузка файла с историческими данными
uploaded_file = st.file_uploader("Загрузите файл с историческими данными (CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values(by='timestamp')

    # Выбор города
    cities = data['city'].unique()
    selected_city = st.selectbox("Выберите город", cities)

    # Ввод API-ключа
    api_key = st.text_input("Введите API-ключ OpenWeatherMap")

    if api_key:
        try:
            current_temp = get_current_temperature(selected_city, api_key)
            st.write(f"Текущая температура в {selected_city}: {current_temp:.2f} °C")

            # Определение сезона
            city_data = data[data['city'] == selected_city]
            city_data['season'] = city_data['timestamp'].dt.month % 12 // 3 + 1
            current_season = city_data['season'].iloc[-1]
            season_name = get_season_name(current_season)
            season_profile = city_data.groupby('season')['temperature'].agg(['mean', 'std']).loc[current_season]

            # Проверка на аномальность
            if (current_temp > season_profile['mean'] - 2 * season_profile['std']) and (current_temp < season_profile['mean'] + 2 * season_profile['std']):
                st.write(f"Температура в {selected_city} нормальная для текущего сезона: {season_name}.")
            else:
                st.write(f"Температура в {selected_city} аномальная для текущего сезона: {season_name}.")

        except Exception as e:
            st.error(f"Ошибка: {e}")

    # Описательная статистика
    st.subheader("Описательная статистика")
    st.write(city_data.describe())

    # Визуализация временного ряда температур
    st.subheader("Временной ряд температур")
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=city_data, x='timestamp', y='temperature', label='Температура')
    plt.axhline(y=season_profile['mean'], color='r', linestyle='--', label='Средняя температура')
    plt.title(f"Температура в {selected_city}")
    plt.xlabel("Дата")
    plt.ylabel("Температура (°C)")
    plt.legend()
    st.pyplot(plt)

    # Анализ данных
    analysis_results = analyze_city(city_data)

    # Выделение аномалий
    anomalies = analysis_results['anomalies']
    st.write("Аномалии:")
    st.write(anomalies)

# Сезонные профили
st.subheader("Сезонные профили")
season_profile = city_data.groupby('season')['temperature'].agg(['mean', 'std']).reset_index()
st.write(season_profile)

# Визуализация сезонных профилей с доверительными интервалами
plt.figure(figsize=(10, 5))
sns.barplot(data=season_profile, x='season', y='mean', capsize=0.1)
plt.title("Средняя температура по сезонам")
plt.xlabel("Сезон")
plt.ylabel("Средняя температура (°C)")
st.pyplot(plt)


# Запуск приложения
if __name__ == "__main__":
    st.write("Запустите приложение с помощью команды `streamlit run <имя_файла>.py`")

