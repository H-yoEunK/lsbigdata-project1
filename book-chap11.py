import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import folium
from folium.plugins import MarkerCluster

geo_seoul = json.load(open("./data/SIG_Seoul.geojson", encoding = "UTF-8"))

type(geo_seoul)
len(geo_seoul)
geo_seoul.keys()
geo_seoul["features"][0]["properties"]
geo_seoul["features"][0].keys()
geo_seoul["features"][0]["geometry"]

lstCoordinate = geo_seoul["features"][0]["geometry"]["coordinates"]

lstCoordinate[0][0]
arrCoordinate = np.array(lstCoordinate[0][0])
x = arrCoordinate[:, 0]
y = arrCoordinate[:, 1]

plt.plot(x[::10], y[::10])
plt.show()
plt.clf()


def drawMap(n):
  gu_name = geo_seoul["features"][n]["properties"]["SIG_KOR_NM"]
  lstCoordinate = geo_seoul["features"][n]["geometry"]["coordinates"]
  lstCoordinate[0][0]
  arrCoordinate = np.array(lstCoordinate[0][0])
  x = arrCoordinate[:, 0]
  y = arrCoordinate[:, 1]
  
  plt.rcParams.update({"font.family": "Malgun Gothic"})
  plt.plot(x[::10], y[::10])
  plt.title(gu_name)
  plt.show()
  plt.clf()
  return


drawMap(11)

# 그냥 해본 거 (넘어가기)-------------------------------------------------------

# [{'type': 'Feature', 'properties': {'SIG_CD': '11110', 'SIG_ENG_NM': 'Jongno-gu', 'SIG_KOR_NM': '종로구'}, 'geometry': {'type': 'MultiPolygon', 'coordinates': 
df = pd.DataFrame(columns=['SIG_KOR_NM', 'x', 'y'])

for i in range(25):
    xy = geo_seoul["features"][i]["geometry"]["coordinates"][0][0]
    colx = []
    coly = []
    for k in xy:
        colx.append(k[0])
        coly.append(k[1])
    df.loc[i] = [geo_seoul["features"][i]["properties"]["SIG_KOR_NM"], np.array(colx), np.array(coly)]

df

# ------------------------------------------------------------------------------

data = []
for i in range(25):
    xy = geo_seoul["features"][i]["geometry"]["coordinates"][0][0]
    for k in xy:
        data.append([geo_seoul["features"][i]["properties"]["SIG_KOR_NM"], k[0], k[1]])

df = pd.DataFrame(data, columns=['SIG_KOR_NM', 'x', 'y'])


# df.plot(kind = 'scatter', x = "x", y = "y", style = 'o', s = 1)
sns.scatterplot(data = df, x = 'x', y = 'y', hue = 'SIG_KOR_NM', s = 2, legend = False)
plt.show()
plt.clf()

# 서울 그래프 그리기
gangnam_df = df.assign(is_gangnam = np.where(df['SIG_KOR_NM'] == "강남구", "강남", "안강남"))

sns.scatterplot(data= gangnam_df, x = 'x', y = 'y', legend = False, 
                palette = {"안강남": "grey", "강남": "red"}, hue = 'is_gangnam', s = 2)
plt.show()
plt.clf()


# ------------------------------------------------------------------------------

geo_seoul = json.load(open("./data/SIG_Seoul.geojson", encoding = "UTF-8"))

geo_seoul["features"][0]["properties"]

df_pop = pd.read_csv("./data/Population_SIG.csv")

df_seoulpop = df_pop.iloc[:26]

df_seoulpop["code"] = df_seoulpop["code"].astype(str)
df_seoulpop.info()



# !pip install folium

center_x=result["x"].mean()
center_y=result["y"].mean()

# p.304
# 밑바탕
map_sig=folium.Map(location = [37.551, 126.973],
                  zoom_start = 12,
                  tiles="cartodbpositron")


bins = df_seoulpop["pop"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1])

# 코로플릿
folium.Choropleth(
    geo_data = geo_seoul,
    data = df_seoulpop,
    columns = ("code", "pop"),
    bins = bins, # 추가 옵션
    fill_color = "viridis", # 추가 옵션
    key_on = "feature.properties.SIG_CD").add_to(map_sig)
    
# 점 찍는 방법
folium.Marker([37.583744, 126.983800], popup = "종로구").add_to(map_sig)

map_sig.save("map_seoul.html")



# House Price ------------------------------------------------------------------

house_df = pd.read_csv("data/houseprice/houseprice-with-lonlat.csv")

house_df = house_df[["Longitude", "Latitude"]]

center_x=house_df["Longitude"].mean()
center_y=house_df["Latitude"].mean()


map_sig=folium.Map(location = [42.034, -93.642],
                  zoom_start = 12,
                  tiles="cartodbpositron")

marker_cluster = MarkerCluster().add_to(map_sig)


# 택1
for idx, row in house_df.iterrows():
    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        popup="집들,,"
    ).add_to(marker_cluster)

# 택2
for i in range(len(house_df)):
    folium.Marker(
        location=[house_df.iloc[i,1], house_df.iloc[i,0]],
        popup="houses,,"
    ).add_to(marker_cluster)


map_sig.save("map_ames.html")












