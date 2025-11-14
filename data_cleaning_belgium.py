import pandas as pd

df_import=pd.read_csv("lap_times_2025_round_13.csv") #import the dataset

df_import["Lap Time"] = pd.to_timedelta(df_import["Lap Time"])
index = df_import.groupby("Driver")["Lap Time"].idxmin()#keep the best time for each driver
df=df_import.loc[index].reset_index(drop=True)

list_column=["Driver","Lap Time","Sector 1","Sector 2","Sector 3"]
df=df[list_column] #keep only the columns in the list behind

df["Lap Time"] = df["Lap Time"].astype(str).str.replace("0 days 00:", "", regex=False) #keep only minutes, seconds and hundredths of seconds
df["Sector 1"]=df["Sector 1"].astype(str).str.replace("0 days 00:", "", regex=False)
df["Sector 2"]=df["Sector 1"].astype(str).str.replace("0 days 00:", "", regex=False)
df["Sector 3"]=df["Sector 1"].astype(str).str.replace("0 days 00:", "", regex=False)

df.to_csv("lap_times_clean.csv", index=False)
