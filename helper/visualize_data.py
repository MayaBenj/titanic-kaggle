import matplotlib.pyplot as plt
import pandas as pd


def plot(kind, df, x=None, y=None):
    df.plot(kind=kind, x=x, y=y)
    plt.show()


def plot_bins(kind, df, bins=None):
    df.plot(kind=kind, bins=bins)
    plt.show()


df = pd.read_csv("../data_csv/train.csv")
df = df.reset_index()

print(df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False)
      .mean().sort_values(by='Survived', ascending=False))
print(df[["Sex", "Survived"]].groupby(['Sex'], as_index=False)
      .mean().sort_values(by='Survived', ascending=False))
print(df[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False)
      .mean().sort_values(by='Survived', ascending=False))

plot("bar", df.groupby("Pclass")["index"].nunique())
plot("bar", df.groupby("Sex")["index"].nunique())
plot_bins("hist", df[["Age"]], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80])
plot("bar", df.groupby("SibSp")["index"].nunique())
plot("bar", df.groupby("Parch")["index"].nunique())
plot_bins("hist", df[["Fare"]], bins=[0, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600])

