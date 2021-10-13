import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import scipy.stats as st
import seaborn as sns


def exploration_forme(dataframe):
    df = dataframe.copy()
    
    display(df.head(5))
    
    print("Nombre de lignes :", df.shape[0])
    print("Nombre de colonnes :", df.shape[1], "\n")
    
    for i in range(len(df.dtypes.value_counts())):
        print("Nombre de variables de type", df.dtypes.value_counts().index[i], ":", df.dtypes.value_counts()[i])
    
    print(" ")
    print("Pourcentage de valeurs manquantes par variable")
    print((df.isna().sum()/df.shape[0]).sort_values(ascending=True))
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.isna(), cbar=False)
    
    plt.show()
    

    years = df["year_survey"].sort_values(ascending=True).unique().tolist()
    print("Année des données :", years)

    countries = df["country"].unique().tolist()
    print("Nombre de pays :", len(countries))

    return years, countries


def check_valeurs(table, colonnes):
    
    test = [booleen for booleen in table["is_genuine"] if re.search(r"^(True|False)$", str(booleen)) is None]
    print("Colonne", str(colonnes[0]), "\n")
    print("Valeurs qui ne suivent pas le formatage défini :\n")
    print(pd.unique(test), "\n")
    print("Nombre d'occurence par valeur :\n")
    print(pd.DataFrame(test).value_counts(), "\n")
    print("Nombre de valeur total dont le formatage est différent :", len(test))
    print("-"*100)
    
    for i in range(len(table.columns)-1):
        test = [number for number in table[table.columns[i+1]] if re.search(r"^\d+.\d+$", str(number)) is None]
        print("Colonne", str(colonnes[i+1]), "\n")
        print("Valeurs qui ne suivent pas le formatage défini :\n")
        print(pd.unique(test), "\n")
        print("Nombre d'occurence par valeur :\n")
        print(pd.DataFrame(test).value_counts(), "\n")
        print("Nombre de valeur total dont le formatage est différent :", len(test), "\n")
        print("-"*100)


# =========================================================================================================================
# Fonctions mission 3
# =========================================================================================================================
def generate_incomes(n, pj):
    # Log du revenus des parents selon une loi normale
    ln_y_parent = st.norm(0, 1).rvs(size=n)
    
    # Terme d'erreur epsilon
    residues = st.norm(0, 1).rvs(size=n)
    
    # Revenus des enfants et des parents
    return np.exp(pj*ln_y_parent + residues), np.exp(ln_y_parent)


def quantiles(var, nb_quantiles):
    # Nombre d'individus
    size = len(var)
    
    # Tri et association des quantiles aux valeurs
    var_sorted = var.copy()
    var_sorted = var_sorted.sort_values()
    quantiles = np.round(np.arange(1, nb_quantiles + 1,
                                   nb_quantiles / size) - 0.5 + 1. / size)
    q_dict = {a: int(b) for a, b in zip(var_sorted, quantiles)}
    
    # Série contenant les classes de revenus
    return pd.Series([q_dict[e] for e in var])


def compute_quantiles(y_child, y_parents, nb_quantiles):
    # Array --> Serie
    y_child = pd.Series(y_child)
    y_parents = pd.Series(y_parents)
    
    # Calcul des quantiles
    c_i_child = quantiles(y_child, nb_quantiles)
    c_i_parent = quantiles(y_parents, nb_quantiles)
    
    # Concaténation des variables
    sample = pd.concat([y_child, y_parents, c_i_child, c_i_parent], axis=1)
    sample.columns = ["y_child", "y_parents", "c_i_child", "c_i_parent"]
    return sample


def distribution(counts, nb_quantiles):
    distrib = []
    total = counts["counts"].sum()  # Nombre d'individus du sous-ensemble

    if total == 0 :
        return [0] * nb_quantiles

    for q_parent in range(1, nb_quantiles + 1):
        # Sous-ensemble pour un c_i_parent donné (et un c_i_child donné)
        subset = counts[counts['c_i_parent'] == q_parent]
        
        # Calcul de la probabilité conditionnelle
        if len(subset):
            nb = subset["counts"].values[0]
            distrib += [nb / total]
        else:
            distrib += [0]
    return distrib


def conditional_distributions(sample, nb_quantiles):
    # Nombre d'individus par association c_i_child / c_i_parent
    counts = sample.groupby(["c_i_child", "c_i_parent"]).apply(len)
    counts = counts.reset_index()
    counts.columns = ["c_i_child", "c_i_parent", "counts"]

    # Matrice des probabilités conditionnelles
    mat = []
    for q_child in np.arange(nb_quantiles) + 1:
        # Sous-ensemble pour un c_i_child donné
        subset = counts[counts['c_i_child'] == q_child]
        mat += [distribution(subset, nb_quantiles)]
    return np.array(mat)


def plot_conditional_distributions(p, cd, nb_quantiles, save):
    plt.figure(figsize=(8, 6))

    # La ligne suivante sert à afficher un graphique en "stack bars", sur ce modèle : 
    # https://matplotlib.org/gallery/lines_bars_and_markers/bar_stacked.html
    cumul = np.array([0] * nb_quantiles)

    for i, c_quantile in enumerate(cd):
        plt.bar(np.arange(nb_quantiles) + 1,
                c_quantile,
                bottom=cumul,
                width=0.95,
                label = str(i+1) +"e")
        
        cumul = cumul + np.array(c_quantile)
        
    sns.despine()

    plt.axis([.5, nb_quantiles*1.3 ,0 ,1])
    plt.title("p=" + str(p))
    plt.legend()
    plt.xlabel("Quantile parents")
    plt.ylabel("Probabilité du quantile enfant")
    
    plt.savefig(save + f"\prob_cond_{str(p).replace('.', '')}", bbox_inches='tight')
    
    plt.show()


# =========================================================================================================================
# Fonctions mission 4
# =========================================================================================================================
def anova_table(aov):
    aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']

    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])

    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))\
        / (sum(aov['sum_sq'])+aov['mean_sq'][-1])

    cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
    aov = aov[cols]
    return aov
