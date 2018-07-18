import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math as mth
import matplotlib
import scipy.stats as st


cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma']),
         ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]

def verif_presence_nan_in_df(data, namedf=''):
    """Determine si le dataframe contient des valeurs nulles
    Parameters
    ----------
    data : DataFrame
        The first parameter.
    namedf : str
        The second parameter.

    Returns
    -------
    bool
        True if successful, False otherwise.
    
    """

    if data.isnull().values.any() == False:
        print("Il n'y a pas de valeur manquante dans {}".format(namedf))
    else:
        print('Présence de valeurs manquante dans {}'.format(namedf))

def verif_doublon(data, namedf=''):
    """Vérifie la présence de doublons
    Parameters
    ----------
    data : DataFrame
        The first parameter.
    namedf : str
        The second parameter.

    Returns
    -------
    bool
        True if successful, False otherwise.
    
    """
    data_ss_doublon = data.drop_duplicates()
    if data.shape == data_ss_doublon.shape:
        print("Absence de doublon, il n'y a pas de retraitement à faire pour {}".format(namedf))
    else:
        print('Suppression des doublons ?')

def analyse_univarie(data,moncaract='',typecaract=''):
    """Construit une analyse univarie selon le type de variable
    - Distribution empirique 
    - Représentation 
    - Mesure de tendance centrale 
    - Mesure de dispersion 
    - Mesure de concentration (cas continue)
    
    Parameters
    ----------
    data : DataFrame
        The first parameter.
    moncaract : str
        The second parameter : Nom de la colonne (variable) à traiter.
    typecaract :
        The third parameter : Type de variable.
    Returns
    -------
        Analyse univariée
    """

    data = data
    moncaract = moncaract
    typecaract = typecaract
    if typecaract=='qtedisc': # Variable quantitative discrète
        analyse_vqtedisc(data,moncaract,typecaract)
    elif typecaract=='qtecont': # Variable quantitative continue
        analyse_vqtecont(data,moncaract,typecaract)
    elif typecaract=='qual':
        analyse_qual(data,moncaract,typecaract)
    else:
        print("""Erreur : Insérer un type de variable parmi les choix suivants :
              - Variable quantitative discrète 'qtedisc' 
              - Variable quantitative continue 'qtecont' 
              - Variable quantitative 'qtequal'""")
def analyse_bivariee(data,nomcaract1='',typecaract1='',nomcaract2='',typecaract2=''):
    """Construit une analyse bivarie selon le type de variable
    - Distribution empirique 
    - Représentation 
    - Mesure de tendance centrale 
    - Mesure de dispersion 
    - Mesure de concentration (cas continue)
    
    Parameters
    ----------
    data : DataFrame
        The first parameter.
    moncaract1 : str
        The second parameter : Nom de la colonne (variable) à traiter.
    moncaract2 : str
        The second parameter : Nom de la colonne (variable) à traiter.
    typecaract1 :
        The third parameter : Type de la première variable.
    typecaract2 :
        The third parameter : Type de la seconde variable.
    Returns
    -------
        Analyse bivariée
    """
    data=data
    if typecaract1 == 'qte' and typecaract2 == 'qte':
        analyse_qte_qte(data,nomcaract1,typecaract1,nomcaract2,typecaract2)
        
    elif typecaract1 == 'qual' and typecaract2 =='qual':
        analyse_qual_qual(data,nomcaract1,typecaract1,nomcaract2,typecaract2)
    elif (typecaract1 == 'qte' and typecaract2 == 'qual') or (typecaract1 == 'qual' and typecaract2 == 'qte'):
        analyse_qte_qual(data,nomcaract1,typecaract1,nomcaract2,typecaract2)
    else:
        print("""Qualitative = 'qual'
        Quantitative = 'qte'""")
        
def analyse_vqtedisc(data,moncaract,typecaract):
 #Construction du tableau de distribution
        effectifs = data[moncaract].value_counts()

        modalites = effectifs.index # l'index de effectifs contient les modalités
        tab = pd.DataFrame(modalites, columns = [moncaract]) #création du tableau à partir des modalités

        tab["n"] = effectifs.values

        tab["f"] = tab["n"] / len(data) # len(data) renvoie la taille de l'échantillon

        tab = tab.sort_values(moncaract) # tri des valeurs de la variable X (croissant)
        tab["F"] = tab["f"].cumsum() # cumsum calcule la somme cumulée
        
        moyenne = data[moncaract].mean()
        mediane = data[moncaract].median()
        mode = data[moncaract].mode()
        
        mesure_tendance_centrale = """Variable {} :
        - Moyenne = {}
        - Médiane = {}
        - Mode = {}""".format(moncaract,moyenne,mediane,mode)
        
        variance = data[moncaract].var(ddof=0)
        ecart_type = data[moncaract].std(ddof=0)
        mesure_dispersion = """Variable {} :
        - Variance = {}
        - Ecart-type = {}""".format(moncaract,variance,ecart_type)
        
        #Représentation
        representation = str(input("Choisir représentation : Diagramme en bâtons ('diagbat') ou Courbe cumulative ('courbcum') :"))
        
        if representation == 'diagbat':
            abcisses = moncaract
            question = str(input('Voulez-vous afficher les couleurs ? (y/n)'))
            if question == 'y':
                print(cmaps)
            else:
                print('affichage des couleurs non demandé')
            choix_couleur = str(input('Choisir couleur du graphique :'))
            if not(choix_couleur):
                choix_couleur=None
            tab_plot = tab.plot(x=abcisses,y='n',kind='bar', figsize=(20,10), colormap=choix_couleur)
            titre = str(input("Donner le nom du titre du graphique :"))
            plt.title(titre)
            legend = str(input("Donner le nom de la légende du graphique :"))
            plt.legend(title=legend, loc='upper left')
            
            ylabel = str(input("Donner le nom de l'axe des ordonnés du graphique :"))
            if not(ylabel):
                plt.ylabel('Effectifs')
            else:
                plt.ylabel(ylabel)
        
            xlabel = str(input("Donner le nom de l'axe des abcisses du graphique :"))
            if not(xlabel):
                plt.xlabel(moncaract)
            else:
                plt.xlabel(xlabel)
            
            plt.show(tab_plot)
            save_image = str(input("Sauvegarder l'image ? (y/n) :"))
            if save_image =='y': 
                image= tab_plot.get_figure()
                chemin = str(input('Indiquer le chemin du dossier'))
                image.savefig('{}/{}'.format(chemin,titre))
            else:
                print("Pas de sauvegarde")
            print(mesure_tendance_centrale)
            print(mesure_dispersion)
            print(tab)
        else:
            abcisses = moncaract
            question = str(input('Voulez-vous afficher les couleurs ? (y/n)'))
            if question == 'y':
                print(cmaps)
            else:
                print('affichage des couleurs non demandé')
            choix_couleur = str(input('Choisir couleur du graphique :'))
            if not(choix_couleur):
                choix_couleur=None
            tab_plot = tab.plot(x=abcisses,y='F',kind='line', figsize=(20,10), colormap=choix_couleur)
            titre = str(input("Donner le nom du titre du graphique :"))
            plt.title(titre)
            legend = str(input("Donner le nom de la légende du graphique :"))
            plt.legend(title=legend, loc='upper left')
            
            ylabel = str(input("Donner le nom de l'axe des ordonnés du graphique :"))
            if not(ylabel):
                plt.ylabel('Fréquences cumulées')
            else:
                plt.ylabel(ylabel)
        
            xlabel = str(input("Donner le nom de l'axe des abcisses du graphique :"))
            if not(xlabel):
                plt.xlabel(moncaract)
            else:
                plt.xlabel(xlabel)
            
            
            
            plt.show(tab_plot)
            save_image = str(input("Sauvegarder l'image ? (y/n) :"))
            if save_image == 'y': 
                image= tab_plot.get_figure()
                chemin = str(input('Indiquer le chemin du dossier'))
                image.savefig('{}/{}'.format(chemin,titre))
            else:
                print("Pas de sauvegarde")
            print(mesure_tendance_centrale)
            print(mesure_dispersion)
            print(tab)
        
def analyse_vqtecont(data,moncaract,typecaract):
#Construction du tableau de distribution
        
        effectifs = data[moncaract].value_counts()

        modalites = effectifs.index # l'index de effectifs contient les modalités
        tab = pd.DataFrame(modalites, columns = [moncaract]) #création du tableau à partir des modalités

        tab["n"] = effectifs.values

        tab["f"] = tab["n"] / len(data) # len(data) renvoie la taille de l'échantillon
        
        

        tab = tab.sort_values(moncaract) # tri des valeurs de la variable X (croissant)
        tab["F"] = tab["f"].cumsum() # cumsum calcule la somme cumulée
        
        moyenne = data[moncaract].mean()
        mediane = data[moncaract].median()
        mode = data[moncaract].mode()
        
        mesure_tendance_centrale = """Variable {} :
        - Moyenne = {}
        - Médiane = {}
        """.format(moncaract,moyenne,mediane,mode)
        
        variance = data[moncaract].var(ddof=0)
        ecart_type = data[moncaract].std(ddof=0)
        mesure_dispersion = """Variable {} :
        - Variance = {}
        - Ecart-type = {}""".format(moncaract,variance,ecart_type)
       
    
        question = str(input("Voulez-vous afficher la courbe de lorenz et l'indice de gini ? (y/n)"))
        if question == 'y':
            #Mesure de concentration

            echantillon = data[moncaract]
            #Sélection du sous-échantillon de travail que l'on appelle  revenus
            ech = echantillon.values
            #On place les observations dans une variable
            lorenz = np.cumsum(np.sort(ech)) / ech.sum()

            lorenz = np.append([0],lorenz) # La courbe de Lorenz commence à 0

            plot_lorenz = plt.figure()
            question = str(input('Voulez-vous afficher les couleurs ? (y/n)'))
            if question == 'y':
                print(matplotlib.colors.cnames.items())
            else:
                print('affichage des couleurs non demandé')
            choix_couleur = str(input('Choisir couleur du graphique :'))
            if not(choix_couleur):
                choix_couleur=None
            plt.plot(np.linspace(0,1,len(lorenz)),lorenz,drawstyle='steps-post',color=choix_couleur)

            titre_lorenz = 'Courbe de Lorenz'
            plt.title(titre_lorenz)
            ylabel_lorenz = 'F(N,x)'
            plt.ylabel(ylabel_lorenz)
            xlabel_lorenz = 'F(x)'
            plt.xlabel(xlabel_lorenz)


            #Indice de Gini
            aire_ss_courbe = lorenz[:-1].sum()/len(lorenz) # aire sous la courbe de Lorenz. La dernière valeur ne participe pas à l'aire, d'où "[:-1]"
            S = 0.5 - aire_ss_courbe # aire entre la 1e bissectrice et la courbe de Lorenz
            gini = 2*S

            plt.show(plot_lorenz)
            save_image_lorenz = str(input("Sauvegarder la courbe de lorenz ? (y/n) :"))
            if save_image_lorenz =='y': 
                image_lorenz= plot_lorenz.get_figure()
                chemin = str(input('Indiquer le chemin du dossier'))
                image.savefig('{}/{}'.format(chemin,titre))
               
            print("L'indice de Gini est égal à {}".format(gini))
        else:
            print("Mesure de concentration non affichée")
        #indice de Huntsberger : Pour connaitre le nombre ideal de classes pour la distribution

        #N(cl) = 1 + 3,3 log10(N)
        #N = nombre d’observations
        #N(cl) = nombre de classes
        N= data[moncaract].values.sum()
        nombre_classe = 1 + 3.3*mth.log10(N)
        nombre_classe = round(nombre_classe)
        tab[moncaract]= pd.cut(tab[moncaract],bins=nombre_classe)
        
        #Représentation
        representation = str(input("Choisir représentation : Histogramme ('hist') ou BoxPlot ('boxplot') :"))
        if representation == 'boxplot':
            abcisses = moncaract
            outliers = str(input("Afficher les outliers (y/n) :"))
            if outliers == 'y':
                outliers = True
            plot_data = data.boxplot(column=abcisses,vert=False, showfliers=outliers)
            titre = str(input("Donner le nom du titre du graphique :"))
            plt.title(titre)
            
        
            xlabel = str(input("Donner le nom de l'axe des abcisses du graphique :"))
            if not(xlabel):
                plt.xlabel(moncaract)
            else:
                plt.xlabel(xlabel)
            
            
            plt.show(plot_data)
            save_image = str(input("Sauvegarder l'image ? (y/n) :"))
            if save_image == 'y': 
                image= plot_data.get_figure()
                chemin = str(input('Indiquer le chemin du dossier'))
                image.savefig('{}/{}'.format(chemin,titre))
            else:
                print("Pas de sauvegarde")
            upper_quartile = np.percentile(data[moncaract], 75)
            lower_quartile = np.percentile(data[moncaract], 25)

            iqr = upper_quartile - lower_quartile
            upper_whisker = data[moncaract] [data[moncaract]<=upper_quartile+1.5*iqr].max()
            lower_whisker = data[moncaract] [data[moncaract]>=lower_quartile-1.5*iqr].min()
            print("""La mediane est {}, Q1 est égal à {} et Q3 est égal à {} 
            L'écart inter-quartile est égal à {} et les bornes sont respectivement de {} à {}""".format(mediane, lower_quartile,upper_quartile,iqr,lower_whisker,upper_whisker))
            print(mesure_tendance_centrale)
            print(mesure_dispersion)
            
            print(tab)
        
        if representation == 'hist':
            abcisses = moncaract
            question = str(input('Voulez-vous afficher les couleurs ? (y/n)'))
            if question == 'y':
                print(matplotlib.colors.cnames.items())
            else:
                print('affichage des couleurs non demandé')
            choix_couleur = str(input('Choisir couleur du graphique :'))
            if not(choix_couleur):
                choix_couleur=None
            
            plot_data = data[moncaract].hist(density=True, bins=nombre_classe,color=choix_couleur)
            titre = str(input("Donner le nom du titre du graphique :"))
            plt.title(titre)
            
            ylabel = str(input("Donner le nom de l'axe des ordonnés du graphique :"))
            if not(ylabel):
                plt.ylabel('Fréquences')
            else:
                plt.ylabel(ylabel)
        
            xlabel = str(input("Donner le nom de l'axe des abcisses du graphique :"))
            if not(xlabel):
                plt.xlabel(moncaract)
            else:
                plt.xlabel(xlabel)
            
            
            
            plt.show(plot_data)
            save_image = str(input("Sauvegarder l'image ? (y/n) :"))
            if save_image == 'y': 
                image= plot_data.get_figure()
                chemin = str(input('Indiquer le chemin du dossier'))
                image.savefig('{}/{}'.format(chemin,titre))
            else:
                print("Pas de sauvegarde")
            
            print(mesure_tendance_centrale)
            print(mesure_dispersion)
            
            print(tab)
        
def analyse_qual(data,moncaract,typecaract):
#Construction du tableau de distribution
        effectifs = data[moncaract].value_counts()

        modalites = effectifs.index # l'index de effectifs contient les modalités
        tab = pd.DataFrame(modalites, columns = [moncaract]) #création du tableau à partir des modalités

        tab["n"] = effectifs.values

        tab["f"] = tab["n"] / len(data) # len(data) renvoie la taille de l'échantillon

        
        mode = data[moncaract].mode()
        
        mesure_tendance_centrale = """Variable {} :
        
        - Mode = {}""".format(moncaract,mode)
        
        

        #Représentation
        representation = str(input("Choisir représentation : Camenbert ('camenb') ou Tuyau d'orgue ('tuyau') :"))
        
        if representation == 'camenb':
            abcisses = moncaract
            question = str(input('Voulez-vous afficher les couleurs ? (y/n)'))
            if question == 'y':
                print(cmaps)
            else:
                print('affichage des couleurs non demandé')
            choix_couleur = str(input('Choisir couleur du graphique :'))
            if not(choix_couleur):
                choix_couleur=None
            tab_plot = tab['f'].plot(kind='pie',autopct = lambda x: str(round(x, 2)) + '%', colormap=choix_couleur)
            plt.axis('equal')
            titre = str(input("Donner le nom du titre du graphique :"))
            plt.title(titre)
            
            ylabel = str(input("Donner le nom de l'axe des ordonnés du graphique :"))
            if not(ylabel):
                plt.ylabel(moncaract)
            else:
                plt.ylabel(ylabel)
        
           
            
            plt.show(tab_plot)
            save_image = str(input("Sauvegarder l'image ? (y/n) :"))
            if save_image =='y': 
                image= tab_plot.get_figure()
                chemin = str(input('Indiquer le chemin du dossier'))
                image.savefig('{}/{}'.format(chemin,titre))
            else:
                print("Pas de sauvegarde")
            print(mesure_tendance_centrale)
            print(mesure_dispersion)
            print(tab)
        elif representation == 'tuyau':
            abcisses = moncaract
            question = str(input('Voulez-vous afficher les couleurs ? (y/n)'))
            if question == 'y':
                print(cmaps)
            else:
                print('affichage des couleurs non demandé')
            choix_couleur = str(input('Choisir couleur du graphique :'))
            if not(choix_couleur):
                choix_couleur=None
            tab_plot = tab.plot(x=abcisses,y='n',kind='bar', figsize=(20,10), colormap=choix_couleur)
            titre = str(input("Donner le nom du titre du graphique :"))
            plt.title(titre)
            legend = str(input("Donner le nom de la légende du graphique :"))
            plt.legend(title=legend, loc='upper left')
            
            ylabel = str(input("Donner le nom de l'axe des ordonnés du graphique :"))
            if not(ylabel):
                plt.ylabel('Effectifs')
            else:
                plt.ylabel(ylabel)
        
            xlabel = str(input("Donner le nom de l'axe des abcisses du graphique :"))
            if not(xlabel):
                plt.xlabel(moncaract)
            else:
                plt.xlabel(xlabel)
                
            plt.show(tab_plot)
            save_image = str(input("Sauvegarder l'image ? (y/n) :"))
            if save_image == 'y': 
                image= tab_plot.get_figure()
                chemin = str(input('Indiquer le chemin du dossier'))
                image.savefig('{}/{}'.format(chemin,titre))
            else:
                print("Pas de sauvegarde")
            print(mesure_tendance_centrale)
            print(mesure_dispersion)
            print(tab)
        
def analyse_qte_qte(data,nomcaract1,typecaract1,nomcaract2,typecaract2):
       
        abcisses = nomcaract1
        ordonnes = nomcaract2
        question = str(input('Voulez-vous afficher les couleurs ? (y/n)'))
        if question == 'y':
            print(matplotlib.colors.cnames.items())
        else:
            print('affichage des couleurs non demandé')
        choix_couleur = str(input('Choisir couleur du graphique :'))
        if not(choix_couleur):
            choix_couleur=None
        data_plot = data.plot(x=abcisses,y=ordonnes,kind='scatter', figsize=(20,10), color=choix_couleur)
        titre = str(input("Donner le nom du titre du graphique :"))
        plt.title(titre)
        legend = str(input("Donner le nom de la légende du graphique :"))
        plt.legend(title=legend, loc='upper left')
        
        ylabel = str(input("Donner le nom de l'axe des ordonnés du graphique :"))
        if not(ylabel):
            plt.ylabel(nomcaract2)
        else:
            plt.ylabel(ylabel)
        
        xlabel = str(input("Donner le nom de l'axe des abcisses du graphique :"))
        if not(xlabel):
            plt.xlabel(nomcaract1)
        else:
            plt.xlabel(xlabel)
        
        plt.show(data_plot)
        save_image = str(input("Sauvegarder l'image ? (y/n) :"))
        if save_image =='y': 
            image= data_plot.get_figure()
            chemin = str(input('Indiquer le chemin du dossier'))
            image.savefig('{}/{}'.format(chemin,titre))
        else:
            print("Pas de sauvegarde")
        
        # Analyse de la corrélation
        coef_corr_pearson = round(st.pearsonr(data[nomcaract1],data[nomcaract2])[0],2)
        print("Le coefficient de corrélation (Pearson) est égal à {}".format(coef_corr_pearson))
        if coef_corr_pearson <= 0 and coef_corr_pearson > -0.40:
            print('Les variables ne sont pas négativement corrélées car {} est supérieur à -0,40'.format(coef_corr_pearson))
        elif coef_corr_pearson < -0.60:
            print('Les variables sont négativement corrélées car {} est inférieur à -0,60'.format(coef_corr_pearson))
        elif coef_corr_pearson >= 0 and coef_corr_pearson < 0.40:
            print('Les variables ne sont pas positivement corrélées car {} est inférieur à 0,40'.format(coef_corr_pearson))
        elif coef_corr_pearson > 0.60:
            print('Les variables sont corrélées positivement car {} est supérieur à 0,60'.format(coef_corr_pearson))
        else:
            seuil_confiance = float(input('Choisir un seuil de confiance 0.1 ou 0.05 :'))
            p_value = round(st.pearsonr(data[nomcaract1],data[nomcaract2])[1],2)
            if p_value < seuil_confiance:
                print('On retient H1 : Les variables sont corrélées car {} (p-valeur) est inférieure à {} (seuil de confiance)'.format(p_value,seuil_confiance))
            else:
                print('On retient H0 : Les variables ne sont pas corrélées car {} (p-valeur) est supérieure à {} (seuil de confiance)'.format(p_value,seuil_confiance))
def analyse_qte_qual(data,nomcaract1,typecaract1,nomcaract2,typecaract2):
 # Représentation
        fig= plt.figure(figsize=(20,20))
        abcisses = nomcaract1
        ordonnes = str(input('Choisir variable de mesure (qte) :'))
        teinte= nomcaract2
        color_palette_names = ['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind']
        question = str(input('Voulez-vous afficher les couleurs ? (y/n)'))
        if question == 'y':
            print(color_palette_names)
        else:
            print('affichage des couleurs non demandé')
        choix_couleur = str(input('Choisir couleur du graphique :'))
        if not(choix_couleur):
            choix_couleur=None

        data_plot = sns.barplot(x=abcisses, y=ordonnes,hue=teinte, data=data, palette=sns.color_palette(choix_couleur,2))
        titre = str(input('Choisissez le titre du graphique'))
        data_plot.set_title(titre)
       
        ylabel = str(input("Donner le nom de l'axe des ordonnés du graphique :"))
        if not(ylabel):
            data_plot.set_ylabel(nomcaract2)
        else:
            data_plot.set_ylabel(ylabel)
        
        xlabel = str(input("Donner le nom de l'axe des abcisses du graphique :"))
        if not(xlabel):
            data_plot.set_xlabel(nomcaract1)
        else:
            data_plot.set_xlabel(xlabel)
        plt.show(fig)
        save_image = str(input("Sauvegarder l'image ? (y/n) :"))
        if save_image =='y': 
            image= fig.get_figure()
            chemin = str(input('Indiquer le chemin du dossier'))
            image.savefig('{}/{}'.format(chemin,titre))
        else:
            print("Pas de sauvegarde")
        
        # Analyse de la corrélation
        x= data[nomcaract1]
        y= data[nomcaract2]
        
        moyenne_y = y.mean()
        classes = []
        for classe in x.unique():
            yi_classe = y[x==classe]
            classes.append({'ni': len(yi_classe),
                            'moyenne_classe': yi_classe.mean()})
        SCT = sum([(yj-moyenne_y)**2 for yj in y])
        SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
        eta_squared= SCE/SCT
        print("Le coefficient de corrélation (eta-squared) est égal à {}".format(eta_squared))
        if eta_squared <=0 and eta_squared > -0.40:
            print('Les variables ne sont pas négativement corrélées car {} est supérieur à -0,40'.format(eta_squared))
        elif eta_squared < -0.60:
            print('Les variables sont négativement corrélées car {} est inférieur à -0,60'.format(eta_squared))
        elif eta_squared >= 0 and eta_squared < 0.40:
            print('Les variables ne sont pas positivement corrélées car {} est inférieur à 0,40'.format(eta_squared))
        elif eta_squared > 0.60:
            print('Les variables sont positivement corrélées car {} est supérieur à 0,60'.format(eta_squared))
        else:
            seuil_confiance = float(input('Choisir un seuil de confiance 0.1 ou 0.05 :'))
            p_value = round(st.pearsonr(data[nomcaract1],data[nomcaract2])[1],2)
            if p_value < seuil_confiance:
                print('On retient H1 : Les variables sont corrélées car {} (p-valeur) est inférieure à {} (seuil de confiance)'.format(p_value,seuil_confiance))
            else:
                print('On retient H0 : Les variables ne sont pas corrélées car {} (p-valeur) est supérieure à {} (seuil de confiance)'.format(p_value,seuil_confiance))
    
def analyse_qual_qual(data,nomcaract1,typecaract1,nomcaract2,typecaract2):
      
        X = nomcaract1

        Y = nomcaract2


        c = data[[X,Y]].pivot_table(index=X,columns=Y,aggfunc=len)

        cont = c.copy()


        tx = data[X].value_counts()

        ty = data[Y].value_counts()


        cont.loc[:,"Total"] = tx

        cont.loc["total",:] = ty

        cont.loc["total","Total"] = len(data)



        tx = pd.DataFrame(tx)


        ty = pd.DataFrame(ty)

        tx.columns = ["foo"]

        ty.columns = ["foo"]

        n = len(data)

        indep = (tx.dot(ty.T) / n)


        c = c.fillna(0) # on remplace les valeurs nulles par des 0

        mesure = (c-indep)**2/indep

        xi_n = mesure.sum().sum()

        d = (mesure/xi_n)
        fig = plt.figure(figsize=(20,20))
        plot = sns.heatmap(d, annot=c)
        titre = plot.set_title('Tableau de contingence coloré')
        plt.show(plot)
        
        save_image = str(input("Sauvegarder l'image ? (y/n) :"))
        if save_image =='y': 
            image= fig.get_figure()
            chemin = str(input('Indiquer le chemin du dossier'))
            image.savefig('{}/{}'.format(chemin,titre))
        else:
            print("Pas de sauvegarde")
        
        print('Table des coefficients de corrélation :')
        print(xi_n)
