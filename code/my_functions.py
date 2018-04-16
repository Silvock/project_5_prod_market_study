import os
if __name__=="__main__":
    def analyse_univarie(data,moncaract='',typecaract=''):
        """Construit une analyse univarie selon le type de variable
        - Distribution empirique 
        - Représentation 
        - Mesure de tendance centrale 
        - Mesure de dispersion 
        - Mesure de concentration (cas continue)"""

        if typecaract=='qtedisc': # Variable quantitative discrète
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
                abcisses = str(input("Donner le nom de la variable x :"))
                tab_plot = tab.plot(x=abcisses,y='n',kind='bar', figsize=(20,10), colormap='copper')
                titre = str(input("Donner le nom du titre du graphique :"))
                plt.title(titre)
                legend = str(input("Donner le nom de la légende du graphique :"))
                plt.legend(title=legend, loc='upper left')
                ylabel = str(input("Donner le nom de l'axe des ordonnés du graphique :"))
                plt.ylabel(ylabel)
                xlabel = str(input("Donner le nom de l'axe des abcisses du graphique :"))
                plt.xlabel(xlabel)
                plt.show(tab_plot)
                save_image = str(input("Sauvegarder l'image ? (y/n) :"))
                if save_image =='y': 
                    image= tab_plot.get_figure()
                    image.savefig('Images/{}'.format(titre))
                else:
                    print("Pas de sauvegarde")
                print(mesure_tendance_centrale)
                print(mesure_dispersion)
                return tab
            else:
                abcisses = str(input("Donner le nom de la variable x :"))
                tab_plot = tab.plot(x=abcisses,y='F',kind='line', figsize=(20,10), colormap='copper')
                titre = str(input("Donner le nom du titre du graphique :"))
                plt.title(titre)
                legend = str(input("Donner le nom de la légende du graphique :"))
                plt.legend(title=legend, loc='upper left')
                ylabel = str(input("Donner le nom de l'axe des ordonnés du graphique :"))
                plt.ylabel(ylabel)
                xlabel = str(input("Donner le nom de l'axe des abcisses du graphique :"))
                plt.xlabel(xlabel)
                plt.show(tab_plot)
                save_image = str(input("Sauvegarder l'image ? (y/n) :"))
                if save_image == 'y': 
                    image= tab_plot.get_figure()
                    image.savefig('Images/{}'.format(titre))
                else:
                    print("Pas de sauvegarde")
                print(mesure_tendance_centrale)
                print(mesure_dispersion)
                return tab

        elif typecaract=='qtecont': # Variable quantitative continue
            #Construction du tableau de distribution

            effectifs = data[moncaract].value_counts()

            modalites = effectifs.index # l'index de effectifs contient les modalités
            tab = pd.DataFrame(modalites, columns = [moncaract]) #création du tableau à partir des modalités

            tab["n"] = effectifs.values

            tab["f"] = tab["n"] / len(data) # len(data) renvoie la taille de l'échantillon

            amplitude = int(input("Choisir l'amplitude des classes :"))
            tab[moncaract]= pd.cut(tab[moncaract],bins=20)

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

            #Mesure de concentration

            echantillon = data[moncaract]
            #Sélection du sous-échantillon de travail que l'on appelle  revenus
            ech = echantillon.values
            #On place les observations dans une variable
            lorenz = np.cumsum(np.sort(ech)) / ech.sum()

            lorenz = np.append([0],lorenz) # La courbe de Lorenz commence à 0

            plot_lorenz = plt.figure()
            plt.plot(np.linspace(0,1,len(lorenz)),lorenz,drawstyle='steps-post',color='rosybrown')

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
                image_lorenz.savefig('Images/{}'.format(titre_lorenz))
            print("L'indice de Gini est égal à {}".format(gini))

            #Représentation
            representation = str(input("Choisir représentation : Histogramme ('hist') ou BoxPlot ('boxplot') :"))
            if representation == 'boxplot':
                abcisses = str(input("Donner le nom de la variable x :"))
                outliers = bool(input("Afficher les outliers (Bool) :"))
                plot_data = data.boxplot(column=abcisses,vert=False, showfliers=outliers)
                titre = str(input("Donner le nom du titre du graphique :"))
                plt.title(titre)
                xlabel = str(input("Donner le nom de l'axe des abcisses du graphique :"))
                plt.xlabel(xlabel)
                plt.show(plot_data)
                save_image = str(input("Sauvegarder l'image ? (y/n) :"))
                if save_image == 'y': 
                    image= plot_data.get_figure()
                    image.savefig('Images/{}'.format(titre))
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

                return tab

            if representation == 'hist':
                abcisses = str(input("Donner le nom de la variable x :"))

                plot_data = data[moncaract].hist(density=True, bins=amplitude,color='rosybrown')
                titre = str(input("Donner le nom du titre du graphique :"))
                plt.title(titre)
                xlabel = str(input("Donner le nom de l'axe des abcisses du graphique :"))
                plt.xlabel(xlabel)
                ylabel = str(input("Donner le nom de l'axe des ordonnés du graphique :"))
                plt.ylabel(ylabel)
                plt.show(plot_data)
                save_image = str(input("Sauvegarder l'image ? (y/n) :"))
                if save_image == 'y': 
                    image= plot_data.get_figure()
                    image.savefig('Images/{}'.format(titre))
                else:
                    print("Pas de sauvegarde")

                print(mesure_tendance_centrale)
                print(mesure_dispersion)

                return tab
        elif typecaract=='qual':
            #Construction du tableau de distribution
            effectifs = data[moncaract].value_counts()

            modalites = effectifs.index # l'index de effectifs contient les modalités
            tab = pd.DataFrame(modalites, columns = [moncaract]) #création du tableau à partir des modalités

            tab["n"] = effectifs.values

            tab["f"] = tab["n"] / len(data) # len(data) renvoie la taille de l'échantillon

            #Représentation
            representation = str(input("Choisir représentation : Camenbert ('camenb') ou Tuyau d'orgue ('tuyau') :"))

            if representation == 'diagbat':
                abcisses = str(input("Donner le nom de la variable x :"))
                tab_plot = tab['f'].plot(kind='pie',autopct = lambda x: str(round(x, 2)) + '%')
                plt.axis('equal')
                titre = str(input("Donner le nom du titre du graphique :"))
                plt.title(titre)

                ylabel = str(input("Donner le nom de l'axe des ordonnés du graphique :"))
                plt.ylabel(ylabel)

                plt.show(tab_plot)
                save_image = str(input("Sauvegarder l'image ? (y/n) :"))
                if save_image =='y': 
                    image= tab_plot.get_figure()
                    image.savefig('Images/{}'.format(titre))
                else:
                    print("Pas de sauvegarde")
                print(mesure_tendance_centrale)
                print(mesure_dispersion)
                return tab
            else:
                abcisses = str(input("Donner le nom de la variable x :"))
                tab_plot = tab.plot(x=abcisses,y='n',kind='bar', figsize=(20,10), colormap='copper')
                titre = str(input("Donner le nom du titre du graphique :"))
                plt.title(titre)
                legend = str(input("Donner le nom de la légende du graphique :"))
                plt.legend(title=legend, loc='upper left')
                ylabel = str(input("Donner le nom de l'axe des ordonnés du graphique :"))
                plt.ylabel(ylabel)
                xlabel = str(input("Donner le nom de l'axe des abcisses du graphique :"))
                plt.xlabel(xlabel)
                plt.show(tab_plot)
                save_image = str(input("Sauvegarder l'image ? (y/n) :"))
                if save_image == 'y': 
                    image= tab_plot.get_figure()
                    image.savefig('Images/{}'.format(titre))
                else:
                    print("Pas de sauvegarde")
                print(mesure_tendance_centrale)
                print(mesure_dispersion)
                return tab

        else:
            print("""Erreur : Insérer un type de variable parmi les choix suivants :
                  - Variable quantitative discrète 'qtedisc' 
                  - Variable quantitative continue 'qtecont' 
                  - Variable quantitative 'qtequal'""")