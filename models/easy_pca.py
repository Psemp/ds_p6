import numpy as np
import pandas as pd
import seaborn as sns

from types import NoneType
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA


class Easy_pca():
    """
    Desc : Aims to facilitate and streamline most of the PCA use.
    Args :
    - dataset : the dataset as a pandas.DataFrame object on which to perform a PCA. All columns must be numeric
    except the target col which is used for the Hues of plots and can be categorical or numerical
    - taget_cal : the name of the column (str) targeted in case of clustering purpose, default is None
    """

    def __init__(self, dataset: pd.DataFrame, target_col: str = None) -> None:

        if target_col is None:
            self.data_pca = dataset
            self.features = self.data_pca.columns
            self.target_col = None
        elif isinstance(target_col, str):
            self.data_pca = dataset.drop(columns=[target_col])
            self.features = self.data_pca.columns
            self.target_col = target_col
        elif not isinstance(target_col, (str, NoneType)):
            raise TypeError(f"taget_col must be NoneType or str, not {type(target_col)}")

        scaler = preprocessing.StandardScaler()

        self.data_pca = self.data_pca.fillna(self.data_pca.mean())
        self.X_scaled = scaler.fit_transform(self.data_pca.values)
        self.pca = PCA(n_components=len(self.features))
        self.pca.fit(self.X_scaled)
        self.X_new = self.pca.transform(self.X_scaled)
        self.percentage_variation = np.round(self.pca.explained_variance_ratio_ * 100, decimals=1)

        pcs = self.X_new.T
        pcdict = {}
        component_nb = 1
        for pc in pcs:
            pcdict[f"PC{component_nb}"] = pc
            component_nb += 1

        self.pcframe = pd.DataFrame(pcdict)
        if target_col is not None:
            self.pcframe[target_col] = dataset[target_col]

    def display_circles(self, couple_pc: tuple, show: bool = True):
        """
        Function : Display pca of pcs couple_pc correlation circles for pca of dataframe
        in parameter.
        Original : https://github.com/AnisHdd
        Args:
        - couple_pc : tuple (x, y) as x and y indexes of principal components
        - show : bool, default is True, display(True) or return figure(False)
        Returns:
        - Void, displays plot by default, returns matplotlib.figure.Figure object if not show
        """
        x_pc = couple_pc[0]
        y_pc = couple_pc[1]

        fig, (ax1) = plt.subplots(
            ncols=1,
            nrows=1,
            figsize=(6, 6),
            dpi=150,
        )

        for i in range(0, self.pca.components_.shape[1]):
            ax1.arrow(
                0,
                0,  # origin
                self.pca.components_[x_pc, i],
                self.pca.components_[y_pc, i],
                head_width=0.1,
                head_length=0.1
            )

            plt.text(
                self.pca.components_[x_pc, i] + 0.05,
                self.pca.components_[y_pc, i] + 0.05,
                self.features[i]
            )

        an = np.linspace(0, 2 * np.pi, 100)
        plt.plot(np.cos(an), np.sin(an))  # Draw circle

        ###
        # Titles/Lables
        ax1.set_title(f"Correlation Circle : PC{x_pc + 1} / PC{y_pc + 1} ")
        ax1.set_xlabel(f"PC{x_pc + 1}")
        ax1.set_ylabel(f"PC{y_pc + 1}")
        #
        ###

        if show:
            plt.show()
        elif not show:
            return plt.gcf()

    def show_contribution(self, lim_pc: int = None) -> pd.DataFrame:
        """
        Takes the pca object, the list of target columns for the pca and the percentages of variation
        to return a pd.DataFrame object with details on the principal components
        Args :
        - lim_pc : Optionnal, default = None, the last Principal component (PC after PC_lim_pc will be ignored)
        Returns :
        - DataFrame with detailed characteristics of each PC
        """
        sers = {}
        contrib_dict = {}

        if lim_pc is None:

            for i in range(0, len(self.pca.components_)):
                sers[f"PC{i + 1}"] = pd.Series(self.pca.components_[i], index=self.features)
                contrib_dict[f"PC{i + 1}"] = pd.Series(self.percentage_variation[i], index=["Inertia"])

        elif lim_pc is not None:

            for i in range(0, lim_pc):
                sers[f"PC{i + 1}"] = pd.Series(self.pca.components_[i], index=self.features)
                contrib_dict[f"PC{i + 1}"] = pd.Series(self.percentage_variation[i], index=["Inertia"])

        components_df = pd.DataFrame(sers)
        self.temp = pd.DataFrame(contrib_dict)
        frames = [components_df, self.temp]
        components_df = pd.concat(frames)
        return components_df

    def get_scree_plot(self, show: bool = True):
        """
        Shows or returns the scree plot of the PCA.
        Args :
        - show : bool, default is True, display(True) or return figure(False)
        Returns :
        - Void, displays plot by default, returns matplotlib.figure.Figure object if not show
        """

        labels = ["PC" + str(component) for component in range(1, len(self.percentage_variation) + 1)]
        x_bars = np.arange(1, len(self.percentage_variation) + 1, 1)

        cummulative_percentage = np.cumsum(self.percentage_variation)

        fig, ax1 = plt.subplots(
            ncols=1,
            nrows=1,
            figsize=(6, 6),
            dpi=150,
        )

        ax1.bar(x_bars, height=self.percentage_variation)

        ax1.set_xticks(range(1, len(labels) + 1, 1))
        ax1.set_xticklabels(labels)
        ax1.plot(ax1.get_xticks(), cummulative_percentage, marker="o", color="r", linewidth=1)
        ax1.set_xlabel("Principal components by order of importance")
        ax1.set_ylabel("Inertia percentage")

        plt.title("Scree plot (hist), cummulative inertia (red line)")

        if show:
            self.scree = plt.gcf()
            plt.show()
        elif not show:
            return plt.gcf()

    def biplot(self, pc_select: tuple, labels=None):
        """
        Loosely based on implementation by Serafeim Loukas, serafeim.loukas@epfl.ch
        """
        fig, ax1 = plt.subplots(
            ncols=1,
            nrows=1,
            figsize=(6, 6),
            dpi=150,
        )

        first = pc_select[0] + 1  # Indexed at 0 so PC1 is arr[0]
        second = pc_select[1] + 1
        # n = coeff.shape[0]
        if self.target_col is not None:
            sns.scatterplot(x=f"PC{first}", y=f"PC{second}", hue=self.target_col, data=self.pcframe, ax=ax1)
        else:
            sns.scatterplot(x=f"PC{first}", y=f"PC{second}", data=self.pcframe, ax=ax1)
        # for i in range(n):
        #     plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        #     if labels is None:
        #         plt.text(
        # coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        #     else:
        #         plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')

        ###
        # Titles/Lables
        ax1.grid(True)
        fig.suptitle(f"Projection of the population on PC{first} and PC{second}")
        plt.xlabel(f"PC{first}")
        plt.ylabel(f"PC{second}")
        #
        ###
        plt.tight_layout()
        plt.show()
