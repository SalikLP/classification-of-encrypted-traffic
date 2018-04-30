from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def runpca(X, num_comp=None):
    pca = PCA(n_components=num_comp, svd_solver='full')
    pca.fit(X)
    # print(pca.n_components_)
    # print(pca.explained_variance_ratio_)
    # print(sum(pca.explained_variance_ratio_))
    return pca


def plotvarianceexp(pca, num_comp):
    # plt.ion()
    rho = pca.explained_variance_ratio_
    rhosum = np.empty(len(rho))
    for i in range(1, len(rho) + 1):
        rhosum[i - 1] = np.sum(rho[0:i])
    ind = np.arange(num_comp)
    width = 0.35
    opacity = 0.8
    fig, ax = plt.subplots()
    ax.set_ylim(0, 1.1)
    # Variance explained by single component
    bars1 = ax.bar(ind, rho[0:num_comp], width, alpha=opacity, color="xkcd:blue", label='Single')
    # Variance explained by cummulative component
    bars2 = ax.bar(ind + width, rhosum[0:num_comp], width, alpha=opacity, color="g", label='Cummulative')
    # add some text for labels, title and axes ticks
    ax.set_ylabel('Variance Explained', fontsize=16)
    # ax.set_title('Variance Explained by principal components')
    ax.set_xticks(ind + width / 2)

    labels = ["" for x in range(num_comp)]
    for k in range(0, num_comp):
        labels[k] = ('$v{0}$'.format(k + 1))
    ax.set_xticklabels(labels, fontsize=16)
    ax.legend(fontsize=16)
    autolabel(bars2, ax)
    plt.yticks(fontsize=16)
    plt.draw()


def componentprojection(data, pca):
    V = pca.components_.T
    Z = data @ V
    return Z

def plotprojection(Z, pc, labels, class_labels):
    diff_labels = np.unique(labels)
    opacity = 0.8
    fig, ax = plt.subplots()
    color_map = {0: 'orangered', 1: 'royalblue', 2: 'lightgreen', 3: 'darkorchid', 4: 'teal', 5: 'darkslategrey',
                 6: 'darkgreen', 7: 'darkgrey'}
    for label in diff_labels:
        idx = labels == label
        ax.plot(Z[idx, pc], Z[idx, pc + 1], 'o', alpha=opacity, c=color_map[label], label='{label}'.format(label=class_labels[label]))
    # ax.plot(Z[idx_below, pc], Z[idx_below, pc + 1], 'o', alpha=opacity,
    #         label='{name} below mean'.format(name=attributeNames[att]))
    ax.set_ylabel('$v{0}$'.format(pc + 2))
    ax.set_xlabel('$v{0}$'.format(pc + 1))
    ax.legend()
    ax.set_title('Data projected on v{0} and v{1}'.format(pc+1, pc+2))
    # fig.savefig('v{0}_v{1}_{att}.png'.format(pc + 1, pc + 2, att=attributeNames[att]), dpi=300)
    plt.draw()


def showplots():
    plt.show()


def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%4.2f' % height,
                ha='center', va='bottom', fontsize=16)