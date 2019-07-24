import matplotlib.pyplot as plt
import numpy as np

class SWEAT:
    def __init__(self, model_1, model_2, A, B):
        self.model_1 = model_1
        self.model_2 = model_2
        self.A = A
        self.B = B

    def word_assoc(self, model, w):
        return np.mean([model.wv.similarity(w, a) for a in self.A]) - np.mean([model.wv.similarity(w, b) for b in self.B])


    def test(self, X, n=10000, same=True, two_tails=True):
        """
        :param X:
        :param n:
        :param same:
        :param two_tails:
        :return:
        """

        if same:
            modscore_one = [self.word_assoc(self.model_1, x) for x in X]
            modscore_two = [self.word_assoc(self.model_2, x) for x in X]
            assoc_scores = modscore_one + modscore_two

            sum_0 = sum(modscore_one)
            sum_1 = sum(modscore_two)

            score = sum(modscore_one) - sum(modscore_two)
            eff_size = (np.mean(modscore_one) - np.mean(modscore_two)) / np.std(assoc_scores)

            if abs(sum_0) < 1e-3: print("Warning: Model1 is neutral")
            if abs(sum_1) < 1e-3: print("Warning: Model2 is neutral")

            # permutation test
            ds = []
            for _ in range(n):
                np.random.shuffle(assoc_scores)
                ds.append(sum(assoc_scores[:len(X)]) - sum(assoc_scores[-len(X):]))

            if two_tails:
                over = sum([abs(d) <= abs(score) for d in ds])
            else:
                over = sum([d <= score for d in ds])

            pval = 1 - (over / n)

            return round(score, 4), round(eff_size,4), round(pval, 4)
        else:
            raise NotImplementedError

    def plot(self, X, names=None):
        """ Plot SWEAT associations for target terms X wrt polarization sets A&B for models slices
            - models: gensim models
            - X: target terms (strings)
            - A: first attribute terms list (strings)
            - B: second attribute terms list (strings)
            - names: dictionary for plot labels
        """

        if names is not None:
            if type(names) != dict:
                raise RuntimeError("Names argument must be dictionary")

        assocs = [
            [
                [
                    [m.wv.similarity(w, a) for a in self.A],
                    [m.wv.similarity(w, b) for b in self.B]
                ] for w in X
                ] for m in [self.model_1, self.model_2]
            ]

        f, axes = plt.subplots(1, 2, sharey=True)
        f.set_size_inches(12, 6)

        for i, ass_mod in enumerate(assocs):

            S = []  # vector for computing cumulative sum of association deltas
            ax = axes[i]

            for j, ass_word in enumerate(ass_mod):
                assA = ass_word[0]
                assB = ass_word[1]

                boxA = ax.boxplot(assA,
                                  positions=[2 * j - 0.3], widths=0.3,
                                  boxprops=dict(color="red"), vert=False, showmeans=True, meanline=True)

                boxB = ax.boxplot(assB,
                                  positions=[2 * j + 0.3], widths=0.3,
                                  boxprops=dict(color="blue"), vert=False, showmeans=True, meanline=True)

                # compute means and delta
                muA = np.mean(assA)
                muB = np.mean(assB)
                dAB = muA - muB
                S.append(dAB)

                # word arrow
                ax.arrow(muA, 2 * j, -dAB, 0,
                         head_width=0.15, head_length=0.02, lw=1.5, length_includes_head=True, color='black'
                         )
            # cumulative arrow
            ax.arrow(0, -2, -sum(S), 0,
                     head_width=0.2, head_length=0.02, lw=3, length_includes_head=True, color='black')

            # plot setup & cosmetics
            ax.set_yticks(list(range(0, 2 * len(X), 2)) + [-2])
            ax.set_yticklabels(X + ['Cumulative'])

            if names is None:
                labels = ["A", "B"]
            else:
                labels = [names['A'], names['B']]
            ax.legend(handles=[boxA['boxes'][0], boxB['boxes'][0]], labels=labels)

            ax.axvline(0, lw=1, ls='--', alpha=0.3, color='k')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-3, (2 * len(X)))
            ax.set_xlabel('cosine similarity')
            ax.axhline(-1, color='black', lw=1)
            if names is None:
                ax.set_title("model %s" % i)
            else:
                ax.set_title(names['models'][i])

        plt.show()
