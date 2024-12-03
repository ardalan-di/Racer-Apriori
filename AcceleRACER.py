from typing import Tuple, Union
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
from typing import Union
import numpy as np
from discretization import *
from mlxtend.frequent_patterns import apriori, association_rules



class RACERPreprocesser:
    def __init__(self, framework="numpy"):
        assert framework in [
            "numpy",
            "torch",
        ], "framework must be either 'numpy' or 'torch'"
        self.framework = framework

    def fit_transform(
        self, data: Union[pd.DataFrame, np.ndarray], dataTypes
    ):
        """Preprocesses the dataset by replacing nominal values with dummy variables.
        Converts to torch bool tensors and returns the dataset. All numerical values are discretized
        into equal-sized bins using a quantile-based method (pd.qcut).
        Args:
            X (pandas.DataFrame or np.ndarray): features vector
            y (pandas.DataFrame or np.ndarray): targets vector
        Returns:
            X (torch.Tensor or np.ndarray): features vector
            y (torch.Tensor or np.ndarray): targets vector
        """
        
        classIndex = data.shape[1]-1;        
        X = data[:, 0:data.shape[1]-1];
        y = data[:, data.shape[1]-1];
        #numerics = X.select_dtypes(include=[np.number]).columns.tolist()
                
        numAttrIndex = [i for i, t in enumerate(dataTypes) if t == 'numeric' or t == 'integer' or t == 'real'];        
        
        if numAttrIndex:
            #X[numerics] = pd.qcut(X[numerics], bins=self.bins)
            for num in numAttrIndex:                
                newData = Discretization.basedOnEntropy(data[:,[num,classIndex]]);
                X[:, num] = newData;
        
        X = pd.get_dummies(pd.DataFrame(X)).to_numpy()
        y = pd.get_dummies(y).to_numpy()
        if self.framework == "torch":
            import torch

            X, y = torch.tensor(X, dtype=torch.bool), torch.tensor(y, dtype=torch.bool)
        return X, y


from numpy import (
    logical_and as AND,
    logical_not as NOT,
    logical_or as OR,
    logical_xor as XOR,
)


def XNOR(input: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Computes the XNOR gate. (semantically the same as `input == other`)
    Args:
        input (np.ndarray): Input tensor
        other (np.ndarray): Other input tensor
    Returns:
        np.ndarray: Output tensor
    """
    return NOT(XOR(input, other)).astype(bool)


class RACER:
    def __init__(
        self,
        alpha=0.9,
        fitness_fn="weighted_average",
        gamma=0.6,
        suppress_warnings=False,
        benchmark=False,
    ):
        """Initialize the RACER class.
        Args:
            alpha (float, optional): Value of alpha according to the RACER paper. Defaults to 0.9.
            fitness_function (str in ["weighted_average", "f-beta"], optional): Choice of fitness function to use. Defaults to "weighted_average".
            gamma (float, optional): Weight given to coverage score during closest_matching if rules cannot perfectly describe a given input.
                                      values > 0.5 recommended. Defaults to 0.6.
            suppress_warnings (bool, optional): Whether to suppress any warnings raised during prediction. Defaults to False.
            benchmark (bool, optional): Whether to time the `fit` method for benchmark purposes. Defaults to False.
        """
        self._alpha, self._beta = alpha, 1 - alpha
        self._gamma = gamma

        assert fitness_fn in [
            "weighted_average",
            "f-beta",
        ], "fitness_function must be either 'weighted_average' or 'f-beta'"
        self._fitness_fn = (
            self._fitness_weighted_avg
            if fitness_fn == "weighted_average"
            else self._fitness_f_beta
        )
        self._fitness_fn_name = (
            "Weighted Average" if fitness_fn == "weighted_average" else "F-Beta"
        )

        self._suppress_warnings = suppress_warnings

        self._benchmark = benchmark

        self._has_fit = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the RACER algorithm on top of input data X and targets y.
        The code is written in a way that it is similar to the pseudo-code provided in the RACER paper.
        Args:
            X (np.ndarray): features vector
            y (np.ndarray): targets vector
        """
        np.set_printoptions(threshold=np.inf)

        if self._benchmark:
            from time import perf_counter

            tic = perf_counter()

        self._X, self._y = X, y

        # Step 1: Prepare Data with Class Labels
        # Convert self._X and self._y into a DataFrame, including class labels as additional columns
        feature_columns = [f'feature_{i}' for i in range(self._X.shape[1])]
        class_columns = [f'class_{i}' for i in range(self._y.shape[1])]
        
        X_df = pd.DataFrame(self._X, columns=feature_columns)
        y_df = pd.DataFrame(self._y, columns=class_columns)
        combined_df = pd.concat([X_df, y_df], axis=1).astype(bool)

        # Step 2: Generate Apriori frequent itemsets and association rules
        frequent_itemsets = apriori(combined_df, min_support=0.1, use_colnames=True)
        apriori_rules = association_rules(frequent_itemsets, metric="confidence", support_only=True, min_threshold=0)
        # Step 3: Separate IF and THEN Parts Using Class Labels in Consequents
        apriori_if = []
        apriori_then = []


        for _, rule in apriori_rules.iterrows():

            flag = False
            for s in rule['antecedents']:
                if 'class_' in s or len(rule['antecedents']) < 1:
                    flag = True

            for s in rule['consequents']:
                if 'feature_' in s or len(rule['consequents']) != 1:
                    flag = True

            if flag == True:
                continue

            # Create binary vector for the IF part based on features
            antecedent_binary = np.array([1 if feature in rule['antecedents'] else 0 for feature in feature_columns])
            apriori_if.append(antecedent_binary)
            
            # Create binary vector for the THEN part based on class labels
            consequent_binary = np.array([1 if feature in rule["consequents"] else 0 for feature in class_columns])
            apriori_then.append(consequent_binary)

        # Step 4: Convert lists to numpy arrays
        # apriori_if = np.array(apriori_if)
        # apriori_then = np.array(apriori_then)

        # print("generated rule by apriori:",len(apriori_if))
        # if(len(apriori_if) > 0):
        #     self._X = np.vstack([self._X, apriori_if])
        #     self._y = np.vstack([self._y, apriori_then])

        self._cardinality, self._rule_len = self._X.shape
        self._classes = np.unique(self._y, axis=0)
        self._class_indices = {
            self._label_to_int(cls): np.where(np.min(XNOR(self._y, cls), axis=-1))[0]
            for cls in self._classes
        }
        print("apriori finished")
        high_quality_apriori_rules_if = []
        high_quality_apriori_rules_then = []
        for i in range(len(apriori_if)):
            fitness = self._fitness_fn(
                apriori_if[i], apriori_then[i]
            )                    
            if(fitness >= 0.2):
                high_quality_apriori_rules_if.append(apriori_if[i]) 
                high_quality_apriori_rules_then.append(apriori_then[i])  

        apriori_if = np.array(high_quality_apriori_rules_if)
        apriori_then = np.array(high_quality_apriori_rules_then)

        print("generated rule by apriori:",len(high_quality_apriori_rules_if))
        if(len(high_quality_apriori_rules_if) > 0):
            self._extants_if = apriori_if
            self._extants_then = apriori_then

        self._create_init_rules()

        for cls in self._class_indices.keys():
            indices = self._class_indices[cls]
            i, j = np.triu_indices(len(indices), k=1)
            i, j = indices[i], indices[j]
            for i_idx, j_idx in zip(i, j):
                self._process_rules(i_idx, j_idx)

        independent_indices = ~(self._extants_covered)
        self._extants_if, self._extants_then, self._fitnesses = (
            self._extants_if[independent_indices],
            self._extants_then[independent_indices],
            self._fitnesses[independent_indices],
        )

        self._generalize_extants()

        args = np.argsort(self._fitnesses)[::-1]
        self._final_rules_if, self._final_rules_then, self._fitnesses = (
            self._extants_if[args],
            self._extants_then[args],
            self._fitnesses[args],
        )

        self._has_fit = True

        if self._benchmark:
            self._bench_time = perf_counter() - tic

    def predict(self, X: np.ndarray, convert_dummies=True) -> np.ndarray:
        """Given input X, predict label using RACER
        Args:
            X (np.ndarray): input features vector
            convert_dummies (bool): whether to convert the output to a one-dimensional array
        Returns:
            np.ndarray: label as predicted by RACER
        """
        assert self._has_fit, "RACER has not been fit yet."
        labels = np.zeros((len(X), self._final_rules_then.shape[1]), dtype=bool)
        found = np.zeros(len(X), dtype=bool)
        for i in range(len(self._final_rules_if)):
            covered = self._covered(X, self._final_rules_if[i])
            labels[AND(covered, NOT(found))] = self._final_rules_then[i]
            found[covered] = True
            if found.sum() == len(X):  # -> every instance was matched to a rule
                break

        all_found = found.sum() == len(X)
        if not all_found:
            print(
                f"Warning: RACER was unable to find a perfect match for {len(X) - found.sum()} instances out of {len(X)}."
            )
            print(
                "Labels for these instances will be determined by a closest match algorithm."
            )
            leftover_indices = np.where(~found)[0]
            for idx in leftover_indices:
                labels[idx] = self._closest_match(X[idx])

        if convert_dummies:
            labels = np.argmax(labels, axis=-1)

        return labels

    def display_rules(self):
        """Print out the final rules"""        
        assert self._has_fit, "RACER has not been fit yet."
        print("Algorithm Parameters:")
        print(f"\t- Fitness Function: {self._fitness_fn_name}")
        print(f"\t- Alpha: {self._alpha}")
        print(f"\t- Gamma: {self._gamma}")
        if self._benchmark:
            print(f"\t- Compute Device: CPU")
            print(f"\t- Time to fit: {self._bench_time}s")
        print(
            f"\nFinal Rules ({len(self._final_rules_if)} total): (if --> then (label) | fitness)"
        )
        for i in range(len(self._final_rules_if)):
            print(
                f"\t{np.array2string(self._final_rules_if[i].astype(int), separator='')} -->"
                f" {np.array2string(self._final_rules_then[i].astype(int), separator='')}"
                f" ({np.argmax(self._final_rules_then[i].astype(int))})"
                f" | {self._fitnesses[i]}"
            )            

    def _closest_match(self, X: np.ndarray) -> np.ndarray:
        """Find the closest matching rule to `X`
        Args:
            X (np.ndarray): input `X`
        Returns:
            np.ndarray: matched rule
        """
        # coverage := count of covered bits by a rule. Higher is better.
        int_X = X.astype(int)  # <- cast boolean array to integer array
        overlap = OR(NOT(int_X), AND(self._final_rules_if, int_X)).sum(axis=-1)
        overlap = overlap / self._rule_len  # -> normalize by rule length
        scores = np.multiply(self._gamma * overlap, (1 - self._gamma) * self._fitnesses)
        argmax = np.argmax(scores)
        return self._final_rules_then[argmax]

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Returns accuracy on the provided test data
        Args:
            X_test (np.ndarray): test features vector
            y_test (np.ndarray): test targets vector
        Returns:
            float: accuracy score
        """
        assert self._has_fit, "RACER has not been fit yet."
        X_test, y_test = X_test.astype(np.float32), y_test.astype(np.float32)
        if y_test.ndim != 1 and y_test.shape[1] != 1:
            y_test = np.argmax(y_test, axis=-1)
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def _fitness_weighted_avg(
        self, rule_if: np.ndarray, rule_then: np.ndarray
    ) -> float:
        """Returns fitness for a given rule according to the RACER paper
        Args:
            rule_if (np.ndarray): if part of a rule (x)
            rule_then (np.ndarray): then part of a rule (y)
        Returns:
            float: fitness score for the rule as defined in the RACER paper
        """
        n_covered, n_correct = self._confusion(rule_if, rule_then)
        accuracy = n_correct / n_covered
        coverage = n_covered / self._cardinality
        return self._alpha * accuracy + self._beta * coverage

    def _fitness_f_beta(self, rule_if: np.ndarray, rule_then: np.ndarray) -> np.ndarray:
        """Returns f-beta-score fitness for a given rule
        Args:
            rule_if (np.ndarray): if part of a rule (x)
            rule_then (np.ndarray): then part of a rule (y)
        Returns:
            np.ndarray: f-beta-score fitness for the rule
        """
        # `beta` in f-beta-score is chosen such that recall is considered `beta` times as important as precision
        # https://en.wikipedia.org/wiki/F-score
        beta = self._beta / self._alpha
        n_covered, n_correct = self._confusion(rule_if, rule_then)
        accuracy = n_correct / n_covered
        coverage = n_covered / self._cardinality
        return (
            (1 + beta**2) * (accuracy * coverage) / (beta**2 * accuracy + coverage)
        )

    def _covered(self, X: np.ndarray, rule_if: np.ndarray) -> np.ndarray:
        """Returns indices of instances in `X` that are covered by `rule_if`.
        Note that rule covers instance if EITHER of the following holds in a bitwise manner:
        1. instance[i] == 0
        2. instance[1] == 1 AND rule[i] == 1
        Args:
            X (np.ndarray): instances
            rule_if (np.ndarray): if part of rule (x)
        Returns:
            np.ndarray: An array containing indices of instances in `X_same_class` that are covered by `rule_if`
        """
        covered = OR(NOT(X), AND(rule_if, X)).min(axis=-1)
        return covered

    def _confusion(
        self, rule_if: np.ndarray, rule_then: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns n_correct and n_covered for instances classified by a rule.
        Args:
            rule_if (np.ndarray): if part of rule (x)
            rule_then (np.ndarray): then part of rule (y)
        Returns:
            Tuple[np.ndarray, np.ndarray]: (n_covered, n_correct)
        """
        covered = self._covered(self._X, rule_if)
        n_covered = covered.sum()
        y_covered = self._y[covered]
        n_correct = XNOR(y_covered, rule_then).min(axis=-1).sum()
        return n_covered, n_correct

    def _create_init_rules(self):
        """Creates an initial set of rules from the input feature vectors"""
        if(hasattr(self,'_extants_if')):
            self._extants_if = np.vstack([self._X, self._extants_if])
            self._extants_then = np.vstack([self._y, self._extants_then])
        else:
            self._extants_if = self._X
            self._extants_then = self._y 
        self._extants_covered = np.zeros(len(self._extants_if), dtype=bool)
        self._fitnesses = np.array(
            [
                self._fitness_fn(rule_if, rule_then)
                for rule_if, rule_then in zip(self._extants_if,self._extants_then)
            ]
        )

    def _composable(self, idx1: int, idx2: int) -> bool:
        """Returns true if two rules indicated by their indices are composable
        Args:
            idx1 (int): index of the first rule
            idx2 (int): index of the second rule
        Returns:
            bool: True if labels match and neither of the rules are covered. False otherwise.
        """
        labels_match = XNOR(self._extants_then[idx1], self._extants_then[idx2]).min()
        return (
            labels_match
            and not self._extants_covered[idx1].all()
            and not self._extants_covered[idx2].all()
        )

    def _process_rules(self, idx1: int, idx2: int):
        """Process two rules indicated by their indices
        Args:
            idx1 (int): index of the first rule
            idx2 (int): index of the second rule
        """
        if self._composable(idx1, idx2):
            composition = self._compose(self._extants_if[idx1], self._extants_if[idx2])
            composition_fitness = self._fitness_fn(
                composition, self._extants_then[idx1]
            )

            if composition_fitness > np.maximum(
                self._fitnesses[idx1], self._fitnesses[idx2]
            ):
                self._update_extants(
                    idx1, composition, self._extants_then[idx1], composition_fitness
                )

    def _compose(self, rule1: np.ndarray, rule2: np.ndarray) -> np.ndarray:
        """Composes rule1 with rule2
        Args:
            rule1 (np.ndarray): the first rule
            rule2 (np.ndarray): the second rule
        Returns:
            np.ndarray: The composed rule which is simply the bitwise OR of the two rules
        """
        return OR(rule1, rule2)

    def _update_extants(
        self,
        index: int,
        new_rule_if: np.ndarray,
        new_rule_then: np.ndarray,
        new_rule_fitness: np.ndarray,
    ):
        """Remove all rules from current extants that are covered by `new_rule`.
        Then append new rule to extants.
        Args:
            index (int): index of `new_rule`
            new_rule_if (np.ndarray): if part of `new_rule` (x)
            new_rule_then (np.ndarray): then part of `new_rule` (y)
            new_rule_fitness (np.ndarray): fitness of the `new_rule`
        """
        same_class_indices = self._class_indices[self._label_to_int(new_rule_then)]
        covered = self._covered(self._extants_if[same_class_indices], new_rule_if)
        self._extants_covered[same_class_indices[covered]] = True
        self._extants_covered[index] = False  # -> except new rule from covered
        self._extants_if[index], self._extants_then[index], self._fitnesses[index] = (
            new_rule_if,
            new_rule_then,
            new_rule_fitness,
        )

    def _label_to_int(self, label: np.ndarray) -> int:
        """Converts dummy label to int
        Args:
            label (np.ndarray): label to convert
        Returns:
            int: converted label
        """
        return int(np.argmax(label))

    def _generalize_extants(self):
        """Generalize the extants by flipping every 0 to a 1 and checking if the fitness improves."""
        new_extants_if = np.zeros_like(self._extants_if, dtype=bool)
        for i in range(len(self._extants_if)):
            for j in range(len(self._extants_if[i])):
                if not self._extants_if[i][j]:
                    self._extants_if[i][j] = True
                    fitness = self._fitness_fn(
                        self._extants_if[i], self._extants_then[i]
                    )
                    if fitness > self._fitnesses[i]:
                        self._fitnesses[i] = fitness
                    else:
                        self._extants_if[i][j] = False
            new_extants_if[i] = self._extants_if[i]
        self._extants_if = new_extants_if

    def getNumOfRules(self):
        return len(self._final_rules_if);

    def reduceRules(self):
        assert self._has_fit, "RACER has not been fit yet."
        i = 0;
        j = 0;
        while(i < len(self._final_rules_if)-1):
            j = i+1;
            coveredIndex = [];
            while(j < len(self._final_rules_if)):
                covered = True;
                for t in range(len(self._final_rules_if[j])):
                    if self._final_rules_if[j][t] and not self._final_rules_if[i][t]:
                        covered = False;
                        break;
                if covered :
                    coveredIndex.append(j);
                j += 1;
            self._final_rules_if = np.delete(self._final_rules_if, coveredIndex, axis = 0);
            self._final_rules_then = np.delete(self._final_rules_then, coveredIndex, axis = 0);
            self._fitnesses = np.delete(self._fitnesses, coveredIndex, axis = 0);
            i += 1;
                        

