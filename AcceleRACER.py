from typing import Tuple, Union

import numpy as np
import pandas as pd

from optbinning import MDLP, MulticlassOptimalBinning as MOB, OptimalBinning as OB
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from mlxtend.frequent_patterns import apriori, association_rules

class RACERPreprocessor:
    def __init__(self, target: str = "auto", max_n_bins=32, max_num_splits=32, use_optimal_quantizer=False):
        """RACER preprocessing step that quantizes numerical columns and dummy encodes the categorical ones.
        Quantization is based on the optimal binning algorithm for "multiclass" tasks and the entropy-based MDLP
        algorithm for "binary" tasks.

        Args:
            target (str, optional): Whether the task is "multiclass" or "binary" classification. Defaults to "auto" which attempts automatically infer the task from `y`.
            max_n_bins (int, optional): Maximum number of bins to quantize in. Defaults to 32.
            max_num_splits (int, optional): Maximum number of splits to consider at each partition for MDLP. Defaults to 32.
        """
        assert target in [
            "multiclass",
            "binary",
            "auto",
        ], "`target` must either be 'multiclass', 'binary' or 'auto'."
        if use_optimal_quantizer:
            self._quantizer = OB()
        else:
            if target == "multiclass":
                self._quantizer = MOB(max_n_bins=max_n_bins)
            elif target == "binary":
                self._quantizer = MDLP(max_candidates=max_num_splits)
            else:
                self._quantizer = "infer"
                self._max_n_bins = max_n_bins
                self._max_candidates = max_num_splits

    def fit_transform_pandas(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocesses the dataset by replacing nominal vaues with dummy variables.
        Converts to numpy boolean arrays and returns the dataset. All numerical values are discretized
        using an optimal binning strategy that employs a decision tree as a preprocessing step.
        (This uses the legacy pandas dummy encoder. You can use this to retain total backward compatibility with previous code)

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Features matrix
            y (Union[pd.DataFrame, np.ndarray]): Targets vector

        Returns:
            Tuple[np.ndarray, np.ndarray]: Transformed features matrix and targets vectors.
        """
        X, y = pd.DataFrame(X), pd.DataFrame(y)
        if self._quantizer == "infer":
            uniques = y.nunique().values
            if uniques > 2:
                self._quantizer = MOB(max_n_bins=self._max_n_bins)
            else:
                self._quantizer = MDLP(max_candidates=self._max_candidates)
        numerics_X = X.select_dtypes(include=[np.number]).columns.tolist()
        if numerics_X:
            for col in numerics_X:
                self._quantizer.fit(X[col].values, np.squeeze(y.values))
                bins = [X[col].min()] + self._quantizer.splits.tolist() + [X[col].max()]
                X[col] = pd.cut(X[col], bins=bins, include_lowest=True, labels=False)
        X, y = X.astype("category"), y.astype("category")
        X = pd.get_dummies(X).to_numpy()
        y = pd.get_dummies(y).to_numpy()
        return X, y

    def fit_transform(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocesses the dataset by replacing nominal vaues with dummy variables.
        Converts to numpy boolean arrays and returns the dataset. All numerical values are discretized
        using an optimal binning strategy that employs a decision tree as a preprocessing step.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Features matrix
            y (Union[pd.DataFrame, np.ndarray]): Targets vector

        Returns:
            Tuple[np.ndarray, np.ndarray]: Transformed features matrix and targets vectors.
        """
        X, y = pd.DataFrame(X), pd.DataFrame(y)
        if self._quantizer == "infer":
            uniques = y.nunique().values
            if uniques > 2:
                self._quantizer = MOB(max_n_bins=self._max_n_bins)
            else:
                self._quantizer = MDLP(max_candidates=self._max_candidates)
        numerics_X = X.select_dtypes(include=[np.number]).columns.tolist()
        if numerics_X:
            for col in numerics_X:
                self._quantizer.fit(X[col].values, np.squeeze(y.values))
                bins = [X[col].min()] + self._quantizer.splits.tolist() + [X[col].max()]
                X[col] = pd.cut(X[col], bins=bins, include_lowest=True, labels=False)
        X, y = X.astype("category"), y.astype("category")
        X = OneHotEncoder(sparse_output=False).fit_transform(X).astype(bool)
        y = LabelBinarizer().fit_transform(y).astype(bool)
        return X, y

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]
    ):
        """Fits the preprocessor on X and y for downstream transformations.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Features vector
            y (Union[pd.DataFrame, np.ndarray]): Targets vector
        """
        print(
            "It is strongly recommended that you use fit_transform on your entire dataset."
        )
        print(
            "Use this option ONLY if you're certain new unseen values will not be encountered at test time."
        )
        X, y = pd.DataFrame(X), pd.DataFrame(y)
        if self._quantizer == "infer":
            uniques = y.nunique().values
            if uniques > 2:
                self._quantizer = MOB(max_n_bins=self._max_n_bins)
            else:
                self._quantizer = MDLP(max_candidates=self._max_candidates)
        numerics_X = X.select_dtypes(include=[np.number]).columns.tolist()
        if numerics_X:
            self._bins = []
            for col in numerics_X:
                self._quantizer.fit(X[col].values, np.squeeze(y.values))
                bins = [X[col].min()] + self._quantizer.splits.tolist() + [X[col].max()]
                self._bins.append(bins)
                X[col] = pd.cut(X[col], bins=bins, include_lowest=True, labels=False)
        X, y = X.astype("category"), y.astype("category")
        self._X_encoder = OneHotEncoder(sparse_output=False).fit(X)
        self._y_encoder = LabelBinarizer().fit(y)

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms the provided new X and y with previously fitted preprocessor.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Features matrix
            y (Union[pd.DataFrame, np.ndarray]): Targets vector

        Returns:
            Tuple[np.ndarray, np.ndarray]: Transformed features matrix and targets vectors.
        """
        X, y = pd.DataFrame(X), pd.DataFrame(y)
        numerics_X = X.select_dtypes(include=[np.number]).columns.tolist()
        if numerics_X:
            for col, bin in zip(numerics_X, self._bins):
                X[col] = pd.cut(X[col], bins=bin, include_lowest=True, labels=False)
        X, y = X.astype("category"), y.astype("category")
        X, y = self._X_encoder.transform(X).astype(bool), self._y_encoder.transform(
            y
        ).astype(bool)
        return X, y
    
from typing import Tuple

import numpy as np
from numpy import (
    bitwise_and as AND,
    bitwise_not as NOT,
    bitwise_or as OR,
    bitwise_xor as XOR,
)


def XNOR(input: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Computes the XNOR gate. (semantically the same as `input == other`)

    Args:
        input (np.ndarray): Input array
        other (np.ndarray): Other input array

    Returns:
        np.ndarray: XNOR(input, other) as an array
    """
    return NOT(XOR(input, other))


class RACER:
    def __init__(
        self,
        alpha=0.9,
        suppress_warnings=False,
        benchmark=False,
        fitness_treshhold = 0.7,
        support_treshhold = 0.1,
        feature_train=False,
        feature_class=False,
        feature_no_fitness_change=False
    ):
        """Initialize the RACER class

        Args:
            alpha (float, optional): Value of alpha according to the RACER paper. Defaults to 0.9.
            suppress_warnings (bool, optional): Whether to suppress any warnings raised during prediction. Defaults to False.
            benchmark (bool, optional): Whether to time the `fit` method for benchmark purposes. Defaults to False.
        """
        self._alpha, self._beta = alpha, 1.0 - alpha
        self._suppress_warnings = suppress_warnings
        self._benchmark = benchmark
        self._has_fit = False
        self._feature_train = feature_train
        self._feature_class = feature_class
        self._fitness_treshhold = fitness_treshhold
        self._support_treshhold = support_treshhold
        self._feature_no_fitness_change = feature_no_fitness_change

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the RACER algorithm on top of input data X and targets y.
        The code is written in close correlation to the pseudo-code provided in the RACER paper with some slight modifications.

        Args:
            X (np.ndarray): Features vector
            y (np.ndarray): Targets vector
        """
        if self._benchmark:
            from time import perf_counter

            tic = perf_counter()

        np.set_printoptions(threshold=np.inf)

        self._X, self._y = X, y
        self._cardinality, self._rule_len = self._X.shape
        self._classes = np.unique(self._y, axis=0)
        self._class_indices = {
            self._label_to_int(cls): np.where(XNOR(self._y, cls).min(axis=-1))[0]
            for cls in self._classes
        }

        print(len(self._X))

        self._apriori_merge(fitness_treshhold = self._fitness_treshhold, support_treshhold = self._support_treshhold)

        if (self._feature_train == True):
            self._cardinality, self._rule_len = self._X.shape
            self._classes = np.unique(self._y, axis=0)
            self._class_indices = {
            self._label_to_int(cls): np.where(XNOR(self._y, cls).min(axis=-1))[0]
            for cls in self._classes
            }
        # if (self._feature_no_fitness_change_train == True):


        self._create_init_rules()
        print(len(self._extants_if))

        for cls in self._class_indices.keys():
            for i in range(len(self._class_indices[cls])):
                for j in range(i + 1, len(self._class_indices[cls])):
                    self._process_rules(
                        self._class_indices[cls][i], self._class_indices[cls][j]
                    )

        independent_indices = NOT(self._extants_covered)
        self._extants_if, self._extants_then, self._fitnesses = (
            self._extants_if[independent_indices],
            self._extants_then[independent_indices],
            self._fitnesses[independent_indices],
        )

        self._generalize_extants()

        # https://stackoverflow.com/questions/64238462/numpy-descending-stable-arg-sort-of-arrays-of-any-dtype
        args = (
            len(self._fitnesses)
            - 1
            - np.argsort(self._fitnesses[::-1], kind="stable")[::-1]
        )

        self._final_rules_if, self._final_rules_then, self._fitnesses = (
            self._extants_if[args],
            self._extants_then[args],
            self._fitnesses[args],
        )

        self._finalize_rules()

        print(len(self._final_rules_if))


        self._has_fit = True

        if self._benchmark:
            self._bench_time = perf_counter() - tic

    def _apriori_merge(self, fitness_treshhold, support_treshhold):


        # Step 1: Prepare Data with Class Labels
        # Convert self._X and self._y into a DataFrame, including class labels as additional columns
        feature_columns = [f'feature_{i}' for i in range(self._X.shape[1])]
        class_columns = [f'class_{i}' for i in range(self._y.shape[1])]
        
        if(self._feature_class == True):
            feature_columns = [f'feature_{i}' for i in range(self._X.shape[1])]
            
            high_quality_apriori_rules_if = []
            high_quality_apriori_rules_then = []

            for cls in self._class_indices.keys():
                class_indices = self._class_indices[cls]
                
                X_class = self._X[class_indices]
                
                X_class_df = pd.DataFrame(X_class, columns=feature_columns)

                frequent_itemsets_class = apriori(X_class_df, min_support=support_treshhold, use_colnames=True)
                if frequent_itemsets_class.__len__() > 0:
                    apriori_rules_class = association_rules(frequent_itemsets_class, metric="confidence", support_only=True, min_threshold=0)
                else:
                    print("didn't find rule with apriori")
                    continue

                apriori_if = []
                apriori_then = []

                seen_rules = set()

                for _, rule in apriori_rules_class.iterrows():
                    # Create the binary vector for antecedents and consequents
                    antecedent_binary = np.array([1 if (feature in rule['antecedents'] or feature in rule["consequents"]) else 0 for feature in feature_columns])

                    # Convert the binary array to a tuple for efficient hashing and comparison
                    antecedent_tuple = tuple(antecedent_binary)

                    # Check if the tuple is already in the set
                    if antecedent_tuple not in seen_rules:
                        apriori_if.append(antecedent_binary)
                        apriori_then.append(self._y[class_indices][0])
                        seen_rules.add(antecedent_tuple)  # Add the new rule to the set

                for i in range(len(apriori_if)):
                    fitness = self._fitness_fn(
                        apriori_if[i], apriori_then[i]
                    )                    
                    if(fitness >= fitness_treshhold):
                        high_quality_apriori_rules_if.append(apriori_if[i]) 
                        high_quality_apriori_rules_then.append(apriori_then[i])  
                print("generated rule by apriori:",len(high_quality_apriori_rules_if))
                
            if(len(high_quality_apriori_rules_if) > 0):
                    self._X = np.vstack([self._X, np.array(high_quality_apriori_rules_if)])
                    self._y = np.vstack([self._y, np.array(high_quality_apriori_rules_then)])
        else:

            X_df = pd.DataFrame(self._X, columns=feature_columns)
            y_df = pd.DataFrame(self._y, columns=class_columns)
            combined_df = pd.concat([X_df, y_df], axis=1).astype(bool)

            # Step 2: Generate Apriori frequent itemsets and association rules
            frequent_itemsets = apriori(combined_df, min_support=support_treshhold, use_colnames=True)
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
                antecedent_binary = np.array([True if feature in rule['antecedents'] else False for feature in feature_columns])
                apriori_if.append(antecedent_binary)
                
                # Create binary vector for the THEN part based on class labels
                consequent_binary = np.array([True if feature in rule["consequents"] else False for feature in class_columns])
                apriori_then.append(consequent_binary)
                

            print("apriori finished")

            high_quality_apriori_rules_if = []
            high_quality_apriori_rules_then = []
            for i in range(len(apriori_if)):
                fitness = self._fitness_fn(
                    apriori_if[i], apriori_then[i]
                )                    
                if(fitness >= fitness_treshhold):
                    high_quality_apriori_rules_if.append(apriori_if[i]) 
                    high_quality_apriori_rules_then.append(apriori_then[i])  

            apriori_if = np.array(high_quality_apriori_rules_if)
            apriori_then = np.array(high_quality_apriori_rules_then)

            print("generated rule by apriori:",len(high_quality_apriori_rules_if))
            if(len(high_quality_apriori_rules_if) > 0):
                self._X = np.vstack([self._X, apriori_if])
                self._y = np.vstack([self._y, apriori_then])

    def predict(self, X: np.ndarray, convert_dummies=True) -> np.ndarray:
        """Given input X, predict label using RACER

        Args:
            X (np.ndarray): Input features vector
            convert_dummies (bool, optional): Whether to convert dummy labels back to integert format. Defaults to True.

        Returns:
            np.ndarray: Label as predicted by RACER
        """
        assert self._has_fit, "RACER has not been fit yet."
        labels = np.zeros((len(X), self._final_rules_then.shape[1]), dtype=bool)
        found = np.zeros(len(X), dtype=bool)
        for i in range(len(self._final_rules_if)):
            covered = self._covered(X, self._final_rules_if[i])
            labels[AND(covered, NOT(found))] = self._final_rules_then[i]
            found[covered] = True
            all_found = found.sum() == len(X)
            if all_found:
                break

        if not all_found:
            if not self._suppress_warnings:
                print(
                    f"WARNING: RACER was unable to find a perfect match for {len(X) - found.sum()} instances out of {len(X)}"
                )
                print(
                    "These instances will be labelled as the majority class during training."
                )
            leftover_indices = np.where(NOT(found))[0]
            for idx in leftover_indices:
                labels[idx] = self._closest_match(X[idx])

        if convert_dummies:
            labels = np.argmax(labels, axis=-1)

        return labels

    def _bool2str(self, bool_arr: np.ndarray) -> str:
        """Converts a boolean array to a human-readable string

        Args:
            bool_arr (np.ndarray): The input boolean array

        Returns:
            str: Human-readable string output
        """
        return np.array2string(bool_arr.astype(int), separator="")

    def display_rules(self) -> None:
        """Print out the final rules"""
        assert self._has_fit, "RACER has not been fit yet."
        print("Algorithm Parameters:")
        print(f"\t- Alpha: {self._alpha}")
        if self._benchmark:
            print(f"\t- Time to fit: {self._bench_time}s")
        print(
            f"\nFinal Rules ({len(self._final_rules_if)} total): (if --> then (label) | fitness)"
        )
        for i in range(len(self._final_rules_if)):
            print(
                f"\t{self._bool2str(self._final_rules_if[i])} -->"
                f" {self._bool2str(self._final_rules_then[i])}"
                f" ({self._label_to_int(self._final_rules_then[i])})"
                f" | {self._fitnesses[i]}"
            )

    def _closest_match(self, X: np.ndarray) -> np.ndarray:
        """Find the closest matching rule to `X` (This will be extended later)

        Args:
            X (np.ndarray): Input rule `X`

        Returns:
            np.ndarray: Matched rule
        """
        return self._majority_then

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Returns accuracy on the provided test data.

        Args:
            X_test (np.ndarray): Test features vector
            y_test (np.ndarray): Test targets vector

        Returns:
            float: Accuracy score
        """
        assert self._has_fit, "RACER has not been fit yet."
        try:
            from sklearn.metrics import accuracy_score
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required to use the score function. Install wit `pip install scikit-learn`."
            )
        if y_test.ndim != 1 and y_test.shape[1] != 1:
            y_test = np.argmax(y_test, axis=-1)
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def _fitness_fn(self, rule_if: np.ndarray, rule_then: np.ndarray) -> np.ndarray:
        """Returns fitness for a given rule according to the RACER paper

        Args:
            rule_if (np.ndarray): If part of a rule (x)
            rule_then (np.ndarray): Then part of a rule (y)

        Returns:
            np.ndarray: Fitness score for the rule as defined in the RACER paper
        """
        n_covered, n_correct = self._confusion(rule_if, rule_then)
        accuracy = n_correct / n_covered
        coverage = n_covered / self._cardinality
        return self._alpha * accuracy + self._beta * coverage

    def _covered(self, X: np.ndarray, rule_if: np.ndarray) -> np.ndarray:
        """Returns indices of instances if `X` that are covered by `rule_if`.
        Note that rule covers instance if EITHER of the following holds in a bitwise manner:
        1. instance[i] == 0
        2. instance[i] == 1 AND rule[i] == 1

        Args:
            X (np.ndarray): Instances
            rule_if (np.ndarray): If part of rule (x)

        Returns:
            np.ndarray: An array containing indices in `X` that are covered by `rule_if`
        """
        covered = OR(NOT(X), AND(rule_if, X)).min(axis=-1)
        return covered

    def _confusion(
        self, rule_if: np.ndarray, rule_then: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns n_correct and n_covered for instances classified by a rule.

        Args:
            rule_if (np.ndarray): If part of rule (x)
            rule_then (np.ndarray): Then part of rule (y)

        Returns:
            Tuple[np.ndarray, np.ndarray]: (n_covered, n_correct)
        """
        covered = self._covered(self._X, rule_if)
        n_covered = covered.sum()
        y_covered = self._y[covered]
        n_correct = XNOR(y_covered, rule_then).min(axis=-1).sum()
        return n_covered, n_correct

    def _get_majority(self) -> np.ndarray:
        """Return the majority rule_then from self._y

        Returns:
            np.ndarray: Majority rule_then
        """
        u, indices = np.unique(self._y, axis=0, return_inverse=True)
        return u[np.bincount(indices).argmax()]

    def _create_init_rules(self) -> None:
        """Creates an initial set of rules from theinput feature vectors"""
        self._extants_if = self._X.copy()
        self._extants_then = self._y.copy()
        self._extants_covered = np.zeros(len(self._X), dtype=bool)
        self._majority_then = self._get_majority()
        self._fitnesses = np.array(
            [
                self._fitness_fn(rule_if, rule_then)
                for rule_if, rule_then in zip(self._X, self._y)
            ]
        )

    def _composable(self, idx1: int, idx2: int) -> bool:
        """Returns true if two rules indicated by their indices are composable

        Args:
            idx1 (int): Index of the first rule
            idx2 (int): Index of the second rule

        Returns:
            bool: True if labels match and neither of the rules are covered. False otherwise.
        """
        labels_match = XNOR(self._extants_then[idx1], self._extants_then[idx2]).min()
        return (
            labels_match
            and not self._extants_covered[idx1]
            and not self._extants_covered[idx2]
        )

    def _process_rules(self, idx1: int, idx2: int) -> None:
        """Process two rules indiciated by their indices

        Args:
            idx1 (int): Index of the first rule
            idx2 (int): Index of the second rule
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
            rule1 (np.ndarray): The first rule
            rule2 (np.ndarray): The second rule

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
            index (int): Index of `new_rule`
            new_rule_if (np.ndarray): If part of `new_rule` (x)
            new_rule_then (np.ndarray): Then part of `new_rule` (y)
            new_rule_fitness (np.ndarray): Fitness of the `new_rule`
        """
        same_class_indices = self._class_indices[self._label_to_int(new_rule_then)]
        covered = self._covered(self._extants_if[same_class_indices], new_rule_if)
        self._extants_covered[same_class_indices[covered]] = True
        self._extants_covered[index] = False
        self._extants_if[index], self._extants_then[index], self._fitnesses[index] = (
            new_rule_if,
            new_rule_then,
            new_rule_fitness,
        )

    def _label_to_int(self, label: np.ndarray) -> int:
        """Converts dummy label to int

        Args:
            label (np.ndarray): Label to convert

        Returns:
            int: Converted label
        """
        return int(np.argmax(label))

    def _generalize_extants(self) -> None:
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

    def _finalize_rules(self) -> None:
        """Removes redundant rules to form the final ruleset"""
        temp_rules_if = self._final_rules_if
        temp_rules_then = self._final_rules_then
        temp_rules_fitnesses = self._fitnesses
        i = 0
        while i < len(temp_rules_if) - 1:
            mask = np.ones(len(temp_rules_if), dtype=bool)
            covered = self._covered(temp_rules_if[i + 1 :], temp_rules_if[i])
            mask[i + 1 :][covered] = False
            temp_rules_if, temp_rules_then, temp_rules_fitnesses = (
                temp_rules_if[mask],
                temp_rules_then[mask],
                temp_rules_fitnesses[mask],
            )
            i += 1

        self._final_rules_if, self._final_rules_then, self._fitnesses = (
            temp_rules_if,
            temp_rules_then,
            temp_rules_fitnesses,
        )