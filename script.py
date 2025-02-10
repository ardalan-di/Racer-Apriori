import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from AcceleRACER import RACER, RACERPreprocessor
import os
from itertools import product
from openpyxl import load_workbook

# Base directory
base_file = "C:\\Users\\Hkr\\Desktop\\bachelor project\\racerCode\\Racer-Apriori\\dataSet\\"

# List of ARFF file paths
arff_files = [
    base_file + "car evaluation\\car evaluation.arff",
    base_file + "Iris\\iris.arff",
    base_file + "tic tac\\tic tac.arff",
    base_file + "Contraceptive Method Choice\\cmc.arff",
]

support_thresholds = [0.3, 0.2, 0.1, 0.01]
fitness_thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]
alpha = 0.95

# Generate configurations dynamically
configurations = [
    {"alpha": alpha, "feature_apriori": False}
] + [
    {"alpha": alpha, "feature_apriori": fa, "feature_class": fc, "feature_train": ft, 
     "support_treshhold": s, "fitness_treshhold": f}
    for fa, fc, ft, s, f in product(
        [True], [True, False], [True, False], support_thresholds, fitness_thresholds
    )
]

output_excel_path = base_file + "results.xlsx"

def append_to_excel(df, file_path, sheet_name="Results"):
    """Appends a DataFrame to an existing Excel file or creates a new one."""
    if os.path.exists(file_path):
        # Load existing workbook
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
            try:
                # Load the existing sheet
                existing_df = pd.read_excel(file_path, sheet_name=sheet_name)
                # Append new data
                new_df = pd.concat([existing_df, df], ignore_index=True)
                # Overwrite the sheet
                new_df.to_excel(writer, sheet_name=sheet_name, index=False)
            except ValueError:
                # If sheet doesn't exist, create a new one
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        # Create new Excel file if it doesn't exist
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="w") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

# Loop through each ARFF file
for arff_file_path in arff_files:
    data, meta = arff.loadarff(arff_file_path)
    df = pd.DataFrame(data)

    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].str.decode('utf-8')

    X = df.drop(columns=['Class']).astype('category')
    Y = df[['Class']].astype('category')

    # Apply RACERPreprocessor
    X, Y = RACERPreprocessor().fit_transform(X, Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1, test_size=0.1)

    # Process each configuration
    for idx, config in enumerate(configurations, start=1):
        try:
            racer = RACER(**config)
            apriori_results = racer.fit(X_train, Y_train)
            score = racer.score(X_test, Y_test)

            result = {
                "ARFF File": os.path.basename(arff_file_path),
                **config,
                "Apriori_results": apriori_results,
                "Score": score,
            }

            # Convert result to DataFrame and append to Excel
            result_df = pd.DataFrame([result])
            append_to_excel(result_df, output_excel_path)

            print(f"✅ Saved config {idx} for {os.path.basename(arff_file_path)}")

        except Exception as e:
            print(f"❌ Error processing config {idx} for {os.path.basename(arff_file_path)}: {e}")

print(f"\nResults saved to {output_excel_path}")
