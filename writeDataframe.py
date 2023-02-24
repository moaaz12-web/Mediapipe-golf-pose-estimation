import os
import pandas as pd

def writeDataframe(df1, df2, output_path):
    # Reset the index of both dataframes
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    # Find the common column names between df1 and df2
    common_cols = list(set(df1.columns) & set(df2.columns))

    # Combine the two dataframes into a single dataframe
    result = pd.concat([df1[common_cols], df2[common_cols]], axis=1)

    # Create an empty dataframe to store the output values
    output_df = pd.DataFrame()

    # Loop through each row of the common columns
    for row_idx, row in result.iterrows():

        # Create a dataframe to store the formatted values for the row
        formatted_values = pd.DataFrame(columns=common_cols)

        # Loop through each column of the row
        for col_name in common_cols:

            # Get the values for the current column from df1 and df2, filling with "NONE" if missing
            v1 = str(df1[col_name].iloc[row_idx]) if row_idx < len(df1) else "0"
            v2 = str(df2[col_name].iloc[row_idx]) if row_idx < len(df2) else "0"

            # Format the values as "df1_value/df2" or "df1/df2"
            formatted_values[col_name] = [f"{v1}/{v2}"]

        # Append the formatted values for the row to the output dataframe
        output_df = pd.concat([output_df, formatted_values], ignore_index=True)

    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Write the output dataframe to an Excel file in the output directory
    output_file_path = os.path.join(output_path, "data.xlsx")
    output_df.to_excel(output_file_path, index=False)
