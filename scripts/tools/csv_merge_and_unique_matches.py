import os
import csv

def merge_csv_files_without_duplicates(input_csv_paths, output_csv_path):

    if not input_csv_paths:
        print("No input CSV files provided.")
        return

    seen_rows = set()  # store unique rows in a set to quickly detect duplicates
    duplicates_found = False

    unique_rows_order = []  # collect unique rows in a list so that can write them in the order they appear

    header = None
    header_written = False

    for csv_path in input_csv_paths:
        if not os.path.isfile(csv_path):
            print(f"File not found: {csv_path}")
            continue

        with open(csv_path, mode='r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            file_header = next(reader, None)
            if file_header is None:
                print(f"File is empty: {csv_path}")
                continue
            if header is None:
                header = file_header  # if didn't store a header yet, store the first file's header as the "master" header
            else:
                if file_header != header:
                    print(f"Warning: Header in {csv_path} does not match the primary header. "
                          "Skipping file.")  # file's header doesn't match the master header
                    continue

            for row in reader:
                row_tuple = tuple(row)  # convert list to a tuple for set-operations
                if row_tuple in seen_rows:
                    print(f"Duplicate row found in {csv_path}: {row}")  # duplicate detected
                    duplicates_found = True
                else:
                    seen_rows.add(row_tuple)
                    unique_rows_order.append(row_tuple)  # unique row

    if header is None:
        print("No valid rows/headers found in input files. Output CSV will not be created.")  # write all unique rows to the output CSV
        return

    with open(output_csv_path, mode='w', encoding='utf-8', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(header)
        for row_tuple in unique_rows_order:
            writer.writerow(row_tuple)

    print(f"Merged output written to: {output_csv_path}")
    if not duplicates_found:
        print("No duplicate rows were found among the input CSV files.")


if __name__ == "__main__":
    input_files = [
        "C:\\Users\\kutalekj\\PycharmProjects\\MyFlashscoreScraper\\src\\m_25-09-03_BEL_NED_FRA_xG.csv",
        "C:\\Users\\kutalekj\\PycharmProjects\\MyFlashscoreScraper\\src\\m_25-09-03_ENG_xG.csv",
        "C:\\Users\\kutalekj\\PycharmProjects\\MyFlashscoreScraper\\src\\m_25-09-03_GER_ITA_SPA_xG.csv",
        "C:\\Users\\kutalekj\\PycharmProjects\\MyFlashscoreScraper\\src\\m_25-09-03_POR_POL_DEN_xG.csv",
        "C:\\Users\\kutalekj\\PycharmProjects\\MyFlashscoreScraper\\src\\m_25-09-03_SCO_AUS_TUR_SUI_AUT_SA_IND_xG.csv"
    ]
    output_file = "C:\\Users\\kutalekj\\PycharmProjects\\MyFlashscoreScraper\\src\\m_25-09-03_full.csv"

    merge_csv_files_without_duplicates(input_files, output_file)

