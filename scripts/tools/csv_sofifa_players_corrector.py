import os
import csv
import shutil

def process_csv_files(folder_path):
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        temp_file_path = os.path.join(folder_path, f"temp_{csv_file}")
        print(f"Processing file: {file_path}")
        
        # Dictionary to store all occurrences of each player_id
        player_id_occurrences = {}  # key: player_id, value: list of (index, row)
        rows = []
        duplicates_counter = 0
        empty_rows_counter = 0
        missing_id_counter = 0

        def is_empty_row(row):
            # Treat as empty if row is empty OR all cells are blank/whitespace
            return (not row) or (all((c is None) or (str(c).strip() == "") for c in row))
        
        with open(file_path, mode='r', encoding='utf-8', newline='') as infile:
            reader = csv.reader(infile)
            try:
                header = next(reader)
            except StopIteration:
                print(f"{csv_file} is empty. Skipping.")
                continue

            rows.append(header)
            
            if 'player_id' not in header:
                print(f"'player_id' column not found in {csv_file}. Skipping this file.")
                continue
            
            player_id_index = header.index('player_id')
            full_name_index = header.index('full_name') if 'full_name' in header else None
            
            # Read each row and store occurrences
            for index, row in enumerate(reader, start=1):
                # Skip completely empty rows
                if is_empty_row(row):
                    empty_rows_counter += 1
                    continue

                # Some rows might be short; pad them so indexing doesn't crash
                if len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))

                # DNS error assertion (only if row has at least 3 columns)
                if len(row) > 2 and row[2] == "DNS resolution error | sofifa.com | Cloudflare":
                    # We don't count these as "empty"; they're filtered for data quality
                    print(f"DNS error found for player (unknown id yet) in file {csv_file}. Skipping this player.")
                    continue

                player_id = row[player_id_index].strip()

                # Skip rows with missing player_id
                if player_id == "":
                    missing_id_counter += 1
                    continue

                if player_id in player_id_occurrences:
                    player_id_occurrences[player_id].append((index, row))
                else:
                    player_id_occurrences[player_id] = [(index, row)]
                rows.append(row)
        
        # List to store unique rows (keeping only the last occurrence)
        unique_rows = [rows[0]]  # Start with header
        
        # Process each player_id
        for player_id, occurrences in player_id_occurrences.items():
            # Keep the last occurrence
            last_occurrence = occurrences[-1]
            
            # If there are duplicates, remove previous occurrences
            if len(occurrences) > 1:
                duplicates_to_remove = occurrences[:-1]
                for idx, duplicate_row in duplicates_to_remove:
                    full_name = duplicate_row[full_name_index] if full_name_index is not None else 'N/A'
                    # print(f"Removing duplicate: player_id={player_id}, full_name={full_name}")
                    duplicates_counter += 1

            # Add the last occurrence to unique_rows
            unique_rows.append(last_occurrence[1])
        
        # Write the unique rows to a temporary CSV file
        with open(temp_file_path, mode='w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(unique_rows)
        
        # Replace the original file with the temporary file
        try:
            os.remove(file_path)
            shutil.move(temp_file_path, file_path)
            print(f"Finished processing {csv_file}. Duplicate rows removed based on 'player_id'.")
        except Exception as e:
            print(f"Error replacing the original file for {csv_file}: {e}")
            # Cleanup: Remove the temporary file if it exists
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        print(f"Removed {duplicates_counter} duplicate rows")
        print(f"Skipped {empty_rows_counter} completely empty rows")
        print(f"Skipped {missing_id_counter} rows missing 'player_id'\n")

# Specify the folder containing your CSV files
folder_path = r'C:\\Users\\kutalekj\\PycharmProjects\\sofifa-web-scraper\\output_optimized_phase5\\full'  # Replace with your folder path

process_csv_files(folder_path)
