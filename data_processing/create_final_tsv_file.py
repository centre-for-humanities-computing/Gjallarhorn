import os

def traverse_and_create_new_tsv_file(traverse_root_path: str, tsv_out_file: str):
    meta_data_files = []
    counter = 0
    print("Traversing")

    for root, subdirs, files in os.walk(traverse_root_path):
        for f in files:
            if f[-3:] == "tsv":
                meta_data_files.append((root, f))
                counter += 1
            
        if counter % 100 == 0:
            print(counter)


    print("Appending")
    all_lines_final = []
    counter_last = 0
    for root, f in meta_data_files:
        counter_last += 1
        with open(os.path.join(root, f), "r", encoding="utf-8") as f:
            pre_path = "/".join(str(f).split("/")[4:-1])

            all_lines = f.read().split("\n")
            for line in all_lines:
                final_line = pre_path + "/" + line
                all_lines_final.append(final_line)

        if counter_last % 100 == 0:
            print(f"{counter_last}/{len(meta_data_files)}")

    print("Writing")
    with open(tsv_out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines_final))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create final tsv')
    parser.add_argument("--directory_root", default="/work/processed_wav", type=str)
    parser.add_argument("--tsv_out", default="/work/processed_wav/metadata.tsv", type=str)
    args = parser.parse_args()
    traverse_and_create_new_tsv_file(args.directory_root, args.tsv_out)

