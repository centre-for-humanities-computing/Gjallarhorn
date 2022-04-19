import os

def traverse_and_create_new_tsv_file(traverse_root_path: str, tsv_out_file: str):
    meta_data_files = []
    for root, subdirs, files in os.walk(traverse_root_path):
        for f in files:
            if f[-3:] == "tsv":
                meta_data_files.append((root, f))

    print(meta_data_files)
    #for root, f in meta_data_files:
    #    with open(os.path.join(root, f), "r", encoding="utf-8") as f:
    #        all_lines = f.read().split("\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create final tsv')
    parser.add_argument("--directory_root", default="/work/processed_wav", type=str)
    parser.add_argument("--tsv_out", default="/work/processed_wav/metadata.tsv", type=str)
    args = parser.parse_args()
    traverse_and_create_new_tsv_file(args.directory_root, args.tsv_out)

    dir_processor.process(args.directory_in)
