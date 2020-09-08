import os
from tqdm.auto import tqdm
import glob


def parse_dataset(_file_path):
    max_len = 80
    with open(_file_path, encoding='utf-8', mode='r') as file_:
        lines = file_.read().splitlines()
    data_x, data_y = [], []
    sentence_tags = []
    for line in tqdm(lines, desc='Parsing Data'):
        if line == '':
            if sentence_tags:
                data_y.append(sentence_tags[:max_len])
                sentence_tags = []
        elif line[0] == '#':
            sentence = line.replace('# ', '')
            data_x.append([word.lower() for word in (sentence.split()[:max_len])])
            if sentence_tags:
                data_y.append(sentence_tags[:max_len])
                sentence_tags = []
        elif line[0].isdigit():
            sentence_tags.append(line.split('\t')[-1])
    return data_x, data_y



# Merge 2 tsv files
def merge_tsv_files(file_path_, new_file="new_file.tsv"):
    file_path_ = os.path.join(os.getcwd(), "Data") + "/*.tsv"
    with open(new_file, encoding="utf-8", mode="w+") as out_file:
        for in_path in glob.glob(file_path_):
            with open(in_path) as in_file:
                for line in in_file:
                    columns = line.split("\t")
                    print(line, file=out_file, end='')


# if __name__ == '__main__':
    # sentences_x, sentences_y = [], []

    # file_path_ = os.path.join(os.getcwd(), 'Data', 'train.tsv')
    # file_path_dev = os.path.join(os.getcwd(), 'Data', 'dev.tsv')
    # file_path_test = os.path.join(os.getcwd(), 'Data', 'test.tsv')
    # files_path =[file_path_, file_path_dev, file_path_test]
    # d_x, d_y = parse_dataset(file_path_)
    # sentences_x.extend(d_x)
    # sentences_y.extend(d_y)

    # d_x_d, d_y_d = parse_dataset(file_path_dev)
    # sentences_x.extend(d_x_d)
    # sentences_y.extend(d_y_d)
    # d_x_t, d_y_t = parse_dataset(file_path_test)
    # sentences_x.extend(d_x_t)
    # sentences_y.extend(d_y_t)
