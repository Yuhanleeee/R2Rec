from torch.utils.data import Dataset, ConcatDataset
import pickle


def add_datasets_args(parent_args):
    parser = parent_args.add_argument_group('img seq data args')
    parser.add_argument("--datasets_path_train", type=str, default=None, required=True, nargs='+', help="A folder containing the training data of instance sequence.",)
    parser.add_argument("--datasets_path_val", type=str, default=None, required=True, nargs='+', help="A folder containing the val data of instance sequence.",)
    parser.add_argument("--datasets_path_test", type=str, default=None, required=True, nargs='+', help="A folder containing the test data of instance sequence.",)
    return parent_args


class PKLDataset(Dataset):
    def __init__(self, data_raw):
        self.data_raw = data_raw

    def __len__(self):
        return len(self.data_raw)

    def __getitem__(self, idx):
        data_temp = self.data_raw[idx]
        return {idx: data_temp}


def process_pool_read_dataset(input_path, data_argu_flag=False):
    with open(input_path[0], 'rb') as f:
        data_raw = pickle.load(f)
    if data_argu_flag:
        data_raw = data_argument(data_raw=data_raw)
    pkl_dataset = PKLDataset(data_raw) 
    return pkl_dataset


def data_argument(data_raw):
    data_lists = []
    for data_temp in data_raw:
        for i in range(len(data_temp)-1):
            data_lists.append(data_temp[0:i+2])
    return data_lists


def load_data(args):
    data_train = process_pool_read_dataset(args.datasets_path_train, data_argu_flag=True)
    data_val = process_pool_read_dataset(args.datasets_path_val, data_argu_flag=False)
    data_test = process_pool_read_dataset(args.datasets_path_test, data_argu_flag=False)
    return {'train': data_train, 'val': data_val, 'test': data_test}
     

def main():
    pass


if __name__ == "__main__":
    main()
