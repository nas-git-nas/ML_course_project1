import numpy as np

class DataPreprocessing:
    def __init__(self, test=False, verb=False):
        """
        Preprocessing data including: loading data, convert data to numbers/nan,
        deleting column/rows that have too many nans, replacing nans with mean/median
        Args:
            test: if True only a small part of data-set is used (defined by nb_rows_to_keep)
            verb: if True results are printed
        """
        # data arrays
        self._y = None
        self._tx = None
        self._id = None

        # testing
        if test: # keep only small part of data for better visualisation
            nb_rows_to_keep = 5
        else: # keep all data
            nb_rows_to_keep = 250000 
        self._skip_footer = 250000 - nb_rows_to_keep

        # verbose
        self._verb = verb

        # load data
        path_dataset = "data/train.csv"
        self._loadData(path_dataset)

    def getData(self):
        """
        Get data
        Return:
            y: labels (N)
            tx: samples (N x D)
            id: identity numbers (N)
        """
        if self._verb:
            print(f"Labels y = \n{self._y}")
            print(f"Samples tx = \n{self._tx}")
            print(f"Identity numbers = \n{self._id}")
        
        return np.copy(self._y), np.copy(self._tx), np.copy(self._id)

    def remove(self, modus, thr):
        """
        Remove all columns/rows where procentage of valid data (that is not nan) < threshold
        Args:
            modus: can be either 'columns' (remove columns) or 'rows' (remove rows)
            thr: threshold given in %
        """
        tx = np.copy(self._tx)

        if modus == "columns": # columns are analysed
            axis = 1
            nb_all_elements = tx.shape[0]
        elif modus == "rows": # rows are analysed
            axis = 0
            nb_all_elements = tx.shape[1]           

        remove_id = []
        remove_procentage = []
        for i in range(tx.shape[axis]): # loop through all columns/rows
            if modus == "columns": # get one column
                txn = tx[:,i]
            elif modus == "rows": # get one row
                txn = tx[i,:]
            
            # calculate procentage of valid data in column/row
            valid_procentage = np.count_nonzero(~np.isnan(txn)) / nb_all_elements

            # add column/row to removing list if there is not enough valid data
            if valid_procentage < thr: 
                remove_id.append(i)
                remove_procentage.append(valid_procentage)

        # remove columns/rows
        self._tx = np.delete(tx, remove_id, axis=axis)

        if self._verb:
            print(f"Removed columns/rows: {remove_id}")
            print(f"Procentage of valid elements in removed columns/rows: {remove_procentage}")

    def replace(self, modus):
        """
        Replace all nans with either the mean or median of the column
        Args:
            modus: can be either 'mean' (replace by mean) or 'median' (replace by median)
        """
        tx = np.copy(self._tx)

        if modus == "mean": # replace nan by mean of columns
            col_rep = np.nanmean(tx, axis=0)
        elif modus == "median": # replace nan by median of columns
            col_rep = np.nanmedian(tx, axis=0)      

        nan_idx = np.where(np.isnan(tx)) # get indices of nan entries
        self._tx[nan_idx] = np.take(col_rep, nan_idx[1]) # replace nan by mean/median

        if self._verb:
            print(f"Nan indices = {nan_idx}")
            print(f"Column replacement = {col_rep}")          

    def knn(self):
        pass # TODO: implement KNN algorithm to replace nan

    def _loadData(self, path_dataset):
        """
        Load data and convert it to the metric system
        Args:
            path_dataset: path to data-set
        """
        # load labels
        y = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, skip_footer=self._skip_footer, usecols=[1], dtype=str)
        self._y = np.where(y == "s", 0, 1) # convert labels: "s" -> 0, "b" -> 1

        # load samples
        tx = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, skip_footer=self._skip_footer, usecols=np.arange(2,32))
        self._tx= np.where(tx == -999.0, np.nan, tx) # convert samples: -999.0 -> nan

        # load identity numbers
        self._id = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, skip_footer=self._skip_footer, usecols=[0])


def test_data_preprocessing():
    """
    Test class DataPreprocessing
    """
    dpp = DataPreprocessing(test=True, verb=True)
    dpp.remove(modus="columns", thr=0.4)
    dpp.replace(modus="median")

    y, tx, id = dpp.getData()
    


if __name__ == "__main__":
    test_data_preprocessing()