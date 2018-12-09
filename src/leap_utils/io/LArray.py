import logging
import copy
import warnings
import numpy as np


class LArray(np.ndarray):
    """Labelled array."""

    # TODO: nicer __repr__/__str__ -
    # TODO: proper tests - for empy row/col names, wrong input shapes
    # TODO: DONE error messages if trying to do string indexing with empy col_names or row selection with empty row_names
    # TODO: concatenate - concat row_names when concatenating two LArrays
    # TODO: add row (or col)

    def __new__(cls, input_array, col_names=[], row_names={}):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        if col_names and len(col_names) != obj.shape[1]:
            raise ValueError(f'Number of column names {len(col_names)} must match number of array columns {obj.shape[1]}.')

        col_names_str = [isinstance(col_name, str) for col_name in col_names]
        if col_names and not all(col_names_str):
            # TODO: more informative error message - report which rows are the wrong length
            raise ValueError(f'Column names must be strings.')
        obj.col_names = col_names

        bad_row_lens= [len(row_name)!=obj.shape[0] for key, row_name in row_names.items()]
        if row_names and any(bad_row_lens):
            # TODO: more informative error message - report which rows are the wrong length
            raise ValueError(f'Length of each row must match number of array rows {obj.shape[0]}.')
        obj.row_names = row_names

        return obj

    def __array_finalize__(self, obj):
        # see col_namesArray.__array_finalize__ for comments
        if obj is None:
            return
        self.col_names = getattr(obj, 'col_names', None)
        self.row_names = getattr(obj, 'row_names', None)

    def __str__(self) -> str:
        """Represent table as string for printing."""
        if len(self.shape) == 3:
            s = print_table(col_values=self.asarray(),
                            col_headers=self.col_names,
                            row_values=np.array(list(self.row_names.values())).T,
                            row_headers=list(self.row_names.keys()),
                            last_row=min(self.shape[0], 8))
        else:
            s = super().__str__()
        return s

    def print(self, first_row: int = 0, last_row: int = None) -> None:
        """Print table.

        Args:
            first_row: defaults to first row
            last_row: defaults to last row in table

        """
        if len(self.shape) == 3:
            ss = print_table(col_values=self,
                             col_headers=self.col_names,
                             row_values=np.array(list(self.row_names.values())).T,
                             row_headers=list(self.row_names.keys()),
                             first_row=first_row,
                             last_row=last_row)
        else:
            ss = super().__str__(self)
        print(ss)

    def print_head(self):
        """Print first 5 rows."""
        self.print(first_row=0, last_row=5)

    def print_tail(self):
        """Print last 5 rows."""
        self.print(first_row=max(self.shape[0]-5, 8), last_row=self.shape[0])

    def __getitem__(self, key):
        new_key = self._process_key(key)
        x = copy.deepcopy(super(LArray, self).__getitem__(new_key))
        if x.shape:
            new_row_names = {}
            for key, val in x.row_names.items():
                new_row_names[key] = val[new_key[0]]
            x.row_names = new_row_names

            if x.col_names:
                x.col_names = np.array(x.col_names)[new_key[1]].tolist()
        return x

    def __setitem__(self, key, val):
        new_key = self._process_key(key)
        super(LArray, self).__setitem__(new_key, val)


    def asarray(self):
        return np.array(self.data)


    def _process_key(self, key):
        if isinstance(key, (str, tuple, list)) and all([isinstance(key_str, str) for key_str in key]):
            logging.debug(f'got multiple strings: {key} - transforming to indices.')
            new_key = self.idx(key)
        if isinstance(key, (dict, )):
            logging.debug(f'   row selection with dict: {key}.')
            new_key = (self._match_rows(**key), Ellipsis)
        elif isinstance(key, tuple):
            logging.debug(f'got tuple from sliced index: {key} - processing all items.')
            key_sl = []
            for key_num, sl in enumerate(key):
                if key_num == 0:
                    if isinstance(sl, (dict, )):
                        logging.debug(f'   row selection with dict: {sl}.')
                        key_sl.append(self._match_rows(**sl))
                    else:
                        logging.debug(f'   row selection standard: {sl}.')
                        key_sl.append(sl)
                else:
                    if isinstance(sl, (str, tuple, list)) and all([isinstance(key_str, str) for key_str in sl]):
                        logging.debug(f'   dim selection with single string or multiple strings: {sl}.')
                        slm = self.idx(sl)
                        if not slm:
                            slm = slice(0, 0)
                        key_sl.append(slm)
                    elif isinstance(sl, slice):
                        logging.debug(f'   dim selection with slice: {sl}.')
                        slice_stop = self.idx(sl.stop)
                        if slice_stop and isinstance(sl.stop, str):
                            logging.debug(f'       using inclusive bounds when indexing string slice {slice_stop} "{sl.stop}".')
                            slice_stop = slice_stop + 1

                        slice_points = [self.idx(sl.start), slice_stop, self.idx(sl.step)]
                        if [] in slice_points:
                            slice_points = (0, 0)
                        key_sl.append(slice(*slice_points))
                    else:
                        logging.debug(f'   dim selection with something else: {sl}.')
                        # key_sl.append(slice(sl, sl+1))  # preserves dims
                        key_sl.append(sl)  # removes singleton dims
            logging.debug(f'tuple transformed to: {key_sl}.')
            for key in key_sl:
                if isinstance(key, str):
                    logging.debug(f'{key} is still a string')
            new_key = tuple(key_sl)
        else:
            # simple numpy integer index access
            new_key = key
        return new_key


    def idx(self, key):
        """Get index for key or list of keys."""
        if key is None or isinstance(key, (int, slice, type(Ellipsis))) :  # these are valid indices
            logging.debug(f'      idx got valid key {key}')
            return key
        else:
            logging.debug(f'    idx tries to match invalid key {key}')
            if len(self.col_names) == 0:
                warnings.warn('matching w/o col_names returns empty matches')

            try:  # to use key directly
                return self.col_names.index(key)
            except ValueError:  # key not in col
                pass

            try:  # maybe it's an iterable of keys
                return [self.col_names.index(k) for k in key]

            except TypeError:  # not iterable
                pass
            except ValueError:
                pass
            return []  # empty

    def rows(self, *args, **kwargs):
        """Select rows.

        Args:
            args - positional
            kwargs - by row name
        """
        matches = self._match_rows(*args, **kwargs)
        return self[matches,...]

    def _match_rows(self, *args, **kwargs):
        # get row_name for args
        if not self.row_names:
            warnings.warn('matching w/o row_names returns empty matches')
            matches = np.zeros((self.shape[0],)).astype(np.bool)
        else:
            matches = np.ones((self.shape[0],)).astype(np.bool)
            logging.debug(f'transforming {args} into kwargs.')
            for pos, val in enumerate(args):
                key = list(self.row_names.keys())[pos]
                kwargs[key] = val
            logging.debug(f'   new kwargs: {kwargs}.')
            matches = np.ones((self.shape[0],)).astype(np.bool)
            for key, val in kwargs.items():
                cond = np.zeros((self.shape[0],)).astype(np.bool)
                logging.debug(f'      matching {key}:{val}')
                for v in self._tuplify(val):
                    cond = np.logical_or(cond, self.row_names[key] == [v])
                matches = np.logical_and(matches, cond)  # combine rows via AND
        return matches

    def _tuplify(self, val):
        if isinstance(val, range):
            val = tuple(val)
        elif isinstance(val, (str, int, float)):
            val = (val, )
        return val


def print_table(col_values, col_headers=None, row_values=None, row_headers=None, first_row=0, last_row=None):
    """Convert table to string."""
    s = []
    if last_row is None:
        last_row = col_values.shape[0]

    if row_headers is not None:
        for hd in row_headers:
            s.append(f'{hd:<10}')
        s.append(f'|  ')
    elif col_headers is not None and row_values is not None:
        s.append(' '*10*row_values.shape[1] + f'|  ')  # fill white space

    if col_headers is not None:
        for hd in col_headers:
            s.append(f'{hd:<10}')

    # print line separating headers and data
    if row_headers is not None or col_headers is not None:
        row_width = 0
        if row_values is not None:
            row_width = row_values.shape[1]
        if row_headers is not None:
            row_width = max(row_width, row_values.shape[1])
        s.append('\n')
        s.append('-'*10*(row_width+col_values.shape[1]))

    if row_headers is not None or col_headers is not None:
        s.append('\n')

    for row in range(first_row, min(col_values.shape[0], last_row)):
        # row data
        if row_values is not None:
            for col in range(row_values.shape[1]):
                if row == first_row or row_values[row, col] != row_values[row-1, col]:
                    s.append(f'{row_values[row, col]:<10}')
                else:
                    s.append(f"{'':<10}")
            s.append(f'|  ')  # row-col separator

        # column data
        for col in range(col_values.shape[1]):
            s.append(f'{str(np.array(col_values[row, col,:])):<10}')
        s.append('\n')

    # report rows printed
    if first_row > 0 or last_row < col_values.shape[0]:
        s.append(f'showing rows {first_row}:{last_row} out of {col_values.shape[0]}.\n')
    else:
        s.append(f'showing all {col_values.shape[0]} rows.\n')
    s.append('\n')
    return ''.join(s)


def test_tuplify(la):
    print('bool:', la._tuplify(True))
    print('int:', la._tuplify(1))
    print('float', la._tuplify(1.0))
    print('range', la._tuplify(range(3)))
    print('tuple', la._tuplify(tuple(range(3))))
    print('nparray', la._tuplify(np.arange(3)))


def test_get(la):
    print('la[:, 11, :] -> (20, 2) == ', la[:, 11, :].shape)
    print('la[:, "tail", :] -> (20, 2) == ', la[:, 'tail', :].shape)
    print('la[:, "WRONG", :] -> (20, 0, 2) == ', la[:, 'WRONG', :].shape)
    print('la[:, [4, 6], :] -> (20, 2, 2) == ', la[:, [4, 6], :].shape)
    print('la[:, [4, 6]] -> (20, 2, 2) == ', la[:, [4, 6]].shape)

    print("la[::2, ('tail', 'head')] -> (10, 2, 2)", la[::2, ('tail', 'head')].shape)
    print("la[::2, ['tail', 'head']] -> (10, 2, 2)", la[::2, ['tail', 'head']].shape)

    print('----------- ROWS ----------------')
    print("la.rows('rec2') -> (10, 12, 2) ==", la.rows('rec2').shape)
    print("la.rows(fly=1) -> (7, 12, 2) ==", la.rows(fly=1).shape)
    print("la.rows('rec2', fly=1) -> (2, 12, 2) ==", la.rows('rec2', fly=1).shape)
    print("la.rows('rec1', fly=1) -> (5, 12, 2) ==", la.rows('rec1', fly=1).shape)
    print("la.rows('rec1', fly=[0, 1]) -> (10,12,2) ==", la.rows('rec1', fly=[0, 1]).shape)
    print("la.rows(('rec1', 'rec2'), fly=[0, 1]) -> (14,12,2) ==", la.rows(('rec1', 'rec2'), fly=[0, 1]).shape)
    print("la.rows(['rec1', 'rec2'], fly=[0, 1]) -> (14,12,2) ==", la.rows(['rec1', 'rec2'], fly=[0, 1]).shape)
    print("la.rows(frame=range(10), fly=[0, 1]) -> (14, 12, 2) ==", la.rows(frame=range(10), fly=[0, 1]).shape)

    print('----------- ELLIPSES -------------')
    print("la[:2, ...] -> (2,12,1) ==", la[:2, ...].shape)
    print("la[::2, ...] -> (10,12,1) ==", la[::2, ...].shape)
    print("la[2:8:2, ...] -> (3,12,1) ==", la[2:8:2, ...].shape)
    print("la[:, 'tail', ...] -> (20,2) ==", la[:, 'tail', ...].shape)
    print("la[:, 'backR':'tail', ...] -> (20,4,2) ==", la[:, 'backR':'tail', ...].shape)

    mask = np.zeros((20,)).astype(np.bool)
    mask[[1, 4, 5]] = True
    print("la[mask, 'tail', ...] -> (3, 2) ==", la[mask, 'tail', ...].shape)

    print('----------- FANCY ROW MATCHING ---')
    print(la[{'filename': ('rec1', 'rec2'), 'frame': (0, 1, 3)}].shape)
    print(la.rows(('rec1', 'rec2'), (0, 1, 3)).shape)
    print(la.rows(('rec1', 'rec2'), frame=(0, 1, 3)).shape)
    print(la[{'filename': ('rec1', 'rec2'), 'frame': (0, 1, 3)}, ...].shape)


def test_set(lax):
    lax[0, ...] = np.zeros((12, 2))


def test_print(la):
    col_headers = la.col_names
    col_values = la
    row_headers = list(la.row_names.keys())
    row_values = np.array(list(la.row_names.values())).T
    print(print_table(col_values, col_headers, row_values, row_headers, last_row=5))
    print(print_table(col_values, col_headers, row_values, row_headers=None, last_row=5))
    print(print_table(col_values, col_headers, row_values=None, row_headers=None, last_row=5))
    print(print_table(col_values, col_headers=None, row_values=None, row_headers=None, last_row=5))
    print(print_table(col_values, col_headers=None, row_values=None, row_headers=None, last_row=5))
    print(print_table(col_values, col_headers=None, row_values=row_values, row_headers=None, last_row=5))
    print(print_table(col_values, col_headers=None, row_values=row_values, row_headers=row_headers, last_row=5))

    la.print()
    la.print_head()
    la.print_tail()
    print(la)


if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG)
    warnings.simplefilter('default')

    body_parts = ['head', 'neck', 'frontL', 'middleL', 'backL', 'frontR', 'middleR', 'backR', 'thorax', 'wingL', 'wingR',
                  'tail']
    positions = np.random.randint(0, 100, (20, len(body_parts), 2))
    row_names = {'filename': np.array(['rec1'] * 10 + ['rec2'] * 10), 'frame': np.array(list(range(10)) * 2),
                 'fly': np.array([0, 1] * 5 + [0, 1, 2, 3, 4] * 2)}

    la = LArray(positions, col_names=body_parts, row_names=row_names)
    # add row index
    row_sex = {'sex': np.array(['male', 'female'] * 5 + ['male', 'male', 'male', 'male', 'female'] * 2)}
    la.row_names.update(row_sex)

    test_tuplify(la)
    test_get(la)
    test_get(LArray(positions, col_names=body_parts))
    test_get(LArray(positions, row_names=row_names))
    test_get(LArray(positions))
    test_set(la)
    test_print(la)
