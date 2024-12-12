import pickle

class Atom_File_handler():
    def __init__(self) -> None:
        pass
    
    def save_atom_collections(self, atom_cols, filename):
        try:
            file_obj = open(filename, "xb")
        except:
            file_obj = open(filename, "ab")
        pickle.dump(atom_cols, file=file_obj)

    def load_atom_collections(self, filename):
        try:
            file_obj = open(filename, "rb")
        except:
            raise Exception(f"The specified file: {filename} : Does not exist")
        atom_cols = pickle.load(file_obj)
        return atom_cols