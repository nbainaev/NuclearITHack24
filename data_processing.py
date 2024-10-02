import numpy as np
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

def mol_dsc_calc(mols):
    return pd.DataFrame({k: f(Chem.MolFromSmiles(m)) for k, f in descriptors.items()} for m in mols.values)

# список конституционных и физико-химических дескрипторов из библиотеки RDKit
descriptors = {"HeavyAtomCount": Descriptors.HeavyAtomCount,
               "NHOHCount": Descriptors.NHOHCount,
               "NOCount": Descriptors.NOCount,
               "NumHAcceptors": Descriptors.NumHAcceptors,
               "NumHDonors": Descriptors.NumHDonors,
               "NumHeteroatoms": Descriptors.NumHeteroatoms,
               "NumRotatableBonds": Descriptors.NumRotatableBonds,
               "NumValenceElectrons": Descriptors.NumValenceElectrons,
               "NumAromaticRings": Descriptors.NumAromaticRings,
               "NumAliphaticHeterocycles": Descriptors.NumAliphaticHeterocycles,
               "RingCount": Descriptors.RingCount,
               "MW": Descriptors.MolWt,
               "LogP": Descriptors.MolLogP,
               "MR": Descriptors.MolMR,
               "TPSA": Descriptors.TPSA,
               "Molecular Weight": Descriptors.MolWt}

def rdkit_fp(smiles_column: pd.Series, radius=3, nBits=2048, useChirality=False):
    # morganFP_rdkit
    def desc_gen(mol):
        mol = Chem.MolFromSmiles(mol)
        bit_vec = np.zeros((1,), np.int16)
        DataStructs.ConvertToNumpyArray(
            AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, useChirality=useChirality), bit_vec)
        return bit_vec

    return pd.DataFrame.from_records(smiles_column.apply(func=desc_gen), columns=[f'bit_id_{i}' for i in range(nBits)])


def rdkit_2d(smiles_column: pd.Series):
    # 2d_rdkit
    descriptors = {i[0]: i[1] for i in Descriptors._descList}
    return pd.DataFrame({k: f(Chem.MolFromSmiles(m)) for k, f in descriptors.items()} for m in smiles_column)

def extract_smiles(raw_data: pd.DataFrame, smiles: pd.Series, add_bit_vec: bool=True, add_2d_rdkit: bool=False) -> pd.DataFrame:

    data = raw_data.copy()
    columns = data.columns

    if add_bit_vec:
        Y = rdkit_fp(smiles)
        data = data.join(Y)
    
    descriptors_transformer = FunctionTransformer(mol_dsc_calc)
    X = descriptors_transformer.transform(smiles)
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)

    data = data.join(X)
    if add_2d_rdkit:
        Z = rdkit_2d(smiles)
        data = data.join(Z)

    return data


def process(data: pd.DataFrame) -> pd.DataFrame:
    smiles = data["Smiles"]
    data = data[["Cell", "Strain", "Smiles", "DOI"]]
    data = extract_smiles(data, smiles)

    impute_cols = ["Cell", "Strain", "Smiles", "DOI"]
    imputer = SimpleImputer()
    imputer = joblib.load('imputer.joblib')
    data[impute_cols] = pd.DataFrame(imputer.transform(data[impute_cols]), columns=impute_cols)

    return data.drop(columns=["Smiles", "DOI"])
