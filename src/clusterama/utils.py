from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFMCS


def get_largest_fragment(mol):
    """A simple function to return the largest fragment in a molecule.  This is useful for stripping counter-ions etc. """
    frags = list(Chem.GetMolFrags(mol, asMols=True))
    frags.sort(key=lambda x: x.GetNumAtoms(), reverse=True)
    return frags[0]


def display_cluster_members(df, sel, align_mcs=True, strip_counterions=True):
    """A function to generate an image for the molecules from the selected cluster"""
    mol_list = []
    img = "Nothing selected"
    if len(sel):
        sel_df = df.query("Cluster in @sel")
        mol_list = [Chem.MolFromSmiles(x) for x in sel_df.SMILES]
        # strip counterions
        if strip_counterions:
            mol_list = [get_largest_fragment(x) for x in mol_list]
        # Align structures on the MCS
        if align_mcs and len(mol_list) > 1:
            mcs = rdFMCS.FindMCS(mol_list)
            mcs_query = Chem.MolFromSmarts(mcs.smartsString)
            AllChem.Compute2DCoords(mcs_query)
            for m in mol_list:
                AllChem.GenerateDepictionMatching2DStructure(m, mcs_query)
        legends = list(sel_df.Name.astype(str))
        img = Draw.MolsToGridImage(mol_list, molsPerRow=5, legends=legends, useSVG=True)
    return img
