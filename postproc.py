#! /usr/bin/env python3

# Questo script utilizza la libreria "nbformat" per leggere e scrivere programmaticamente
# dei notebook Jupyter. La utilizziamo per creare uno script che prenda un notebook
# sorgente con delle annotazioni relative agli esercizi e crei due nuove copie del
# notebook rispettivamente con ("sol") e senza ("nosol") le soluzioni e un file di testo
# con le sole soluzioni.


from pathlib import Path

import nbformat


MAGIC_PREFIX = "##"


def load_notebook(nbfile):
    with open(nbfile, "r") as f:
        return nbformat.read(f, nbformat.NO_CONVERT)

def save_notebook(nb, nbfile):
    with open(nbfile, "w") as f:
        nbformat.write(nb, f)

def process_notebook(nb):
    nb_nosol = nbformat.v4.new_notebook(nbformat_minor=4)
    nb_sol = nbformat.v4.new_notebook(nbformat_minor=4)
    txt_sol = []
    for nnb in (nb_nosol, nb_sol):
        nnb.metadata.language_info = nb.metadata.language_info
    for cell in nb.cells:
        write_to = [nb_nosol, nb_sol]
        if cell.cell_type == "code" and cell.source.startswith(MAGIC_PREFIX):
            magic, _, cell.source = cell.source.partition("\n")
            magic = magic[len(MAGIC_PREFIX):].split(":")
            if magic[0] == "sol":
                write_to = [nb_sol]
                if len(magic) > 1:
                    txt_sol.append("# {}\n{}".format(magic[1], cell.source))
                else:
                    txt_sol.append(cell.source)
            elif magic[0] == "nosol":
                write_to = [nb_nosol]
            elif magic[0] == "solhead":
                write_to = []
                txt_sol.append("{}# {}".format("\n" if txt_sol else "", magic[1]))
            elif magic[0] == "outsnip":
                write_to = [nb_nosol, nb_sol]
                for cout in cell.outputs:
                    if cout.name == "stdout" and cout.output_type == "stream":
                        lines = cout.text.split("\n")
                        new_lines = lines[:int(magic[1])] + ["[...]"] + lines[-int(magic[2]):]
                        cout.text = "\n".join(new_lines)
        cell.pop("id", None)
        for dnb in write_to:
            dnb.cells.append(cell)
    return nb_nosol, nb_sol, "\n\n".join(txt_sol)


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("notebook_file")
    args = parser.parse_args()
    input_path = Path(args.notebook_file)
    base_file_name = input_path.stem
    nosol_path = input_path.parent / (base_file_name + ".nosol.ipynb")
    sol_path = input_path.parent / (base_file_name + ".sol.ipynb")
    input_nb = load_notebook(input_path.resolve())
    nosol_nb, sol_nb, sol_txt = process_notebook(input_nb)
    save_notebook(nosol_nb, nosol_path)
    save_notebook(sol_nb, sol_path)
    with open(input_path.parent / (base_file_name + ".sol.py"), "w") as f:
        f.write(sol_txt)


if __name__ == "__main__":
    main()
