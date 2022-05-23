#!/usr/bin/env python3

# Questo script utilizza la libreria nbformat per leggere e scrivere programmaticamente
# dei notebook Jupyter. La utilizziamo per leggere un notebook sorgente con delle
# annotazioni relative agli esercizi e creare due nuove copie del notebook
# rispettivamente con ("sol") e senza ("nosol") le soluzioni e un file di testo con le
# sole soluzioni.

##sol[:<commento>] = mostra cella solamente nelle soluzioni [col commento indicato]
##nosol = mostra cella solamente nel notebook senza soluzioni
##solhead:<titolo> = inserisci un titolo della sezione nel file con le sole soluzioni
##outsnip:<X>:<Y> = mantieni solo le prime X e le ultime Y righe dell'output
##noerr = non mostrare gli errori


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
                    cell.source = f"# {magic[1]}\n{cell.source}"
                txt_sol.append(cell.source)
            elif magic[0] == "nosol":
                write_to = [nb_nosol]
            elif magic[0] == "solhead":
                write_to = []
                sep = "\n" if txt_sol else ""
                txt_sol.append(f"{sep}# {magic[1]}")
            elif magic[0] == "outsnip":
                write_to = [nb_nosol, nb_sol]
                for cout in cell.outputs:
                    if cout.output_type == "stream" and cout.name == "stdout":
                        lines = cout.text.split("\n")
                        new_lines = lines[:int(magic[1])] + ["[...]"] + lines[-int(magic[2]):]
                        cout.text = "\n".join(new_lines)
            elif magic[0] == "noerr":
                write_to = [nb_nosol, nb_sol]
                cell.outputs = [
                    out for out in cell.outputs if out.output_type != "stream" or out.name != "stderr"
                ]
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
