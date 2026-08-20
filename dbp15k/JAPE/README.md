# DBP15K source data (JAPE)

Downloaded from the official JAPE repository (ISWC 2017):
https://github.com/nju-websoft/JAPE  (data/dbp15k.tar.gz, zh_en)

Only the zh_en files used by our scripts are kept:
- s_labels / t_labels : entity labels (same content as ../data/{zh,en}_labels.tsv)
- ent_ILLs            : gold alignment (same content as ../data/gold_alignment_zh_en.tsv)
- training_attrs_1/2  : attribute triples (used for attribute-Jaccard signal)
- s_triples / t_triples : relational triples (used for degree-ratio signal)

The first three files are byte-identical (after rename) to the .tsv files in
../data/, verified by md5sum on 2026-08-20.
