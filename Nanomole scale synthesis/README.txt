Data here is gotten from 'https://www.science.org/doi/full/10.1126/science.1259203?versioned=true', specifically the '1259203_datafiles.xlsx' file. From this, the S3 Data file was used.


The numeric identifiers in the '1259203_datafiles.xlsx' file refer to actual compounds in the paper, whose structures are given visually. An online tool was used to draw these structures and then determine their "SMILES" values, after which I manually find-replaced them to create 'S3_data'.

This was the correspondence I used:

Electrophiles
2 - O=C2OC(Cn1ccnn1)CN2c3ccc(I)c(F)c3
4 - COCCCc4cc(CN(C(=O)C1CN(C(=O)OC(C)(C)C)CCC1c2ccn(C)c(=O)c2)C3CC3)cc(Br)c4C
5 - CN/2C(=O)CC(C)(c1cc(Br)cs1)NC2=N/C(=O)OC(C)(C)C
6 - CC(C)n3c(C(=O)N(C)C)c2CCN(Cc1ccc(F)c(Cl)c1)C(=O)c2c(O)c3=O
7 - CN4CCN(C(=O)O[C@H]2c1nccnc1C(=O)N2c3ccc(Cl)cn3)CC4
8 - COC(=O)C(C)(C)Cc4c(SC(C)(C)C)c3cc(OCc2cc1ccccc1cn2)ccc3n4Cc5ccc(Cl)cc5

Nucleophiles
9 - CCOC(=O)N1CCNCC1
10 - COC(=O)C1(CN)CC1
11 - CC(C)(C)OC(N)=O
12 - Nc1cc(F)ccn1
13 - NS(=O)(=O)c1cccs1
14 - CN(C)CC(N)=O
15 - N=C(N)C1CC1
16 - Cn1nccc1CCO
17 - O
19 - Cn1nccc1[B]2OC(C)(C)C(C)(C)O2
20 - C#Cc1cnccn1


Catalysts
36 - CC(C)c4cc(C(C)C)c(c1ccccc1P(C2CCCCC2)C3CCCCC3)c(C(C)C)c4
38 - CC(C)(C)P(c1cc[cH-]c1)C(C)(C)C.CC(C)(C)P(c1cc[cH-]c1)C(C)(C)C.[Fe+2]
42 - CC(C)c2cc(C(C)C)c(c1ccccc1P(C(C)(C)C)C(C)(C)C)c(C(C)C)c2
43 - COc1ccc(OC)c(P(C(C)(C)C)C(C)(C)C)c1c2c(C(C)C)cc(C(C)C)cc2C(C)C
44 - COc2ccc(C)c(c1c(C(C)C)cc(C(C)C)cc1C(C)C)c2P(C(C)(C)C)C(C)(C)C
45 - COc7ccc(OC)c(P(C23CC1CC(CC(C1)C2)C3)C56CC4CC(CC(C4)C5)C6)c7c8c(C(C)C)cc(C(C)C)cc8C(C)C


Bases
S17 - CN(C)P(=NC(C)(C)CC(C)(C)C)(N(C)C)N(C)C
S18 - CN(C)P(=NC(C)(C)C)(N=P(N(C)C)(N(C)C)N(C)C)N(C)C
24 - C2CCC1=NCCCN1CC2
25 - CN1CCCN2CCCN=C12
26 - CN(C)/C(=N\C(C)(C)C)N(C)C
27 - CCN(CC)P1(=NC(C)(C)C)NCCCN1C
28 - CC(C)(C)N=P(N1CCCC1)(N2CCCC2)N3CCCC3
29 - CCN=P(N=P(N(C)C)(N(C)C)N(C)C)(N(C)C)N(C)C





Now, by multiplying together all combinations of electrophiles, nucleophiles, catalysts and bases, you get 6 x 11 x 6 x 8 = 3168 possible configurations. However, the S3 data file only contains about 1500.

Initially, I simply input these as the components for the EDBO optimiser, which then constructed the 3168-sized reaction space. I gave a default value of 0 for the area count. However, this lead to quite poor optimiser performance, likely due to the high amount of 0s present.


So, in the 'fixed' files, I instead directly took the 1500 or so configurations recorded, and made them the reaction space, which improved performance.