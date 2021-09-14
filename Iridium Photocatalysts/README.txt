Data here gotten from https://pubs.acs.org/doi/10.1021/acscatal.0c02247, specifically the xls spreadsheet on the rate constants.

The numeric identifiers in the 'cs0c02247_si_002' data file refer to actual compounds in the paper, whose structures are given visually. An online tool was used to draw these structures and then determine their "SMILES" values, after which I manually find-replaced them to create 'Iridium_data'.

This was the correspondence I used:


CN1 - c2ccc(c1ccccn1)cc2
CN2 - Cc2ccc(c1ccccc1)nc2
CN3 - Cc2ccc(c1ccc(F)cc1)nc2
CN4 - Cc2ccc(c1ccc(Cl)cc1)nc2
CN7 - Cc2ccc(c1ccc(C#N)cc1)nc2
CN9 - COc2ccc(c1ccc(C)cn1)cc2
CN11 - Cc2ccc(c1ccc(F)cc1F)nc2
CN21 - c2ccc(n1cccn1)cc2
CN28 - c1ccc3c(c1)ccc2cccnc23
CN29 - Cc3ccc(c2ccc(c1ccccc1)cc2)nc3
CN30 - c3ccc(c2nc1ccccc1s2)cc3
CN31 - c3ccc(c2nc1ccccc1o2)cc3
CN33 - Cc3ccc(c2nc1ccccc1o2)cc3
CN34 - COc3ccc(c2nc1ccccc1s2)cc3
CN35 - COc3ccc(c2nc1ccccc1o2)cc3
CN37 - Fc3ccc(c2nc1ccccc1o2)cc3
CN38 - Clc3ccc(c2nc1ccccc1s2)cc3
CN39 - Clc3ccc(c2nc1ccccc1o2)cc3
CN40 - FC(F)(F)c3ccc(c2nc1ccccc1s2)cc3
CN41 - FC(F)(F)c3ccc(c2nc1ccccc1o2)cc3
CN44 - Cc2ccc(c1cccc(C#N)c1)nc2
CN46 - COc2cccc(c1ccc(C)cn1)c2
CN48 - Cc2ccc(c1ccc(S(C)(=O)=O)cc1)nc2
CN49 - Cc2ccc(c1ccc(OC(F)(F)F)cc1)nc2
CN54 - Cc2ccc(c1ccc(C)cn1)cc2
CN61 - Cc4ccc(c3ccc(N(c1ccccc1)c2ccccc2)cc3)nc4
CN63 - c4ccc(c3ccc(c2cc(c1ccccc1)ccn2)cc3)cc4
CN64 - Fc2ccc(c1ccccn1)cc2
CN65 - Cc2ccc(c1cccc(F)c1)nc2
CN66 - Cc2ccc(c1ccccc1F)nc2
CN67 - Cc2ccc(c1ccc(F)c(C(F)(F)F)c1)nc2
CN68 - Cc2ccc(c1cccc(Cl)c1)nc2
CN69 - c2ccc(Cc1ccccn1)cc2
CN70 - CCc2ccc(c1ccccc1)nc2
CN72 - Cc2cnc(c1ccccc1)cc2c3ccccc3
CN74 - Fc3ccc(c2cc(c1ccccc1)ccn2)cc3
CN76 - CCc2ccc(c1ccc(Cl)cc1)nc2
CN77 - Clc3ccc(c2cc(c1ccccc1)ccn2)cc3
CN83 - c4ccc(c3ccnc(c2ccc(N1CCCC1)cc2)c3)cc4
CN86 - COc3ccc(c2cc(c1ccccc1)ccn2)cc3
CN87 - COc3ccc(c2cc(c1ccccc1)c(C)cn2)cc3
CN88 - CCc2ccc(c1cccc(OC)c1)nc2
CN89 - COc3cccc(c2cc(c1ccccc1)ccn2)c3
CN90 - COc3cccc(c2cc(c1ccccc1)c(C)cn2)c3
CN91 - c4ccc(c3ccnc(c2ccc1ccccc1c2)c3)cc4
CN94 - FC(F)(F)Oc3ccc(c2nc1ccccc1s2)cc3
CN95 - c4ccc3cc(c2nc1ccccc1o2)ccc3c4
CN101 - Cc3ccc(c2ccc1ccccc1c2)nc3


NN1 - c2ccc(c1ccccn1)nc2
NN2 - Cc2ccnc(c1cc(C)ccn1)c2
NN3 - Cc2ccc(c1ccc(C)cn1)nc2
NN4 - Cc2cccc(c1cccc(C)n1)n2
NN5 - c4ccc(c3ccnc(c2cc(c1ccccc1)ccn2)c3)cc4
NN6 - CC(C)(C)c2ccnc(c1cc(C(C)(C)C)ccn1)c2
NN7 - FC(F)(F)c2ccc(c1ccc(C(F)(F)F)cn1)nc2
NN8 - COc2ccnc(c1cc(OC)ccn1)c2
NN11 - O=C(O)c2ccnc(c1cc(C(=O)O)ccn1)c2
NN14 - O=c2c(=O)c1cccnc1c3ncccc23
NN16 - c1cnc3c(c1)ccc2cccnc23
NN19 - O=c2c1cccnc1c3ncccc23
NN20 - Cc3cnc2c(ccc1c(C)c(C)cnc12)c3C
NN21 - c4ccc3nc(c2ccc1ccccc1n2)ccc3c4
NN23 - Cc5cc(c1ccccc1)c4ccc3c(c2ccccc2)cc(C)nc3c4n5
NN24 - c5ccc(c1ccnc4c1ccc3c(c2ccccc2)ccnc34)cc5
NN26 - Cc2cc1cccnc1c3ncccc23
NN27 - CC1(C)C(=O)C(C)(C)c3c1c2cccnc2c4ncccc34
NN33 - Cc1ccnc3c1ccc2c(C)ccnc23
NN34 - Cc3ccc2ccc1ccc(C)nc1c2n3
NN43 - CCCCCCCCCc2ccnc(c1cc(CCCCCCCCC)ccn1)c2
NN44 - Cc1ccnc3c1ccc2cccnc23
NN46 - CS(C)=O
NN47 - CC(C)(O)Cc2ccc(c1ccc(CC(C)(C)O)cn1)nc2